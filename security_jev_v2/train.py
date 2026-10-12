"""Fine tune v2 BERT, select weights on validation, then calibrate separately."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

# Data, model and calibration are separate because each split has one job.
from security_jev_v2.data import LengthBucketBatchSampler, collate, encode_rows, label_report, load_rows, token_report
# The same tie-aware PR-AUC is used for validation selection and final reports.
from security_jev_v2.metrics import average_precision
from security_jev_v2.model import SecurityJevBert, joint_loss, save_bundle
from security_jev_v2.policy import calibrate, choose_policy, probabilities, report_predictions


def device_name(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_bytes(device: str) -> int:
    if device == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.startswith("cuda"):
        return int(torch.cuda.max_memory_allocated())
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def score_rows(model, tokenizer, rows: list[dict], batch_size: int, device: str) -> dict:
    loader = DataLoader(rows, batch_size=batch_size, collate_fn=lambda group: collate(group, tokenizer))
    block_logits, type_logits = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            block_logits.extend(output["block"].cpu().tolist())
            type_logits.extend(output["types"].cpu().tolist())
    return {"block_logits": block_logits, "type_logits": type_logits,
            "block_y": [row["block_y"] for row in rows],
            "type_y": [row["type_y"] for row in rows],
            "sites": [row["site"] for row in rows]}


def _report(scores: dict, calibration: dict, threshold: float) -> dict:
    block, all_types = probabilities(scores["block_logits"], scores["type_logits"], calibration)
    attacked = [i for i, label in enumerate(scores["block_y"]) if label == 1]
    return report_predictions(block, scores["block_y"], [all_types[i] for i in attacked],
                              [scores["type_y"][i] for i in attacked], scores["sites"], threshold)


def train_model(data: Path, output: Path, model_source: str = "google-bert/bert-base-cased",
                max_length: int = 512, batch_size: int = 4, accum: int = 4,
                epochs: int = 3, encoder_lr: float = 2e-5, head_lr: float = 1e-4,
                max_steps: int | None = None, benchmark_steps: int = 100,
                device: str = "auto") -> dict:
    """Keep test sealed; validation chooses weights and calibration fixes display scores."""
    if min(batch_size, accum, epochs) < 1 or not 4 <= max_length <= 512:
        raise ValueError("invalid training configuration")
    torch.manual_seed(42)
    device = device_name(device)
    raw = {name: load_rows(data / f"{name}.jsonl") for name in ("train", "val", "calib")}
    tokenizer_source = str(Path(model_source) / "tokenizer") if (Path(model_source) / "tokenizer").is_dir() else model_source
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_source)
    encoded = {name: encode_rows(rows, tokenizer, max_length) for name, rows in raw.items()}
    preflight = {name: {"tokens": token_report(encoded[name]), "labels": label_report(raw[name])}
                 for name in encoded}
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model = SecurityJevBert.from_pretrained(model_source).to(device)
    if tokenizer.vocab_size != model.encoder.config.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the BERT encoder")
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": list(model.block.parameters()) + list(model.types.parameters()), "lr": head_lr},
    ], weight_decay=.01)
    bucket_pool = math.ceil(512 / batch_size) * batch_size
    evaluation_batch_size = max(batch_size, 16)
    loader = DataLoader(encoded["train"],
                        batch_sampler=LengthBucketBatchSampler(encoded["train"], batch_size, pool_size=bucket_pool),
                        collate_fn=lambda group: collate(group, tokenizer))
    full_updates = math.ceil(len(loader) / accum) * epochs
    planned = min(full_updates, max_steps) if max_steps is not None else full_updates
    warmup = math.ceil(planned * .06)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        (step + 1) / max(warmup, 1) if step < warmup else
        max(0.0, (planned - step) / max(1, planned - warmup)))
    step, best_key, best_state = 0, (-1.0, -1.0), None
    timings, memories = [], []
    for epoch in range(epochs):
        model.train()
        iterator = iter(loader)
        for start in range(0, len(loader), accum):
            started = time.perf_counter()
            group = [next(iterator) for _ in range(min(accum, len(loader) - start))]
            block_total = sum(batch["block_y"].numel() for batch in group)
            type_total = sum(int((batch["type_y"] >= 0).sum()) for batch in group)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for batch in group:
                result = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                losses = joint_loss(result, batch["block_y"].to(device), batch["type_y"].to(device))
                type_count = int((batch["type_y"] >= 0).sum())
                weighted = losses["block"] * (len(batch["block_y"]) / block_total)
                if type_total:
                    weighted = weighted + losses["types"] * (type_count / type_total)
                weighted.backward()
                loss_value += float(weighted.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            synchronize(device)
            step += 1
            if len(timings) < benchmark_steps:
                timings.append(time.perf_counter() - started)
                memories.append(memory_bytes(device))
            if step % 10 == 0 or step == planned:
                print(f"v2 update {step}/{planned} loss={loss_value:.4f}", flush=True)
            if step >= planned:
                break
        validation = score_rows(model, tokenizer, encoded["val"], evaluation_batch_size, device)
        block = torch.sigmoid(torch.tensor(validation["block_logits"])).tolist()
        choice = choose_policy(block, validation["block_y"])
        ap = average_precision(block, validation["block_y"])
        key = (choice["recall"] or 0, ap or 0)
        print(f"v2 epoch={epoch + 1} validation_recall={key[0]:.4f} ap={key[1]:.4f}", flush=True)
        if key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            selected_epoch, selected_step = epoch + 1, step
        if step >= planned:
            break
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    calib = score_rows(model, tokenizer, encoded["calib"], evaluation_batch_size, device)
    known = [i for i, y in enumerate(calib["type_y"]) if y >= 0]
    scaling = calibrate(calib["block_logits"], calib["block_y"],
                        [calib["type_logits"][i] for i in known], [calib["type_y"][i] for i in known])
    block_scores, _ = probabilities(calib["block_logits"], [], scaling)
    policy = {**scaling, **choose_policy(block_scores, calib["block_y"])}
    benchmark = {"steps_observed": len(timings), "median_seconds_per_update": statistics.median(timings),
                 "estimated_full_seconds": statistics.median(timings) * full_updates,
                 "max_sampled_memory_bytes": max(memories), "device": device,
                 "memory_note": "post-update accelerator sample; not guaranteed instantaneous peak"}
    manifest = {"model_version": f"bert-security-v2-{int(time.time())}", "model_source": model_source,
                "encoder_initialization": "original_pretrained_bert",
                "max_length": max_length, "selected_epoch": selected_epoch,
                "selected_step": selected_step, "benchmark": benchmark, "preflight": preflight,
                "training": {"batch_size": batch_size, "accum": accum, "epochs": epochs,
                             "encoder_lr": encoder_lr, "head_lr": head_lr, "planned_updates": planned,
                             "seed": 42, "gradient_checkpointing": True,
                             "length_bucketing": {"pool_size": bucket_pool},
                             "evaluation_batch_size": evaluation_batch_size},
                "data_sha256": {name: sha256_file(data / f"{name}.jsonl")
                                for name in ("train", "val", "calib", "test")}}
    source_report = data / "report.json"
    if source_report.is_file():
        frozen_data = json.loads(source_report.read_text(encoding="utf-8"))
        manifest["data_summary"] = {key: frozen_data.get(key) for key in
                                    ("counts", "sources", "wcp", "redaction_version", "test_minimums_met")}
    save_bundle(model, tokenizer, output, manifest, policy)
    report = {"model_version": manifest["model_version"], "validation_selection": best_key,
              "calibration": _report(calib, policy, policy["threshold"]),
              "policy": policy, "benchmark": benchmark, "test_used": False}
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/v2"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bert-base-v2"))
    parser.add_argument("--model", default="google-bert/bert-base-cased")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--benchmark-steps", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(train_model(args.data, args.output, args.model, args.max_length,
                                  args.batch_size, args.accum, args.epochs, args.encoder_lr,
                                  args.head_lr, args.max_steps, args.benchmark_steps, args.device),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

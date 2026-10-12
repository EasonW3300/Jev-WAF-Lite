"""Evaluate a frozen v2 bundle exactly once on the untouched test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The evaluator loads only local weights and verifies the frozen data digest.
from security_jev_v2.data import encode_rows, label_report, load_rows, token_report
from security_jev_v2.model import TYPE_NAMES, load_bundle
from security_jev_v2.train import _report, device_name, score_rows, sha256_file


def evaluate_bundle(bundle: Path, data: Path, batch_size: int = 4,
                    device: str = "cpu", output: Path | None = None) -> dict:
    """Write a result once; a second call returns the identical frozen report."""
    device = device_name(device)
    destination = output or bundle / "test_report.json"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    # Even a cached report must not be mistaken for a score on the public,
    # additionally redacted copy of the original frozen test data.
    if sha256_file(data / "test.jsonl") != manifest["data_sha256"]["test"]:
        raise ValueError("test split does not match frozen bundle")
    if destination.exists():
        report = json.loads(destination.read_text(encoding="utf-8"))
        if report.get("model_version") != manifest["model_version"]:
            raise ValueError("existing report belongs to a different model")
        return report
    model, tokenizer, manifest, policy = load_bundle(bundle, device)
    rows = load_rows(data / "test.jsonl")
    encoded = encode_rows(rows, tokenizer, manifest["max_length"])
    scores = score_rows(model, tokenizer, encoded, batch_size, device)
    metrics = _report(scores, policy, policy["threshold"])
    # Report source strata as well as site strata; synthetic attacks remain visible.
    strata = {}
    for field in ("source", "origin"):
        strata[field] = {}
        for value in sorted({row[field] for row in rows}):
            indices = [i for i, row in enumerate(rows) if row[field] == value]
            subset = {key: [values[i] for i in indices] for key, values in scores.items()}
            strata[field][value] = _report(subset, policy, policy["threshold"])
    normals = sum(row["block"] == 0 for row in rows)
    sites = len({row["site"] for row in rows if row["block"] == 0})
    support = {name: sum(row["primary_type"] == name for row in rows) for name in TYPE_NAMES}
    minimums = normals >= 2000 and sites >= 20 and all(value >= 50 for value in support.values())
    block = metrics["block"]
    gate = (minimums and block["fpr"] <= .01 and block["recall"] >= .8
            and all(metrics["choice"]["by_type"][name]["end_to_end_recall"] >= .7 for name in TYPE_NAMES))
    report = {"model_version": manifest["model_version"], "split": "independent_test",
              "data_sha256": manifest["data_sha256"]["test"], "test_used_once": True,
              "support": {"normal": normals, "normal_sites": sites, **support},
              "minimums_met": minimums, "offline_gate_passed": gate,
              "production_validated": False, "policy": policy, "metrics": metrics,
              "by_source": strata["source"], "by_origin": strata["origin"],
              "tokens": token_report(encoded),
              "label_counts": label_report(rows)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/v2"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate_bundle(args.bundle, args.data, args.batch_size, args.device, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

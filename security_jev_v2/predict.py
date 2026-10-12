"""Offline single-request Noul and conditional Choice inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# The shared preprocessor and strict bundle loader prevent train/serve drift.
from security_jev_v2.model import TYPE_NAMES, load_bundle
from security_jev_v2.preprocess import encode_request


def predict_loaded(model, tokenizer, manifest: dict, policy: dict,
                   request: str, device: str) -> dict:
    encoded = encode_request(request, tokenizer, manifest["max_length"])
    ids = torch.tensor([encoded["input_ids"]], device=device)
    with torch.no_grad():
        logits = model(ids, torch.ones_like(ids))
    # Calibration parameters are fit on a split unused for weight selection.
    block_logit = logits["block"] / policy["block_temperature"] + policy["block_bias"]
    type_logit = logits["types"] / policy["type_temperature"]
    p_block = float(torch.sigmoid(block_logit)[0])
    type_values = torch.softmax(type_logit, dim=-1)[0].tolist()
    predicted_block = p_block >= 0.5
    type_scores = dict(zip(TYPE_NAMES, type_values))
    return {
        "model_version": manifest["model_version"],
        "noul": {"probabilities": {"是": p_block, "否": 1.0 - p_block},
                 "selected": "是" if predicted_block else "否"},
        "choice": {"conditional_on": "已是攻击", "probabilities": type_scores,
                   "applicable": predicted_block,
                   "selected": TYPE_NAMES[max(range(len(type_values)), key=type_values.__getitem__)] if predicted_block else None},
        "policy": {"experimental_threshold": policy["threshold"],
                   "would_block": p_block >= policy["threshold"],
                   "status": policy["status"]},
        "truncated": encoded["truncated"], "retained_ratio": encoded["retained_ratio"],
        "production_validated": False,
    }


def predict_request(bundle: Path, request: str, device: str = "cpu") -> dict:
    model, tokenizer, manifest, policy = load_bundle(bundle, device)
    return predict_loaded(model, tokenizer, manifest, policy, request, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(predict_request(args.bundle, args.request_file.read_text(encoding="utf-8"), args.device),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

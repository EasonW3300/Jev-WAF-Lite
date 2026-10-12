"""Run manually labeled trial requests for qualitative error analysis only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# These requests are visible examples; their outcomes are never used for tuning.
from security_jev_v2.model import load_bundle
from security_jev_v2.predict import predict_loaded


def analyze_cases(bundle: Path, cases: Path, device: str = "cpu") -> dict:
    model, tokenizer, manifest, policy = load_bundle(bundle, device)
    results = []
    for line in cases.open(encoding="utf-8"):
        case = json.loads(line)
        prediction = predict_loaded(model, tokenizer, manifest, policy, case["state"], device)
        expected_block = bool(case["expected_block"])
        predicted_block = prediction["policy"]["would_block"]
        expected_type = case["expected_type"]
        predicted_type = prediction["choice"]["selected"]
        results.append({"id": case["id"], "note": case["note"],
                        "expected_block": expected_block, "predicted_block": predicted_block,
                        "block_score": prediction["noul"]["probabilities"]["是"],
                        "expected_type": expected_type, "predicted_type": predicted_type,
                        "correct_block": expected_block == predicted_block,
                        "correct_end_to_end": (expected_block == predicted_block
                                               and (not expected_block or predicted_type == expected_type)),
                        "truncated": prediction["truncated"]})
    return {"model_version": manifest["model_version"], "status": "qualitative_only",
            "production_validated": False, "cases": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("trial_cases.jsonl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_cases(args.bundle, args.cases, args.device)
    body = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(body, encoding="utf-8")
    print(body, end="")


if __name__ == "__main__":
    main()

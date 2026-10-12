"""Audit frozen v2 data before allocating a full BERT training run."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Validation checks exact request, site and payload-family separation; this report
# additionally shows provenance and whether every injected request has its normal pair.
from security_data.v2 import SPLITS, TYPE_NAMES, validate_splits


def audit(data: Path) -> dict:
    splits = {}
    for name in SPLITS:
        with (data / f"{name}.jsonl").open(encoding="utf-8") as stream:
            splits[name] = [json.loads(line) for line in stream if line.strip()]
    counts = validate_splits(splits)
    detail = {}
    for name, rows in splits.items():
        normal_pairs = {row["pair_id"] for row in rows if row["block"] == 0 and row["pair_id"]}
        synthetic = [row for row in rows if row["origin"] == "synthetic_payload"]
        capped = Counter((row["site"], row["primary_type"]) for row in synthetic)
        detail[name] = {
            "source_counts": dict(Counter(row["source"] for row in rows)),
            "origin_counts": dict(Counter(row["origin"] for row in rows)),
            "sites": len({row["site"] for row in rows if row["source"] == "wcp"}),
            "normal_sites": len({row["site"] for row in rows if row["source"] == "wcp" and row["block"] == 0}),
            "synthetic_count": len(synthetic),
            "unpaired_synthetic": sum(row["pair_id"] not in normal_pairs for row in synthetic),
            "max_synthetic_per_site_type": max(capped.values(), default=0),
            "duplicate_ids": len(rows) - len({row["id"] for row in rows}),
            "duplicate_texts": len(rows) - len({row["state"] for row in rows}),
        }
    return {"counts": counts, "detail": detail,
            "test_minimums_met": (counts["test"]["normal"] >= 2000
                                  and detail["test"]["normal_sites"] >= 20
                                  and all(counts["test"][name] >= 50 for name in TYPE_NAMES)),
            "test_is_held_out": all(row["source"] == "wcp" for row in splits["test"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/v2"))
    args = parser.parse_args()
    result = audit(args.data)
    (args.data / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Apply the current redactor and historical-source caps to generated v2 rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# This idempotent stage also handles datasets built before a redaction update.
from security_data.v2 import SPLITS, TYPE_NAMES, _sha256_file, template_id, validate_splits
from security_jev_v2.preprocess import normalize_http


def finalize(data: Path) -> dict:
    rows = {}
    for split in SPLITS:
        with (data / f"{split}.jsonl").open(encoding="utf-8") as stream:
            rows[split] = [json.loads(line) for line in stream if line.strip()]
    legacy_groups = defaultdict(list)
    for row in rows["train"]:
        if row["origin"] == "legacy_weak_label":
            legacy_groups[(row["site"], row["primary_type"] or "normal")].append(row)
    keep_legacy = {row["id"] for (site, category), group in legacy_groups.items()
                   for row in sorted(group, key=lambda item: item["id"])[:100 if category == "normal" else 500]}
    rows["train"] = [row for row in rows["train"]
                     if row["origin"] != "legacy_weak_label" or row["id"] in keep_legacy]
    seen = set()
    # The holdout owns a newly colliding normalized request, so development
    # splits cannot silently make the test text familiar to the model.
    for split in ("test", "calib", "val", "train"):
        clean = []
        for row in rows[split]:
            row["state"] = normalize_http(row["state"])
            row["template_id"] = template_id(row["state"], row["site"])
            if row["state"] in seen:
                continue
            seen.add(row["state"])
            clean.append(row)
        paired = {row["pair_id"] for row in clean if row["block"] == 0 and row["pair_id"]}
        rows[split] = [row for row in clean if row["origin"] != "synthetic_payload" or row["pair_id"] in paired]
    counts = validate_splits(rows)
    hashes = {}
    for split, group in rows.items():
        target = data / f"{split}.jsonl"
        staged = data / f".{split}.jsonl.tmp"
        with staged.open("w", encoding="utf-8") as stream:
            for row in group:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        staged.replace(target)
        hashes[split] = _sha256_file(target)
    report_path = data / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["counts"] = counts
    report["data_sha256"] = hashes
    report["redaction_version"] = "http-canonical-head-tail-v2"
    report["legacy_cap"] = {"normal_per_site": 100, "attack_per_site_type": 500}
    report["test_site_count"] = len({row["site"] for row in rows["test"] if row["block"] == 0})
    report["test_minimums_met"] = (counts["test"]["normal"] >= 2000
                                   and report["test_site_count"] >= 20
                                   and all(counts["test"][name] >= 50 for name in TYPE_NAMES))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/generated/v2"))
    args = parser.parse_args()
    print(json.dumps(finalize(args.data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

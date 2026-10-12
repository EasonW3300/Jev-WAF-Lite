"""Create a more strongly redacted, shareable copy of frozen v2 JSONL splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

# Matching raw and URL-encoded credentials matters because captured requests can
# contain nested URLs or JSON bodies that ordinary header redaction misses.
PATTERNS = (
    ("authorization_json", re.compile(r'''(?i)(["']authorization["']\s*:\s*["'])[^"']*'''), r"\1<REDACTED>"),
    ("bearer", re.compile(r"(?i)\bbearer(?:\s|%20|%2520)+[A-Za-z0-9._~+/%=-]{8,}"), "Bearer <REDACTED>"),
    ("aws_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "<CLOUD_KEY_ID>"),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<EMAIL>"),
    ("ipv4", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"), "<IP_ADDRESS>"),
    ("credential_field", re.compile(
        r"(?i)((?:authorization|access[_-]?token|refresh[_-]?token|id[_-]?token|api[_-]?key|apikey|"
        r"password|passwd|secret|session|cookie|signature|x-amz-credential|x-amz-signature|"
        r"awsaccesskeyid)(?:\\?[\"']|%22|%2522)?(?:\s*[:=]\s*|(?:%25)*%3d)"
        r"(?:\\?[\"']|%22|%2522)?)([^\s&,\"'{}\\]{4,})"),
     r"\1<REDACTED>"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact(text: str) -> tuple[str, Counter]:
    """Remove obvious personal identifiers and credential tokens from one request."""
    counts: Counter = Counter()
    for name, pattern, replacement in PATTERNS:
        text, changed = pattern.subn(replacement, text)
        counts[name] += changed
    return text, counts


def create_public_copy(source: Path, output: Path) -> dict:
    """Preserve labels and lineage while recording that text bytes have changed."""
    if source.resolve() == output.resolve():
        raise ValueError("public copy must not overwrite frozen training data")
    output.mkdir(parents=True, exist_ok=True)
    source_report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    report = {**source_report, "publication_redaction_version": "credential-and-identifier-v1",
              "original_training_data_sha256": source_report["data_sha256"],
              "original_training_counts": source_report["counts"],
              "model_training_data_match": False, "omitted_legacy_csic_rows": 0,
              "redaction_counts": {}, "data_sha256": {}, "counts": {}}
    for split in ("train", "val", "calib", "test"):
        counts: Counter = Counter()
        labels: Counter = Counter()
        with (source / f"{split}.jsonl").open(encoding="utf-8") as reader, \
                (output / f"{split}.jsonl").open("w", encoding="utf-8") as writer:
            for line in reader:
                row = json.loads(line)
                # The historical CSIC mirror came through the old project and
                # its redistribution terms are unclear; do not republish it.
                if row["source"] == "csic2010":
                    report["omitted_legacy_csic_rows"] += 1
                    continue
                row["state"], changed = redact(row["state"])
                counts.update(changed)
                labels[row["primary_type"] or "normal"] += 1
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["redaction_counts"][split] = dict(counts)
        report["counts"][split] = {"rows": sum(labels.values()), **dict(labels)}
        report["data_sha256"][split] = sha256_file(output / f"{split}.jsonl")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/v2"))
    args = parser.parse_args()
    print(json.dumps(create_public_copy(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

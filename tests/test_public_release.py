"""Check that the shareable data copy hides captured identifiers and old CSIC rows."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

# The publication transformer preserves labels while removing source-specific secrets.
from security_data.public_release import create_public_copy


class PublicReleaseTests(unittest.TestCase):
    def test_public_copy_redacts_credentials_and_excludes_csic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private, public = root / "private", root / "public"
            private.mkdir()
            source_hashes = {}
            for split in ("train", "val", "calib", "test"):
                rows = [{"state": ('POST /?AWSAccessKeyId=AKIAABCDEFGHIJKLMNOP HTTP/1.1\n'
                                   'Host: <HOST>\n\n{"Authorization":"Bearer secret-token-value",'
                                   '"email":"person@example.test"}'),
                         "source": "wcp", "block": 0, "primary_type": None}]
                if split == "train":
                    rows.append({"state": "GET /old HTTP/1.1", "source": "csic2010",
                                 "block": 0, "primary_type": None})
                path = private / f"{split}.jsonl"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                source_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            (private / "report.json").write_text(json.dumps({"data_sha256": source_hashes,
                "counts": {split: {} for split in source_hashes}}), encoding="utf-8")

            report = create_public_copy(private, public)
            released = (public / "train.jsonl").read_text(encoding="utf-8")
            self.assertEqual(report["omitted_legacy_csic_rows"], 1)
            self.assertEqual(report["counts"]["train"]["rows"], 1)
            for sensitive in ("AKIAABCDEFGHIJKLMNOP", "secret-token-value", "person@example.test", "/old"):
                self.assertNotIn(sensitive, released)
            self.assertNotEqual(report["data_sha256"]["train"], source_hashes["train"])
            self.assertFalse(report["model_training_data_match"])


if __name__ == "__main__":
    unittest.main()

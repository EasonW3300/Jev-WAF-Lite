"""Data construction must keep site and payload families out of other splits."""

import unittest
import csv
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

# These small examples exercise provenance and paired-request rules without downloads.
from security_data.v2 import build_dataset, inject_payload, payload_family, split_key, template_id, validate_splits
from security_jev_v2.preprocess import normalize_http
from security_data.finalize_v2 import finalize
from security_data.audit_v2 import audit
from security_jev_v2.data import LengthBucketBatchSampler, load_rows


class V2DataTests(unittest.TestCase):
    def test_length_bucket_sampler_covers_each_row_once(self):
        rows = [{"input_ids": list(range(length))} for length in range(3, 25)]
        sampler = LengthBucketBatchSampler(rows, 4, pool_size=8)
        first = [index for batch in sampler for index in batch]
        second = [index for batch in sampler for index in batch]
        self.assertEqual(sorted(first), list(range(len(rows))))
        self.assertEqual(sorted(second), list(range(len(rows))))
        self.assertNotEqual(first, second)

    def test_uncertain_or_composite_attack_is_rejected_from_supervised_data(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            for label in (None, "SQL注入+XSS"):
                path.write_text(json.dumps({"id": "uncertain", "state": "GET / HTTP/1.1",
                                            "block": 1, "primary_type": label}) + "\n")
                with self.assertRaises(ValueError):
                    load_rows(path)

    def test_jsonl_reader_preserves_unicode_line_separator_inside_request(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            row = {"id": "unicode", "state": "GET /?q=a\u2028b HTTP/1.1", "block": 0,
                   "primary_type": None, "site": "example.test", "source": "fixture"}
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            self.assertEqual(load_rows(path)[0]["state"], row["state"])

    def test_preprocessor_masks_host_and_common_secrets_in_url_and_json(self):
        state = ('POST https://shop.example.test/search?token=abc&q=shoes HTTP/1.1\n'
                 'Host: shop.example.test\nAuthorization: Bearer should-not-appear\n'
                 'Content-Type: application/json\n\n{"password":"red-secret","email":"user@example.test","item":"shoes"}')
        result = normalize_http(state)
        self.assertIn('Host: <HOST>', result)
        self.assertIn('token=<REDACTED>', result)
        self.assertIn('"password":"<REDACTED>"', result)
        self.assertNotIn('shop.example.test', result)
        self.assertNotIn('should-not-appear', result)
        self.assertNotIn('user@example.test', result)
        self.assertIn('<EMAIL>', result)

    def test_injection_replaces_a_value_without_changing_template(self):
        normal = "GET /shop/search?q=book&page=2 HTTP/1.1\nHost: <HOST>"
        attack = inject_payload(normal, "' OR '1'='1")
        self.assertIsNotNone(attack)
        self.assertIn("%27", attack)
        self.assertEqual(template_id(normal), template_id(attack))
        self.assertNotEqual(template_id(normal, "site-a.test"), template_id(normal, "site-b.test"))
        self.assertNotEqual(normal, attack)

    def test_site_and_payload_split_assignments_are_stable(self):
        self.assertEqual(split_key("example.org"), split_key("example.org"))
        self.assertIn(split_key("example.org"), ("train", "val", "calib", "test"))
        self.assertEqual(payload_family("%27%20OR%201%3D1"), payload_family("' OR 1=1"))

    def test_cross_split_site_or_family_overlap_fails(self):
        base = {"state": "GET /?q=ok HTTP/1.1", "block": 0,
                "primary_type": None, "source": "wcp", "site": "example.org",
                "payload_family": None, "template_id": "t1"}
        with self.assertRaises(ValueError):
            validate_splits({"train": [base], "val": [{**base, "state": "GET /?q=other HTTP/1.1"}],
                             "calib": [], "test": []})
        attack = {**base, "block": 1, "primary_type": "XSS", "payload_family": "family-a"}
        with self.assertRaises(ValueError):
            validate_splits({"train": [attack], "val": [],
                             "calib": [{**attack, "site": "different.org", "state": "GET /?q=attack HTTP/1.1"}],
                             "test": []})

    def test_builder_preserves_normal_and_attack_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "legitimate.zip"
            with ZipFile(archive, "w") as out:
                for index in range(80):
                    host = f"site{index}.example"
                    out.writestr(f"Legitimate/{host}.json", json.dumps([
                        {"method": "GET", "url": f"/search?q=book{index}",
                         "headers": {"host": host}, "data": ""}]))
            payloads = root / "payload_full.csv"
            with payloads.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["payload", "length", "attack_type", "label"])
                writer.writeheader()
                for name, text in [("sqli", "' OR 1=1"), ("xss", "<svg/onload=alert(1)>"),
                                   ("path-traversal", "../../etc/passwd"), ("cmdi", ";id")]:
                    writer.writerow({"payload": text, "length": len(text), "attack_type": name, "label": "anom"})
            output = root / "generated"
            report = build_dataset(archive, payloads, output, legacy_dir=None, crs_dir=None,
                                   normal_per_site=100, attacks_per_site_type=20)
            self.assertGreater(report["counts"]["test"]["normal"], 0)
            rows = {name: [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
                    for name in ("train", "val", "calib", "test")}
            validate_splits(rows)
            self.assertTrue(all("Host: <HOST>" in row["state"] for split in rows.values() for row in split))
            self.assertTrue(all(row["source"] == "wcp" for split in rows.values() for row in split))
            first = finalize(output)
            second = finalize(output)
            self.assertEqual(first["data_sha256"], second["data_sha256"])
            audited = audit(output)
            self.assertTrue(audited["test_is_held_out"])
            self.assertEqual(audited["detail"]["test"]["unpaired_synthetic"], 0)


if __name__ == "__main__":
    unittest.main()

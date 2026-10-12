"""Deterministic cross-site data construction helpers for the v2 security model."""

from __future__ import annotations

import hashlib
import argparse
import base64
import csv
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote_plus, urlsplit
from zipfile import ZipFile

import ijson
import yaml

# ijson streams each ZIP member because some public JSON files exceed 700 MB.
# PyYAML reads the pinned CRS regression fixtures without constructing objects.
# The shared normalizer removes host identity and credentials before any model input.
from security_jev_v2.preprocess import normalize_http

def _crs_state(payload: dict) -> str:
    """Decode a CRS positive rule fixture into an HTTP request for shared normalization."""
    if "encoded_request" in payload:
        return base64.b64decode(payload["encoded_request"], validate=True).decode("latin-1")
    headers = payload.get("headers") or {}
    if not isinstance(headers, dict):
        raise ValueError("CRS fixture headers must be a mapping")
    body = payload.get("data", "")
    if not isinstance(body, str):
        raise ValueError("CRS fixture request data must be text")
    lines = [f"{payload.get('method', 'GET')} {payload.get('uri', '/')} {payload.get('version', 'HTTP/1.1')}"]
    lines += [f"{key}: {value}" for key, value in headers.items()]
    return "\n".join(lines) + "\n\n" + body

SPLITS = ("train", "val", "calib", "test")
TYPE_NAMES = ("SQL注入", "XSS", "路径遍历", "其他攻击")
PAYLOAD_CLASSES = {"sqli": "SQL注入", "xss": "XSS",
                   "path-traversal": "路径遍历", "cmdi": "其他攻击"}
PARAM = re.compile(r"([?&][^=&\s]+)=([^&\s]*)")
GENERAL_KEYS = {"q", "query", "search", "keyword", "term", "id", "item", "product",
                "user", "name", "file", "path", "url", "page", "redirect", "next",
                "target", "host", "s", "k", "data"}
FILE_KEYS = {"file", "path", "document", "download", "url", "template",
             "page", "resource", "name", "image"}


def _sha256_file(path: Path) -> str:
    """Hash large source archives without loading them into process memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_key(value: str, seed: int = 42) -> str:
    """Assign entire sites and entire payload families before generating pairs."""
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "train" if fraction < .6 else "val" if fraction < .7 else "calib" if fraction < .8 else "test"


def payload_family(payload: str) -> str:
    """Group common encoding, case and numeric variants of the same payload."""
    canonical = payload
    for _ in range(2):
        canonical = unquote_plus(canonical)
    canonical = re.sub(r"\d+", "0", canonical.lower())
    canonical = re.sub(r"\s+", " ", canonical).strip()
    return hashlib.sha256(canonical.encode()).hexdigest()[:20]


def template_id(state: str, site: str = "") -> str:
    """Identify one site's request shape without values or credentials."""
    method, target, _ = state.splitlines()[0].split(None, 2)
    parsed = urlsplit(target)
    keys = sorted(re.findall(r"(?:^|[&])([^=&]+)=", parsed.query))
    shape = f"{site}|{method}|{parsed.path}|{','.join(keys)}"
    return hashlib.sha256(shape.encode()).hexdigest()[:20]


def inject_payload(state: str, payload: str, category: str | None = None) -> str | None:
    """Replace a plausible user-controlled query value in the same request shape."""
    lines = state.split("\n", 1)
    first = lines[0].split(None, 2)
    if len(first) != 3:
        return None
    target = first[1]
    allowed = FILE_KEYS if category == "路径遍历" else GENERAL_KEYS
    match = next((item for item in PARAM.finditer(target)
                  if item.group(1)[1:].lower() in allowed), None)
    if match is None:
        return None
    changed = target[:match.start(2)] + quote(payload, safe="") + target[match.end(2):]
    request_line = f"{first[0]} {changed} {first[2]}"
    return request_line + ("\n" + lines[1] if len(lines) > 1 else "")


def validate_splits(splits: dict[str, list[dict]]) -> dict:
    """Reject exact request, source site and attack-family leakage across splits."""
    if set(splits) != set(SPLITS):
        raise ValueError("train, val, calib and test splits are required")
    owners = {"state": {}, "site": {}, "template_id": {}, "payload_family": {}}
    counts = {}
    for split in SPLITS:
        counts[split] = {"rows": len(splits[split]), "normal": 0,
                         **{name: 0 for name in TYPE_NAMES}}
        for row in splits[split]:
            if row["block"] not in (0, 1):
                raise ValueError("block label must be 0 or 1")
            category = row["primary_type"]
            if (row["block"] == 0 and category is not None) or (row["block"] == 1 and category not in TYPE_NAMES):
                raise ValueError("confirmed attacks need one primary category; benign requests need none")
            counts[split][category if category else "normal"] += 1
            checks = {"state": row["state"],
                      "site": row["site"] if row["source"] == "wcp" else None,
                      "template_id": row.get("template_id") if row["source"] == "wcp" else None,
                      "payload_family": row["payload_family"]}
            for name, key in checks.items():
                if key is None:
                    continue
                previous = owners[name].setdefault(key, split)
                if previous != split:
                    raise ValueError(f"{name} leaks between {previous} and {split}")
    return counts


def _site_key(host: str) -> str:
    """Group ordinary subdomains under one site to avoid simple site leakage."""
    host = host.lower().split(":", 1)[0].strip(".")
    parts = host.split(".")
    if len(parts) < 2:
        return host
    multi_suffix = {"co.uk", "com.au", "co.jp", "co.nz", "com.br", "com.cn", "co.in"}
    size = 3 if ".".join(parts[-2:]) in multi_suffix and len(parts) >= 3 else 2
    return ".".join(parts[-size:])


def _wcp_request(row: dict) -> tuple[str, str] | None:
    method, target, headers, body = row.get("method"), row.get("url"), row.get("headers"), row.get("data")
    if (not isinstance(method, str) or not isinstance(target, str)
            or not isinstance(headers, dict) or not isinstance(body, str)):
        return None
    if method.upper() not in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
        return None
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    host = lowered.get("host") or urlsplit(target).hostname
    if not host:
        return None
    if target.startswith(("http://", "https://")):
        parsed = urlsplit(target)
        target = (parsed.path or "/") + ("?" + parsed.query if parsed.query else "")
    if not target.startswith("/"):
        return None
    header_lines = [f"Host: {host}"]
    for key in ("content-type", "content-disposition"):
        if key in lowered:
            header_lines.append(f"{key}: {lowered[key]}")
    raw = f"{method.upper()} {target} HTTP/1.1\n" + "\n".join(header_lines)
    if body:
        raw += "\n\n" + body
    try:
        return _site_key(host), normalize_http(raw)
    except (ValueError, UnicodeError):
        return None


def _select_wcp(archive: Path, cap: int, max_rows_per_member: int,
                max_member_bytes: int) -> tuple[dict[str, list[dict]], dict]:
    """Keep a stable hash sample per site without holding the full ZIP in RAM."""
    selected: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    selected_states: dict[str, set[str]] = defaultdict(set)
    seen_rows = invalid = skipped_large = third_party = 0
    with ZipFile(archive) as zipped:
        for member_index, entry in enumerate(zipped.infolist(), 1):
            if not entry.filename.endswith(".json"):
                continue
            if entry.file_size > max_member_bytes:
                skipped_large += 1
                continue
            primary_site = None
            with zipped.open(entry) as stream:
                for member_row, source_row in enumerate(ijson.items(stream, "item")):
                    if member_row >= max_rows_per_member:
                        break
                    seen_rows += 1
                    converted = _wcp_request(source_row)
                    if converted is None:
                        invalid += 1
                        continue
                    site, state = converted
                    if primary_site is None:
                        primary_site = site
                    if site != primary_site:
                        third_party += 1
                        continue
                    rank = int.from_bytes(hashlib.sha256(f"42:{site}:{state}".encode()).digest()[:8], "big")
                    heap = selected[site]
                    if state in selected_states[site]:
                        continue
                    candidate = (-rank, state, entry.filename)
                    if len(heap) < cap:
                        heapq.heappush(heap, candidate)
                        selected_states[site].add(state)
                    elif rank < -heap[0][0]:
                        removed = heapq.heapreplace(heap, candidate)
                        selected_states[site].discard(removed[1])
                        selected_states[site].add(state)
            if member_index % 25 == 0:
                print(f"WCP scan {member_index}/{len(zipped.infolist())} members; {seen_rows} requests", flush=True)
    result = {site: [{"state": state, "source_file": filename}
                     for _, state, filename in sorted(heap, reverse=True)]
              for site, heap in selected.items()}
    return result, {"read": seen_rows, "invalid_or_oversize": invalid,
                    "max_rows_per_member": max_rows_per_member,
                    "max_member_bytes": max_member_bytes,
                    "skipped_large_members": skipped_large,
                    "skipped_third_party_requests": third_party,
                    "sites": len(result)}


def _crs_command_payloads(root: Path | None) -> tuple[list[str], list[dict]]:
    """Use only positive RCE rule expectations; do not infer benign from nonmatches."""
    if root is None:
        return [], []
    folder = root / "REQUEST-932-APPLICATION-ATTACK-RCE"
    values, requests = [], []
    for path in sorted(folder.glob("*.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            continue
        rule_id = int(document.get("rule_id", path.stem))
        for case in document.get("tests", []):
            for stage_number, stage in enumerate(case.get("stages", []), 1):
                expected = set((stage.get("output") or {}).get("log", {}).get("expect_ids", []))
                if rule_id not in expected:
                    continue
                inp = stage.get("input") or {}
                try:
                    state = normalize_http(_crs_state(inp))
                except (ValueError, KeyError, TypeError, UnicodeError):
                    continue
                requests.append({"state": state, "source_id": f"{path.stem}-{case.get('test_id')}-{stage_number}"})
                for location in (urlsplit(str(inp.get("uri", ""))).query, inp.get("data", "")):
                    if not isinstance(location, str):
                        continue
                    for items in parse_qs(location, keep_blank_values=False).values():
                        for value in items:
                            if re.search(r"[;&|`$]|\b(?:whoami|curl|wget|bash|powershell|cat)\b", value, re.I):
                                values.append(value)
    return list(dict.fromkeys(values)), requests


def _payloads(csv_path: Path, crs_dir: Path | None) -> tuple[dict[str, dict[str, list[str]]], list[dict]]:
    by_type: dict[str, dict[str, list[str]]] = {name: defaultdict(list) for name in TYPE_NAMES}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            category = PAYLOAD_CLASSES.get(row.get("attack_type"))
            value = row.get("payload")
            if row.get("label") != "anom" or not category or not value or len(value) > 2000:
                continue
            if category == "路径遍历":
                decoded = unquote_plus(unquote_plus(value))
                if "../" not in decoded and "..\\" not in decoded:
                    continue
            by_type[category][payload_family(value)].append(value)
    commands, crs_rows = _crs_command_payloads(crs_dir)
    for value in commands:
        by_type["其他攻击"][payload_family(value)].append(value)
    return by_type, crs_rows


def _make_row(state: str, block: int, category: str | None, site: str,
              source: str, origin: str, source_id: str, family: str | None = None,
              pair_id: str | None = None) -> dict:
    identifier = hashlib.sha256(f"{source}:{source_id}:{state}".encode()).hexdigest()[:24]
    return {"id": identifier, "state": state, "block": block,
            "primary_type": category, "site": site, "source": source, "origin": origin,
            "source_id": source_id, "template_id": template_id(state, site),
            "payload_family": family, "pair_id": pair_id}


def build_dataset(wcp_archive: Path, params_csv: Path, output: Path,
                  legacy_dir: Path | None = None,
                  crs_dir: Path | None = Path("security_data/raw/crs/tests/regression/tests"),
                  normal_per_site: int = 100, attacks_per_site_type: int = 20,
                  max_rows_per_member: int = 5000,
                  max_member_bytes: int = 50 * 1024 * 1024) -> dict:
    """Freeze four deterministic splits before any v2 training or tuning."""
    if min(normal_per_site, attacks_per_site_type, max_rows_per_member, max_member_bytes) < 1:
        raise ValueError("sample caps must be positive")
    sites, read_report = _select_wcp(wcp_archive, normal_per_site, max_rows_per_member, max_member_bytes)
    payloads, crs_rows = _payloads(params_csv, crs_dir)
    split_payloads = {split: {category: sorted((family, value)
                      for family, values in families.items() if split_key(family) == split
                      for value in set(values))
                      for category, families in payloads.items()} for split in SPLITS}
    splits: dict[str, list[dict]] = {name: [] for name in SPLITS}
    seen_states = set()
    # Split sites and payload families first. Attack pairs reuse the exact
    # normal template, preventing site or header presence from becoming a label.
    for site in sorted(sites):
        split = split_key(site)
        available = []
        for item in sites[site]:
            state = item["state"]
            if state in seen_states:
                continue
            seen_states.add(state)
            pair = hashlib.sha256(f"{site}:{state}".encode()).hexdigest()[:20]
            source_id = f"{item['source_file']}:{hashlib.sha256(state.encode()).hexdigest()[:20]}"
            row = _make_row(state, 0, None, site, "wcp", "source_legitimate",
                            source_id, pair_id=pair)
            splits[split].append(row)
            if inject_payload(state, "probe") is not None:
                available.append(row)
        for category in TYPE_NAMES:
            pool = split_payloads[split][category]
            if not pool:
                continue
            for index, normal in enumerate(available[:attacks_per_site_type]):
                family, value = pool[int.from_bytes(hashlib.sha256(
                    f"{site}:{category}:{index}".encode()).digest()[:8], "big") % len(pool)]
                changed = inject_payload(normal["state"], value, category)
                if changed is None or changed in seen_states:
                    continue
                seen_states.add(changed)
                splits[split].append(_make_row(changed, 1, category, site, "wcp",
                                               "synthetic_payload", normal["source_id"],
                                               family=family, pair_id=normal["pair_id"]))
    # Optional historical weak labels can only enter training; this standalone
    # release defaults to the newly sourced WCP and payload data alone.
    if legacy_dir is not None:
        legacy_candidates = []
        for split in ("train", "val", "test"):
            with (legacy_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    old = json.loads(line)
                    category = None if old["noul"]["y"] == 0 else old["choice"]["options"][old["choice"]["y"]]["t"]
                    try:
                        state = normalize_http(old["state"])
                    except ValueError:
                        continue
                    if state in seen_states:
                        continue
                    seen_states.add(state)
                    legacy_candidates.append(_make_row(state, old["noul"]["y"], category,
                                                       "legacy-csic2010" if old["source"] == "csic2010" else "legacy-crs",
                                                       old["source"], "legacy_weak_label", old["id"]))
        # Historical CSIC is mostly one site's normal traffic. Keep it as
        # weak-label development material without overwhelming cross-site rows.
        legacy_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in legacy_candidates:
            legacy_groups[(row["site"], row["primary_type"] or "normal")].append(row)
        for (site, category), rows in sorted(legacy_groups.items()):
            cap = 100 if category == "normal" else 500
            splits["train"].extend(sorted(rows, key=lambda row: row["id"])[:cap])
    for row in crs_rows:
        state = row["state"]
        if state in seen_states:
            continue
        seen_states.add(state)
        splits["train"].append(_make_row(state, 1, "其他攻击", "legacy-crs",
                                          "crs-rce", "positive_rule_expectation", row["source_id"]))
    counts = validate_splits(splits)
    output.mkdir(parents=True, exist_ok=True)
    digests = {}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        digests[split] = _sha256_file(path)
    report = {"counts": counts, "wcp": read_report, "data_sha256": digests,
              "sources": {
                  "wcp": {"url": "https://downloads.openappsec.io/waf-comparison-project/legitimate.zip",
                          "license": "Apache-2.0", "archive_sha256": _sha256_file(wcp_archive)},
                  "http_params": {"url": "https://github.com/Morzeux/HttpParamsDataset",
                                  "commit": "926670a710283f87c05b554680facf3f9530548c",
                                  "license": "MIT", "csv_sha256": _sha256_file(params_csv)},
                  "crs": {"url": "https://github.com/coreruleset/coreruleset",
                          "commit": "af76ccfcd1ad947e38ef9998bdd464f1a39fef86"},
                  **({"legacy_csic": {"commit": "424c6e1c1951570473d9d6c0227eaaac6fdd6d02",
                                      "previously_viewed_test_reused_only_as_training": True}}
                     if legacy_dir is not None else {})},
              "payload_families": {name: len(rows) for name, rows in payloads.items()},
              "test_site_count": len({row["site"] for row in splits["test"] if row["block"] == 0}),
              "test_minimums_met": counts["test"]["normal"] >= 2000
              and len({row["site"] for row in splits["test"] if row["block"] == 0}) >= 20
              and all(counts["test"][name] >= 50 for name in TYPE_NAMES)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wcp", type=Path, default=Path("security_data/raw/wcp-legitimate.zip"))
    parser.add_argument("--params", type=Path, default=Path("security_data/raw/http-params/payload_full.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/v2"))
    parser.add_argument("--max-rows-per-member", type=int, default=5000)
    parser.add_argument("--max-member-bytes", type=int, default=50 * 1024 * 1024)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.wcp, args.params, args.output,
                                   max_rows_per_member=args.max_rows_per_member,
                                   max_member_bytes=args.max_member_bytes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

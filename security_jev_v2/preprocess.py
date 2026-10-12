"""Shared request cleaning and BERT token layout for v2 training and inference."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

# Only known credential fields are masked; attack-bearing URL and body text remains visible.
SECRET = re.compile(r"(?i)([?&]|^)(password|passwd|pwd|token|api_key|apikey|secret|sessionid)=([^&\s]+)")
JSON_SECRET = re.compile(r'(?i)(["\'](?:password|passwd|pwd|token|api_key|apikey|secret|sessionid)["\']\s*:\s*["\'])([^"\']*)(["\'])')
EMAIL = re.compile(r'(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]{1,64}(?:@|%40)[A-Z0-9.-]{1,100}\.[A-Z]{2,12}\b')
MAX_INPUT_BYTES = 64 * 1024


def normalize_http(state: str) -> str:
    """Make site identity and common secrets unavailable as label shortcuts."""
    if not isinstance(state, str) or not state.strip():
        raise ValueError("HTTP request text is required")
    if len(state.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("request exceeds 64 KiB")
    lines = state.replace("\r\n", "\n").strip("\n").split("\n")
    request_line = lines[0].split()
    if len(request_line) != 3 or not request_line[2].startswith("HTTP/"):
        raise ValueError("first line must be METHOD path HTTP/version")
    method, target, version = request_line
    if target.startswith(("http://", "https://")):
        parsed = urlsplit(target)
        target = (parsed.path or "/") + ("?" + parsed.query if parsed.query else "")
    output = [f"{method.upper()} {target} {version}"]
    in_body = False
    for line in lines[1:]:
        if not in_body and line == "":
            in_body = True
            output.append("")
            continue
        if not in_body and ":" in line:
            name, value = line.split(":", 1)
            key = name.strip().lower()
            if key == "host":
                output.append("Host: <HOST>")
                continue
            # Keep only headers represented consistently across both public
            # corpora; identity, browser and credential headers cause shortcuts.
            if key in ("content-type", "content-disposition"):
                output.append(f"{name.strip()}: {value.strip()}")
        else:
            output.append(line)
    cleaned = SECRET.sub(lambda match: f"{match.group(1)}{match.group(2)}=<REDACTED>", "\n".join(output))
    cleaned = JSON_SECRET.sub(lambda match: f"{match.group(1)}<REDACTED>{match.group(3)}", cleaned)
    return EMAIL.sub("<EMAIL>", cleaned)


def encode_request(state: str, tokenizer, max_length: int = 512) -> dict:
    """Keep both ends of long normalized requests and report actual coverage."""
    if not 4 <= max_length <= 512:
        raise ValueError("max_length must be 4..512")
    normalized = normalize_http(state)
    pieces = tokenizer(normalized, add_special_tokens=False)["input_ids"]
    total = len(pieces)
    budget = max_length - 2
    truncated = total > budget
    if truncated:
        first = (budget + 1) // 2
        pieces = pieces[:first] + pieces[-(budget - first):]
    return {"input_ids": [tokenizer.cls_token_id, *pieces, tokenizer.sep_token_id],
            "truncated": truncated, "retained_ratio": len(pieces) / total if total else 1.0,
            "total_tokens": total, "unknown_tokens": pieces.count(tokenizer.unk_token_id)}

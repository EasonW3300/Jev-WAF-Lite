"""Read frozen v2 splits and make BERT batches with explicit primary labels."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import torch

# The shared encoder is also called by local inference, so token handling cannot drift.
from security_jev_v2.model import TYPE_NAMES
from security_jev_v2.preprocess import encode_request


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        if row.get("block") not in (0, 1):
            raise ValueError(f"invalid block label: {row.get('id')}")
        category = row.get("primary_type")
        if (row["block"] == 0 and category is not None) or (row["block"] == 1 and category not in TYPE_NAMES):
            raise ValueError(f"invalid primary category: {row.get('id')}")
    return rows


def encode_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    encoded = []
    for row in rows:
        item = encode_request(row["state"], tokenizer, max_length)
        encoded.append({**item, "block_y": row["block"],
                        "type_y": TYPE_NAMES.index(row["primary_type"]) if row["block"] else -1,
                        "site": row["site"], "source": row["source"], "id": row["id"]})
    return encoded


def collate(rows: list[dict], tokenizer) -> dict:
    width = max(len(row["input_ids"]) for row in rows)
    ids = [row["input_ids"] + [tokenizer.pad_token_id] * (width - len(row["input_ids"])) for row in rows]
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor([[int(token != tokenizer.pad_token_id) for token in line] for line in ids]),
            "block_y": torch.tensor([row["block_y"] for row in rows], dtype=torch.float),
            "type_y": torch.tensor([row["type_y"] for row in rows], dtype=torch.long)}


class LengthBucketBatchSampler:
    """Shuffle all rows, then group nearby token lengths to reduce pad compute."""

    def __init__(self, rows: list[dict], batch_size: int, seed: int = 42, pool_size: int = 512):
        if batch_size < 1 or pool_size < batch_size or pool_size % batch_size:
            raise ValueError("bucket pool must be a positive multiple of batch size")
        self.rows = rows
        self.batch_size = batch_size
        self.seed = seed
        self.pool_size = pool_size
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.rows)))
        rng.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.pool_size):
            pool = sorted(indices[start:start + self.pool_size],
                          key=lambda index: len(self.rows[index]["input_ids"]))
            batches.extend(pool[offset:offset + self.batch_size]
                           for offset in range(0, len(pool), self.batch_size))
        rng.shuffle(batches)
        return iter(batches)


def token_report(encoded: list[dict]) -> dict:
    lengths = sorted(row["total_tokens"] for row in encoded)
    if not lengths:
        return {"rows": 0}
    return {"rows": len(lengths), "p50_tokens": lengths[len(lengths) // 2],
            "p95_tokens": lengths[min(len(lengths) - 1, int(len(lengths) * .95))],
            "max_tokens": lengths[-1], "truncated": sum(row["truncated"] for row in encoded),
            "unknown_tokens": sum(row["unknown_tokens"] for row in encoded),
            "min_retained_ratio": min(row["retained_ratio"] for row in encoded)}


def label_report(rows: list[dict]) -> dict:
    return dict(Counter("normal" if row["block"] == 0 else row["primary_type"] for row in rows))

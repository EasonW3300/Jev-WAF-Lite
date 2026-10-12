"""Shared pretrained BERT with a binary Noul head and four-way Choice head."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertTokenizerFast

# BertModel supplies the encoder; config and tokenizer are saved with weights for offline loading.
TYPE_NAMES = ("SQL注入", "XSS", "路径遍历", "其他攻击")
FORMAT_VERSION = 2
PREPROCESSING_VERSION = "http-canonical-head-tail-v2"


class SecurityJevBert(nn.Module):
    """One request representation feeds two typed, separately interpretable decisions."""

    def __init__(self, encoder: BertModel):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        hidden = encoder.config.hidden_size
        self.block = nn.Linear(hidden, 1)
        self.types = nn.Linear(hidden, len(TYPE_NAMES))

    @classmethod
    def from_pretrained(cls, source: str = "google-bert/bert-base-cased") -> "SecurityJevBert":
        # Only the official pretrained encoder is accepted by this standalone release.
        return cls(BertModel.from_pretrained(source, add_pooling_layer=False))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        # The first token carries the request-level representation for both decisions.
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        shared = self.dropout(hidden)
        return {"block": self.block(shared).squeeze(-1), "types": self.types(shared)}


def joint_loss(output: dict, block_y: torch.Tensor, type_y: torch.Tensor,
               type_weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Only confirmed attacks train Choice; benign and unknown types use -1."""
    block = F.binary_cross_entropy_with_logits(output["block"], block_y.float())
    known = (block_y == 1) & (type_y >= 0)
    types = (F.cross_entropy(output["types"][known], type_y[known].long(), weight=type_weight)
             if known.any() else output["types"].sum() * 0.0)
    return {"block": block, "types": types, "total": block + types}


def save_bundle(model: SecurityJevBert, tokenizer, output: Path,
                manifest: dict, policy: dict) -> None:
    """Publish a complete v2 bundle atomically without overwriting v1."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output.name}-staged-", dir=output.parent))
    backup = None
    try:
        model.encoder.config.save_pretrained(staged)
        tokenizer.save_pretrained(staged / "tokenizer")
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, staged / "model.pt")
        metadata = {**manifest, "format_version": FORMAT_VERSION,
                    "attack_types": list(TYPE_NAMES), "preprocessing_version": PREPROCESSING_VERSION}
        (staged / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staged / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-backup-", dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        os.replace(staged, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    finally:
        if staged.exists():
            shutil.rmtree(staged)
        if backup is not None and backup.exists():
            shutil.rmtree(backup)


def load_bundle(path: Path, device: str = "cpu") -> tuple[SecurityJevBert, BertTokenizerFast, dict, dict]:
    """Strictly reject old bundles and reconstruct all components from local files."""
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("format_version") != FORMAT_VERSION
            or manifest.get("attack_types") != list(TYPE_NAMES)
            or manifest.get("preprocessing_version") != PREPROCESSING_VERSION):
        raise ValueError("not a compatible v2 security bundle")
    policy = json.loads((path / "policy.json").read_text(encoding="utf-8"))
    config = BertConfig.from_pretrained(path, local_files_only=True)
    model = SecurityJevBert(BertModel(config, add_pooling_layer=False))
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True), strict=True)
    tokenizer = BertTokenizerFast.from_pretrained(path / "tokenizer", local_files_only=True)
    return model.to(device).eval(), tokenizer, manifest, policy

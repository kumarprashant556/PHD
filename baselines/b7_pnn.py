"""B7 — Progressive Neural Network (PNN) baseline (Trainer-based).

Spawns a new FLAN-T5 column per period.  Lateral adapters inject past columns'
encoder hidden states into the new column's input embeddings.  Zero forgetting
by construction; parameter cost grows linearly with the number of periods.

``PNNWrapper.forward`` runs frozen prev columns under no-grad, applies the
laterals, and feeds the result into the current column.  Trainer just trains
all params with ``requires_grad=True`` — i.e. the latest column + its laterals.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForSeq2SeqLM

# Repo root on sys.path so `baselines._runtime` resolves whether invoked as
# `python baselines/b7_pnn.py` (direct) or via the sweep launcher.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from baselines._runtime import (
    INCAConfig, TrainerRunner, TokenizedDataset,
    model_dtype, standard_trainer,
)


def _get_embed_dim(model):
    for attr in ("d_model", "hidden_size", "n_embd"):
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
    raise AttributeError(f"Cannot find embed dim in {type(model).__name__}")


def _get_embed_layer(model):
    if hasattr(model, "shared"):
        return model.shared
    if hasattr(model, "encoder") and hasattr(model.encoder, "embed_tokens"):
        return model.encoder.embed_tokens
    raise AttributeError(f"Cannot find embedding layer in {type(model).__name__}")


class LateralAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h):
        return torch.sigmoid(self.gate) * self.proj(h)


class PNNColumn(nn.Module):
    """A single PNN column: backbone + N lateral adapters (one per prior column)."""

    def __init__(self, model_name, num_prev, device, load_kw):
        super().__init__()
        backbone = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kw)
        if "device_map" not in load_kw:
            backbone = backbone.to(device)
        self.backbone = backbone
        dim = _get_embed_dim(backbone)
        self.laterals = nn.ModuleList(
            [LateralAdapter(dim).to(device) for _ in range(num_prev)]
        )


class PNNWrapper(nn.Module):
    """All PNN columns; forward applies laterals + runs the latest column."""

    def __init__(self, columns: List[PNNColumn]):
        super().__init__()
        self.columns = nn.ModuleList(columns)

    @torch.no_grad()
    def _get_prev_hiddens(self, input_ids, attention_mask=None):
        prev = []
        for col in self.columns[:-1]:
            enc_out = col.backbone.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True,
            )
            prev.append(enc_out.last_hidden_state)
        return prev

    def _lateral_embeds(self, input_ids, attention_mask=None):
        prev    = self._get_prev_hiddens(input_ids, attention_mask)
        last    = self.columns[-1]
        wte     = _get_embed_layer(last.backbone)
        tok_emb = wte(input_ids)
        for h_prev, lat in zip(prev, last.laterals):
            t = min(tok_emb.shape[1], h_prev.shape[1])
            tok_emb[:, :t] = tok_emb[:, :t] + lat(h_prev[:, :t])
        return tok_emb

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        tok_emb = self._lateral_embeds(input_ids, attention_mask)
        return self.columns[-1].backbone(
            inputs_embeds=tok_emb, attention_mask=attention_mask,
            labels=labels, **kwargs,
        )

    def generate(self, input_ids, attention_mask=None, **kwargs):
        tok_emb = self._lateral_embeds(input_ids, attention_mask)
        return self.columns[-1].backbone.generate(
            inputs_embeds=tok_emb, attention_mask=attention_mask, **kwargs,
        )

    def gradient_checkpointing_enable(self, **kw):
        bb = self.columns[-1].backbone
        if hasattr(bb, "gradient_checkpointing_enable"):
            bb.gradient_checkpointing_enable(**kw)
        if hasattr(bb, "enable_input_require_grads"):
            bb.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        bb = self.columns[-1].backbone
        if hasattr(bb, "gradient_checkpointing_disable"):
            bb.gradient_checkpointing_disable()


@dataclass
class B7PNN:
    cfg:  INCAConfig
    name: str = "b7_pnn"

    _wrapper:   Any  = field(init=False, default=None)
    _columns:   List = field(init=False, default_factory=list)
    _tokenizer: Any  = field(init=False, default=None)
    _device:    str  = field(init=False, default="")
    _load_kw:   Dict = field(init=False, default_factory=dict)

    def build_model(self, tokenizer, device: str):
        self._tokenizer = tokenizer
        self._device    = device
        self._load_kw   = {"dtype": model_dtype(device, self.cfg)}
        if "cuda" in device:
            self._load_kw["device_map"] = "auto"
        col0 = PNNColumn(self.cfg.model_name, 0, device, self._load_kw)
        if hasattr(col0.backbone.config, "pad_token_id"):
            col0.backbone.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self._columns.append(col0)
        self._wrapper = PNNWrapper(self._columns).to(device)
        return self._wrapper

    def scoring_model(self):
        return self._wrapper

    def on_period_start(self, period_label, period_idx):
        if period_idx > 0:
            col = PNNColumn(self.cfg.model_name, len(self._columns),
                            self._device, self._load_kw)
            if hasattr(col.backbone.config, "pad_token_id"):
                col.backbone.config.pad_token_id = (
                    self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
                )
            self._columns.append(col)
            # Re-attach the wrapper so the ModuleList sees the new column.
            self._wrapper = PNNWrapper(self._columns).to(self._device)
        # Freeze all prev columns; only the latest is trainable.
        for col in self._columns[:-1]:
            for p in col.parameters():
                p.requires_grad = False
            col.eval()
        for p in self._columns[-1].parameters():
            p.requires_grad = True

    def on_period_end(self, period_label, period_idx, raw_items):
        pass

    def make_trainer(self, args, raw_items, tokenizer, period_label, period_idx):
        max_in = self.cfg.max_input_length
        max_lb = getattr(self.cfg, "max_target_length", max_in)
        train_ds = TokenizedDataset(raw_items, tokenizer, max_in, max_lb)
        return standard_trainer(self._wrapper, args, train_ds, tokenizer)


def main() -> None:
    p = argparse.ArgumentParser(description="B7 PNN baseline (Trainer-based)")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}
    if args.dataset: cfg_dict["dataset"] = args.dataset
    if args.seed:    cfg_dict["seed"]    = args.seed
    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items()
                        if k in INCAConfig.__dataclass_fields__})
    TrainerRunner(cfg, B7PNN(cfg=cfg), device=args.device).run()


if __name__ == "__main__":
    main()

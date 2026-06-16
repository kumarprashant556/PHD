"""B4 — Learning to Prompt (L2P) baseline (Trainer-based).

Backbone is frozen; a prompt pool is prepended to the encoder input embeddings.
Only the pool parameters are trained.

``L2PWrapper.forward`` adds the prompt-pool key-pull regulariser directly into
``out.loss`` so plain ``Seq2SeqTrainer`` can train it without subclassing.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoModelForSeq2SeqLM

# Repo root on sys.path so `baselines._runtime` resolves whether invoked as
# `python baselines/b4_l2p.py` (direct) or via the sweep launcher.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from baselines._runtime import (
    INCAConfig, TrainerRunner, TokenizedDataset,
    model_dtype, standard_trainer,
)


class PromptPool(nn.Module):
    def __init__(self, embed_dim, pool_size=10, prompt_len=5, top_n=3):
        super().__init__()
        self.pool_size  = pool_size
        self.prompt_len = prompt_len
        self.top_n      = min(top_n, pool_size)
        self.keys    = nn.Parameter(torch.randn(pool_size, embed_dim) * 0.02)
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, embed_dim) * 0.02)

    def forward(self, query):
        q   = F.normalize(query, dim=-1)
        k   = F.normalize(self.keys, dim=-1)
        sim = q @ k.t()
        _, topi = sim.topk(self.top_n, dim=-1)
        B   = query.shape[0]
        p   = self.prompts[topi].reshape(B, self.top_n * self.prompt_len, -1)
        pull = -sim.gather(1, topi).mean()
        return p, pull


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


class L2PWrapper(nn.Module):
    """Backbone (frozen) + PromptPool (trainable) packaged for Seq2SeqTrainer."""

    def __init__(self, backbone, pool, key_pull_weight: float = 0.5):
        super().__init__()
        self.backbone        = backbone
        self.pool            = pool
        self.key_pull_weight = key_pull_weight

    def _prepend_prompts(self, input_ids, attention_mask):
        wte      = _get_embed_layer(self.backbone)
        tok_emb  = wte(input_ids)
        prompts, pull = self.pool(tok_emb.mean(dim=1))
        B, P     = prompts.shape[0], prompts.shape[1]
        enc_emb  = torch.cat([prompts, tok_emb], dim=1)
        if attention_mask is not None:
            full_mask = torch.cat(
                [torch.ones(B, P, dtype=attention_mask.dtype, device=attention_mask.device),
                 attention_mask], dim=1,
            )
        else:
            full_mask = None
        return enc_emb, full_mask, pull

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        enc_emb, full_mask, pull = self._prepend_prompts(input_ids, attention_mask)
        out = self.backbone(
            inputs_embeds=enc_emb, attention_mask=full_mask,
            labels=labels, return_dict=True,
        )
        if labels is not None and pull is not None:
            out.loss = out.loss + self.key_pull_weight * pull
        return out

    def generate(self, input_ids, attention_mask=None, **kwargs):
        enc_emb, full_mask, _ = self._prepend_prompts(input_ids, attention_mask)
        return self.backbone.generate(
            inputs_embeds=enc_emb, attention_mask=full_mask, **kwargs,
        )

    def gradient_checkpointing_enable(self, **kw):
        """Forward Trainer's grad-ckpt call into the backbone + enable input grads
        (needed because the backbone is frozen but its output must carry gradient
        back to the prompt pool)."""
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kw)
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()


@dataclass
class B4L2P:
    cfg:              INCAConfig
    pool_size:        int   = 10
    prompt_len:       int   = 5
    top_n:            int   = 3
    key_pull_weight:  float = 0.5
    name:             str   = "b4_l2p"

    _wrapper:   Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device:    str = field(init=False, default="")

    def build_model(self, tokenizer, device: str):
        self._tokenizer = tokenizer
        self._device    = device
        load_kw = {"dtype": model_dtype(device, self.cfg)}
        if "cuda" in device:
            load_kw["device_map"] = "auto"
        backbone = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
        if "device_map" not in load_kw:
            backbone = backbone.to(device)
        if hasattr(backbone.config, "pad_token_id"):
            backbone.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

        pool = PromptPool(_get_embed_dim(backbone),
                          self.pool_size, self.prompt_len, self.top_n).to(device)
        self._wrapper = L2PWrapper(backbone, pool, self.key_pull_weight).to(device)
        return self._wrapper

    def scoring_model(self):
        return self._wrapper

    def on_period_start(self, period_label, period_idx): pass
    def on_period_end(self, period_label, period_idx, raw_items): pass

    def make_trainer(self, args, raw_items, tokenizer, period_label, period_idx):
        max_in = self.cfg.max_input_length
        max_lb = getattr(self.cfg, "max_target_length", max_in)
        train_ds = TokenizedDataset(raw_items, tokenizer, max_in, max_lb)
        return standard_trainer(self._wrapper, args, train_ds, tokenizer)


def main() -> None:
    p = argparse.ArgumentParser(description="B4 L2P baseline (Trainer-based)")
    p.add_argument("--config", required=True)
    p.add_argument("--pool_size", type=int, default=10)
    p.add_argument("--prompt_len", type=int, default=5)
    p.add_argument("--top_n", type=int, default=3)
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
    TrainerRunner(
        cfg,
        B4L2P(cfg=cfg, pool_size=args.pool_size, prompt_len=args.prompt_len,
              top_n=args.top_n),
        device=args.device,
    ).run()


if __name__ == "__main__":
    main()

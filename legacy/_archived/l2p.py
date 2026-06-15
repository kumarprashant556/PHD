"""B4 — Learning to Prompt (L2P). Adapted from google-research/l2p.

Keeps the backbone frozen; learns a prompt pool that is prepended to the
encoder input embeddings. Only the pool parameters are updated.
"""

from __future__ import annotations
import argparse, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch, torch.nn as nn, torch.nn.functional as F, yaml
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from models.inca.config import INCAConfig
from training.cl_runner import (
    Period, make_loader, model_dtype,
    make_epoch_bar, make_batch_bar, BaselineRunner,
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
        if hasattr(model.config, attr): return getattr(model.config, attr)
    raise AttributeError(f"Cannot find embed dim in {type(model).__name__}")


def _get_embed_layer(model):
    if hasattr(model, "shared"): return model.shared
    if hasattr(model, "encoder") and hasattr(model.encoder, "embed_tokens"):
        return model.encoder.embed_tokens
    raise AttributeError(f"Cannot find embedding layer in {type(model).__name__}")


class L2PWrapper(nn.Module):
    """Backbone + PromptPool in a single nn.Module for checkpointing and scoring.

    B4L2P.scoring_model() returns this wrapper so that:
      (a) torch.save(wrapper.state_dict()) captures both pool and backbone.
      (b) _eval_cloze_accuracy() can call wrapper.generate() with prompts prepended.
    """

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
        enc_emb, full_mask, _ = self._prepend_prompts(input_ids, attention_mask)
        return self.backbone(
            inputs_embeds=enc_emb, attention_mask=full_mask,
            labels=labels, return_dict=True,
        )

    def generate(self, input_ids, attention_mask=None, **kwargs):
        enc_emb, full_mask, _ = self._prepend_prompts(input_ids, attention_mask)
        return self.backbone.generate(
            inputs_embeds=enc_emb, attention_mask=full_mask, **kwargs
        )


@dataclass
class B4L2P:
    cfg:              INCAConfig
    pool_size:        int   = 10
    prompt_len:       int   = 5
    top_n:            int   = 3
    key_pull_weight:  float = 0.5
    name:             str   = "b4_l2p"
    extras:           Dict[str, Any] = field(default_factory=dict)

    _backbone:  Any = field(init=False, default=None)
    _pool:      Any = field(init=False, default=None)
    _optimizer: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device:    str = field(init=False, default="")

    def build_model(self, tokenizer, device: str):
        self._tokenizer = tokenizer
        self._device    = device
        load_kw = {"torch_dtype": model_dtype(device)}
        if "cuda" in device: load_kw["device_map"] = "auto"
        backbone = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
        if "device_map" not in load_kw: backbone = backbone.to(device)
        if hasattr(backbone.config, "pad_token_id"):
            backbone.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        for p in backbone.parameters(): p.requires_grad = False
        backbone.eval()

        pool = PromptPool(_get_embed_dim(backbone), self.pool_size,
                          self.prompt_len, self.top_n).to(device)
        self._backbone  = backbone
        self._pool      = pool
        self._optimizer = AdamW(pool.parameters(), lr=self.cfg.lr)
        self.extras.update(pool_size=self.pool_size, prompt_len=self.prompt_len)
        return backbone

    def scoring_model(self):
        # Return wrapper so state_dict captures pool params and generate() uses prompts.
        return L2PWrapper(self._backbone, self._pool, self.key_pull_weight)
    def on_period_start(self, period: Period): pass
    def on_period_end(self, period: Period): pass

    def _forward(self, batch):
        device = self._device
        ids    = batch["input_ids"].to(device)
        mask   = batch.get("attention_mask")
        if mask is not None: mask = mask.to(device)
        labels = batch["labels"].to(device)

        wte     = _get_embed_layer(self._backbone)
        tok_emb = wte(ids)
        prompts, pull = self._pool(tok_emb.mean(dim=1))
        B, P    = prompts.shape[0], prompts.shape[1]
        enc_emb = torch.cat([prompts, tok_emb], dim=1)
        if mask is not None:
            full_mask = torch.cat([torch.ones(B, P, dtype=mask.dtype, device=mask.device), mask], dim=1)
        else:
            full_mask = None

        out = self._backbone(inputs_embeds=enc_emb, attention_mask=full_mask,
                             labels=labels, return_dict=True)
        return out.loss, pull

    def train_period(self, period, scheduler=None, loss_logger=None, text_logger=None):
        cfg      = self.cfg
        max_seq  = cfg.max_input_length
        max_ans  = getattr(cfg, "max_target_length", max_seq)
        accum    = max(1, getattr(cfg, "grad_accum_steps", 1))
        log_every = max(1, getattr(cfg, "log_every_n_steps", 50))
        max_grad  = getattr(cfg, "max_grad_norm", 1.0)

        dl = make_loader(period.train_items, self._tokenizer, batch_size=cfg.batch_size,
                         max_seq_len=max_seq, max_answer_len=max_ans)
        if len(dl) == 0: return 0.0

        n_periods = getattr(cfg, "max_periods", 99)
        epoch_bar = make_epoch_bar(cfg.epochs_per_period, period.label, period.index, n_periods)
        last_loss = 0.0; opt_step = 0

        for epoch in epoch_bar:
            self._pool.train(); total = n = 0; accum_loss = 0.0
            batch_bar = make_batch_bar(dl, epoch, cfg.epochs_per_period)
            for ms, batch in enumerate(batch_bar, 1):
                ce, pull = self._forward(batch)
                if not torch.isfinite(ce): continue
                ((ce + self.key_pull_weight * pull) / accum).backward()
                accum_loss += ce.item()
                if ms % accum == 0 or ms == len(dl):
                    torch.nn.utils.clip_grad_norm_(self._pool.parameters(), max_grad)
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    if scheduler: scheduler.step()
                    opt_step += 1; sl = accum_loss / accum; accum_loss = 0.0
                    total += sl; n += 1
                    if loss_logger and opt_step % log_every == 0:
                        loss_logger(period.label, epoch, opt_step, sl)
            last_loss = total / max(n, 1)
            if loss_logger: loss_logger(period.label, epoch, opt_step, last_loss)
        return last_loss


def main():
    p = argparse.ArgumentParser(description="B4 L2P baseline")
    p.add_argument("--config", required=True); p.add_argument("--pool_size", type=int, default=10)
    p.add_argument("--prompt_len", type=int, default=5); p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--dataset", default=None); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    with open(args.config) as f: cfg_dict = yaml.safe_load(f) or {}
    if args.dataset: cfg_dict["dataset"] = args.dataset
    if args.seed:    cfg_dict["seed"]    = args.seed
    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items() if k in INCAConfig.__dataclass_fields__})
    BaselineRunner(cfg, B4L2P(cfg=cfg, pool_size=args.pool_size, prompt_len=args.prompt_len,
                               top_n=args.top_n), device=args.device).run()

if __name__ == "__main__": main()

"""B4 — Learning to Prompt (L2P). Adapted from google-research/l2p.

Supports seq2seq (FLAN-T5) and causal-LM via cfg.model_type.
For seq2seq: prompts are prepended to the encoder input embeddings.
"""

from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from Phase0.common.config import Phase0Config, load_config, model_dtype
from Phase0.common.harness import Period, make_loader
from Phase0.common.progress import make_epoch_bar, make_batch_bar
from Phase0.common.runner import BaselineRunner


class PromptPool(nn.Module):
    def __init__(self, embed_dim: int, pool_size: int = 10,
                 prompt_len: int = 5, top_n: int = 3):
        super().__init__()
        self.pool_size  = pool_size
        self.prompt_len = prompt_len
        self.top_n      = min(top_n, pool_size)
        self.keys    = nn.Parameter(torch.randn(pool_size, embed_dim) * 0.02)
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, embed_dim) * 0.02)

    def forward(self, query: torch.Tensor):
        q = F.normalize(query, dim=-1)
        k = F.normalize(self.keys, dim=-1)
        sim = q @ k.t()
        topv, topi = sim.topk(self.top_n, dim=-1)
        B = query.shape[0]
        p = self.prompts[topi].reshape(B, self.top_n * self.prompt_len, -1)
        return p, -topv.mean()


def _get_embed_dim(model):
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
    raise AttributeError(f"Cannot find embed dim in {type(model).__name__}")


def _get_embedding_layer(model):
    """Return the token embedding layer for any architecture."""
    # Seq2seq (T5): shared embedding
    if hasattr(model, "shared"):
        return model.shared
    if hasattr(model, "encoder") and hasattr(model.encoder, "embed_tokens"):
        return model.encoder.embed_tokens
    # Causal (GPT-2, pythia)
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "embed_tokens"):
            return inner.decoder.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "word_embeddings"):
        return model.transformer.word_embeddings
    raise AttributeError(f"Cannot find embedding layer in {type(model).__name__}")


@dataclass
class B4L2P:
    cfg: Phase0Config
    pool_size: int = 10
    prompt_len: int = 5
    top_n: int = 3
    key_pull_weight: float = 0.5
    name: str = "b4_l2p"
    extras: Dict[str, Any] = field(default_factory=dict)

    _backbone: Any = field(init=False, default=None)
    _pool: Any = field(init=False, default=None)
    _optimizer: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device: str = field(init=False, default="")

    def build_model(self, tokenizer, device: str):
        self._tokenizer = tokenizer
        self._device    = device
        load_kw: Dict[str, Any] = {"torch_dtype": model_dtype(device)}
        if "cuda" in device:
            load_kw["device_map"] = "auto"
        if self.cfg.model_type == "seq2seq":
            backbone = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
        else:
            backbone = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kw)
        if "device_map" not in load_kw:
            backbone = backbone.to(device)
        if hasattr(backbone.config, "pad_token_id"):
            backbone.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

        pool = PromptPool(_get_embed_dim(backbone), self.pool_size,
                          self.prompt_len, self.top_n).to(device)
        self._backbone  = backbone
        self._pool      = pool
        self._optimizer = AdamW(pool.parameters(), lr=self.cfg.lr)
        self.extras.update(pool_size=self.pool_size, prompt_len=self.prompt_len,
                           top_n=self.top_n, key_pull_weight=self.key_pull_weight)
        return backbone

    def scoring_model(self):
        return self._backbone

    def on_period_start(self, period: Period) -> None:
        return None

    def _forward_with_prompts(self, batch) -> tuple:
        """Run forward pass with prompt prepended. Returns (ce_loss, pull_loss)."""
        device = self._device
        cfg    = self.cfg

        if cfg.model_type == "seq2seq":
            ids    = batch["input_ids"].to(device)
            mask   = batch.get("attention_mask")
            if mask is not None: mask = mask.to(device)
            labels = batch["labels"].to(device)

            # Prepend prompts to encoder embeddings
            wte    = _get_embedding_layer(self._backbone)
            tok_emb = wte(ids)
            query   = tok_emb.mean(dim=1)
            prompts, pull = self._pool(query)
            B, P = prompts.shape[0], prompts.shape[1]
            enc_emb = torch.cat([prompts, tok_emb], dim=1)

            if mask is not None:
                prompt_mask = torch.ones(B, P, dtype=mask.dtype, device=mask.device)
                full_mask   = torch.cat([prompt_mask, mask], dim=1)
            else:
                full_mask = None

            out = self._backbone(
                inputs_embeds=enc_emb,
                attention_mask=full_mask,
                labels=labels,
                return_dict=True,
            )
            return out.loss, pull

        else:
            ids  = batch["input_ids"].to(device)
            mask = batch.get("attention_mask")
            if mask is not None: mask = mask.to(device)

            wte     = _get_embedding_layer(self._backbone)
            tok_emb = wte(ids)
            query   = tok_emb.mean(dim=1)
            prompts, pull = self._pool(query)
            B, P = prompts.shape[0], prompts.shape[1]
            seq  = torch.cat([prompts, tok_emb], dim=1)

            if mask is not None:
                prompt_mask = torch.ones(B, P, dtype=mask.dtype, device=mask.device)
                full_mask   = torch.cat([prompt_mask, mask], dim=1)
            else:
                full_mask = None

            out    = self._backbone(inputs_embeds=seq, attention_mask=full_mask,
                                    return_dict=True)
            logits = out.logits
            labels = torch.full((B, seq.shape[1]), -100, dtype=torch.long, device=ids.device)
            content_labels = ids.clone()
            if mask is not None:
                content_labels[mask == 0] = -100
            labels[:, P:] = content_labels
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1), ignore_index=-100,
            )
            return ce, pull

    def train_period(self, period: Period,
                     scheduler=None, loss_logger=None, text_logger=None) -> float:
        cfg = self.cfg
        dl  = make_loader(
            period.train_items, self._tokenizer,
            batch_size=cfg.batch_size, max_seq_len=cfg.max_seq_len,
            shuffle=True, model_type=cfg.model_type,
            max_answer_len=cfg.max_answer_len,
        )
        if len(dl) == 0:
            return 0.0

        n_epochs    = cfg.epochs_per_period
        accum_steps = max(1, getattr(cfg, "grad_accum_steps", 1))
        log_every   = max(1, getattr(cfg, "log_every_n_steps", 50))
        epoch_bar   = make_epoch_bar(n_epochs, period.label, period.index, cfg.max_periods)
        last_loss   = 0.0
        opt_step    = 0

        for epoch in epoch_bar:
            self._pool.train()
            total, n   = 0.0, 0
            accum_loss = 0.0
            batch_bar  = make_batch_bar(dl, epoch, n_epochs)

            for micro_step, batch in enumerate(batch_bar, 1):
                ce, pull = self._forward_with_prompts(batch)
                if not torch.isfinite(ce):
                    continue
                raw_loss    = ce + self.key_pull_weight * pull
                scaled_loss = raw_loss / accum_steps
                scaled_loss.backward()
                accum_loss += ce.item()

                is_accum_step = (
                    (micro_step % accum_steps == 0) or (micro_step == len(dl))
                )
                if is_accum_step:
                    torch.nn.utils.clip_grad_norm_(
                        self._pool.parameters(), cfg.max_grad_norm,
                    )
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    opt_step  += 1
                    step_loss  = accum_loss / accum_steps
                    accum_loss = 0.0
                    total     += step_loss
                    n         += 1
                    if self._device == "mps" and opt_step % log_every == 0:
                        torch.mps.empty_cache()
                    if loss_logger is not None and opt_step % log_every == 0:
                        loss_logger(period.label, epoch, opt_step, step_loss)
                    batch_bar.set_postfix(
                        ce=f"{step_loss:.4f}", pull=f"{pull.item():.4f}",
                        avg=f"{total/n:.4f}",
                        lr=f"{self._optimizer.param_groups[0]['lr']:.2e}",
                    )

            last_loss = total / max(n, 1)
            epoch_bar.set_postfix(avg_loss=f"{last_loss:.4f}")
            if loss_logger is not None:
                loss_logger(period.label, epoch, opt_step, last_loss)

        return last_loss

    def on_period_end(self, period: Period) -> None:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="B4 L2P baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--pool_size", type=int, default=10)
    p.add_argument("--prompt_len", type=int, default=5)
    p.add_argument("--top_n", type=int, default=3)
    p.add_argument("--key_pull_weight", type=float, default=0.5)
    p.add_argument("--model_type", default=None)
    args, _ = p.parse_known_args()
    ov  = {k: getattr(args, k) for k in ("model_type",) if getattr(args, k) is not None}
    cfg = load_config(args.config, overrides=ov)
    BaselineRunner(cfg, B4L2P(cfg=cfg, pool_size=args.pool_size, prompt_len=args.prompt_len,
                              top_n=args.top_n, key_pull_weight=args.key_pull_weight)).run()


if __name__ == "__main__":
    main()

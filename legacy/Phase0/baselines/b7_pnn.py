"""B7 — Progressive Neural Network (PNN). Adapted from Rusu et al. 2016.

Supports seq2seq (FLAN-T5) and causal-LM via cfg.model_type.
For seq2seq: lateral adapters inject past encoder hidden states into
the new column's encoder, then decoder runs normally.
"""

from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from Phase0.common.config import Phase0Config, load_config, model_dtype
from Phase0.common.harness import Period, make_loader
from Phase0.common.progress import make_epoch_bar, make_batch_bar
from Phase0.common.runner import BaselineRunner


def _get_embed_dim(model):
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(model.config, attr):
            return getattr(model.config, attr)
    raise AttributeError(f"Cannot find embed dim in {type(model).__name__}")


def _get_embedding_layer(model):
    if hasattr(model, "shared"):
        return model.shared
    if hasattr(model, "encoder") and hasattr(model.encoder, "embed_tokens"):
        return model.encoder.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
        if hasattr(inner, "decoder") and hasattr(inner.decoder, "embed_tokens"):
            return inner.decoder.embed_tokens
    raise AttributeError(f"Cannot find embedding layer in {type(model).__name__}")


class LateralAdapter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h):
        return torch.sigmoid(self.gate) * self.proj(h)


class PNNColumnCausal(nn.Module):
    def __init__(self, model_name, num_prev, device, load_kw):
        super().__init__()
        backbone = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
        if "device_map" not in load_kw:
            backbone = backbone.to(device)
        self.backbone = backbone
        dim = _get_embed_dim(backbone)
        self.laterals = nn.ModuleList(
            [LateralAdapter(dim).to(device) for _ in range(num_prev)]
        )

    def forward(self, input_ids, prev_hiddens, attention_mask=None):
        wte     = _get_embedding_layer(self.backbone)
        tok_emb = wte(input_ids)
        for h_prev, lat in zip(prev_hiddens, self.laterals):
            tok_emb = tok_emb + lat(h_prev)
        labels = input_ids.clone()
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        return self.backbone(inputs_embeds=tok_emb, attention_mask=attention_mask,
                             labels=labels, output_hidden_states=True)


class PNNColumnSeq2Seq(nn.Module):
    """PNN column for encoder-decoder (T5-family).

    Lateral adapters inject past encoder hidden states into the new column's
    encoder input embeddings before forwarding through the full model.
    """
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

    def forward(self, input_ids, labels, prev_enc_hiddens, attention_mask=None):
        wte     = _get_embedding_layer(self.backbone)
        tok_emb = wte(input_ids)
        for h_prev, lat in zip(prev_enc_hiddens, self.laterals):
            # h_prev may have different seq len; align by minimum
            t = min(tok_emb.shape[1], h_prev.shape[1])
            tok_emb[:, :t] = tok_emb[:, :t] + lat(h_prev[:, :t])
        return self.backbone(
            inputs_embeds=tok_emb,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )


@dataclass
class B7PNN:
    cfg: Phase0Config
    name: str = "b7_pnn"
    extras: Dict[str, Any] = field(default_factory=dict)

    _columns: List[Any] = field(init=False, default_factory=list)
    _optimizer: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device: str = field(init=False, default="")
    _load_kw: Dict[str, Any] = field(init=False, default_factory=dict)

    def build_model(self, tokenizer, device: str):
        self._tokenizer = tokenizer
        self._device    = device
        self._load_kw   = {"torch_dtype": model_dtype(device)}
        if "cuda" in device:
            self._load_kw["device_map"] = "auto"

        if self.cfg.model_type == "seq2seq":
            col0 = PNNColumnSeq2Seq(self.cfg.model_name, 0, device, self._load_kw)
        else:
            col0 = PNNColumnCausal(self.cfg.model_name, 0, device, self._load_kw)

        if hasattr(col0.backbone.config, "pad_token_id"):
            col0.backbone.config.pad_token_id = (tokenizer.pad_token_id
                                                 or tokenizer.eos_token_id)
        self._columns.append(col0)
        return col0.backbone

    def scoring_model(self):
        return self._columns[-1].backbone

    def on_period_start(self, period: Period) -> None:
        if period.index > 0:
            mt  = self.cfg.model_type
            n   = len(self._columns)
            col = (PNNColumnSeq2Seq if mt == "seq2seq" else PNNColumnCausal)(
                self.cfg.model_name, n, self._device, self._load_kw
            )
            if hasattr(col.backbone.config, "pad_token_id"):
                col.backbone.config.pad_token_id = (self._tokenizer.pad_token_id
                                                    or self._tokenizer.eos_token_id)
            self._columns.append(col)
        for col in self._columns[:-1]:
            for p in col.parameters(): p.requires_grad = False
            col.eval()
        for p in self._columns[-1].parameters():
            p.requires_grad = True
        trainable      = [p for p in self._columns[-1].parameters() if p.requires_grad]
        self._optimizer = AdamW(trainable, lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)

    def _get_prev_hiddens(self, input_ids, attention_mask=None):
        """Return last encoder hidden states from each frozen column."""
        prev = []
        with torch.no_grad():
            for col in self._columns[:-1]:
                if self.cfg.model_type == "seq2seq":
                    enc_out = col.backbone.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    prev.append(enc_out.last_hidden_state)
                else:
                    out = col.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    prev.append(out.hidden_states[-1])
        return prev

    def train_period(self, period: Period,
                     scheduler=None, loss_logger=None, text_logger=None) -> float:
        cfg    = self.cfg
        n_cols = len(self._columns)
        dl     = make_loader(
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
            self._columns[-1].train()
            total, n   = 0.0, 0
            accum_loss = 0.0
            batch_bar  = make_batch_bar(dl, epoch, n_epochs)

            for micro_step, batch in enumerate(batch_bar, 1):
                ids  = batch["input_ids"].to(self._device)
                mask = batch.get("attention_mask")
                if mask is not None: mask = mask.to(self._device)

                prev = self._get_prev_hiddens(ids, mask)

                if cfg.model_type == "seq2seq":
                    labels = batch["labels"].to(self._device)
                    out    = self._columns[-1](ids, labels, prev, mask)
                else:
                    out = self._columns[-1](ids, prev, mask)

                if not torch.isfinite(out.loss):
                    continue

                scaled_loss = out.loss / accum_steps
                scaled_loss.backward()
                accum_loss += out.loss.item()

                is_accum_step = (
                    (micro_step % accum_steps == 0) or (micro_step == len(dl))
                )
                if is_accum_step:
                    trainable = [p for p in self._columns[-1].parameters()
                                 if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
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
                        loss=f"{step_loss:.4f}", avg=f"{total/n:.4f}",
                        cols=n_cols,
                        lr=f"{self._optimizer.param_groups[0]['lr']:.2e}",
                    )

            last_loss = total / max(n, 1)
            epoch_bar.set_postfix(avg_loss=f"{last_loss:.4f}", columns=n_cols)
            if loss_logger is not None:
                loss_logger(period.label, epoch, opt_step, last_loss)

        return last_loss

    def on_period_end(self, period: Period) -> None:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="B7 PNN baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--model_type", default=None)
    args, _ = p.parse_known_args()
    ov  = {k: getattr(args, k) for k in ("model_type",) if getattr(args, k) is not None}
    cfg = load_config(args.config, overrides=ov)
    BaselineRunner(cfg, B7PNN(cfg=cfg)).run()


if __name__ == "__main__":
    main()

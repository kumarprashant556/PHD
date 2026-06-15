"""B7 — Progressive Neural Network (PNN). Adapted from Rusu et al. 2016.

Spawns a new FLAN-T5 column per period. Lateral adapters inject past encoder
hidden states into the new column. Zero forgetting by construction; parameter
cost grows with the number of periods.
"""

from __future__ import annotations
import argparse, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch, torch.nn as nn, yaml
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from models.inca.config import INCAConfig
from training.cl_runner import (
    Period, make_loader, model_dtype,
    make_epoch_bar, make_batch_bar, BaselineRunner,
)


def _get_embed_dim(model):
    for attr in ("d_model", "hidden_size", "n_embd"):
        if hasattr(model.config, attr): return getattr(model.config, attr)
    raise AttributeError(f"Cannot find embed dim in {type(model).__name__}")


def _get_embed_layer(model):
    if hasattr(model, "shared"): return model.shared
    if hasattr(model, "encoder") and hasattr(model.encoder, "embed_tokens"):
        return model.encoder.embed_tokens
    raise AttributeError(f"Cannot find embedding layer in {type(model).__name__}")


class LateralAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h):
        return torch.sigmoid(self.gate) * self.proj(h)


class PNNWrapper(nn.Module):
    """All PNN columns in one nn.Module for proper checkpointing and scoring.

    B7PNN.scoring_model() previously returned only the latest column's backbone,
    discarding lateral adapters and all prior columns.  This wrapper:
      (a) saves all columns in state_dict().
      (b) applies lateral contributions in generate() so eval matches training.
    """

    def __init__(self, columns: list):
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


class PNNColumn(nn.Module):
    """Single PNN column for seq2seq (FLAN-T5)."""
    def __init__(self, model_name, num_prev, device, load_kw):
        super().__init__()
        backbone = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kw)
        if "device_map" not in load_kw: backbone = backbone.to(device)
        self.backbone = backbone
        dim = _get_embed_dim(backbone)
        self.laterals = nn.ModuleList(
            [LateralAdapter(dim).to(device) for _ in range(num_prev)]
        )

    def forward(self, input_ids, labels, prev_enc_hiddens, attention_mask=None):
        wte     = _get_embed_layer(self.backbone)
        tok_emb = wte(input_ids)
        for h_prev, lat in zip(prev_enc_hiddens, self.laterals):
            t = min(tok_emb.shape[1], h_prev.shape[1])
            tok_emb[:, :t] = tok_emb[:, :t] + lat(h_prev[:, :t])
        return self.backbone(inputs_embeds=tok_emb, attention_mask=attention_mask,
                             labels=labels, output_hidden_states=True)


@dataclass
class B7PNN:
    cfg:     INCAConfig
    name:    str = "b7_pnn"
    extras:  Dict[str, Any] = field(default_factory=dict)

    _columns:   List = field(init=False, default_factory=list)
    _optimizer: Any  = field(init=False, default=None)
    _tokenizer: Any  = field(init=False, default=None)
    _device:    str  = field(init=False, default="")
    _load_kw:   Dict = field(init=False, default_factory=dict)

    def build_model(self, tokenizer, device: str):
        self._tokenizer = tokenizer
        self._device    = device
        self._load_kw   = {"torch_dtype": model_dtype(device)}
        if "cuda" in device: self._load_kw["device_map"] = "auto"
        col0 = PNNColumn(self.cfg.model_name, 0, device, self._load_kw)
        if hasattr(col0.backbone.config, "pad_token_id"):
            col0.backbone.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self._columns.append(col0)
        return col0.backbone

    def scoring_model(self):
        # Return full column stack so checkpoints include laterals + all columns,
        # and generate() applies the correct lateral-adapted embeddings.
        return PNNWrapper(self._columns)

    def on_period_start(self, period: Period):
        if period.index > 0:
            col = PNNColumn(self.cfg.model_name, len(self._columns),
                            self._device, self._load_kw)
            if hasattr(col.backbone.config, "pad_token_id"):
                col.backbone.config.pad_token_id = (self._tokenizer.pad_token_id
                                                    or self._tokenizer.eos_token_id)
            self._columns.append(col)
        for col in self._columns[:-1]:
            for p in col.parameters(): p.requires_grad = False
            col.eval()
        for p in self._columns[-1].parameters(): p.requires_grad = True
        self._optimizer = AdamW(
            [p for p in self._columns[-1].parameters() if p.requires_grad],
            lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
        )

    def _get_prev_hiddens(self, input_ids, attention_mask=None):
        prev = []
        with torch.no_grad():
            for col in self._columns[:-1]:
                enc_out = col.backbone.encoder(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, return_dict=True,
                )
                prev.append(enc_out.last_hidden_state)
        return prev

    def train_period(self, period, scheduler=None, loss_logger=None, text_logger=None):
        cfg      = self.cfg
        max_seq  = cfg.max_input_length
        max_ans  = getattr(cfg, "max_target_length", max_seq)
        accum    = max(1, getattr(cfg, "grad_accum_steps", 1))
        log_every = max(1, getattr(cfg, "log_every_n_steps", 50))
        max_grad  = getattr(cfg, "max_grad_norm", 1.0)
        n_cols    = len(self._columns)

        dl = make_loader(period.train_items, self._tokenizer, batch_size=cfg.batch_size,
                         max_seq_len=max_seq, max_answer_len=max_ans)
        if len(dl) == 0: return 0.0

        n_periods = getattr(cfg, "max_periods", 99)
        epoch_bar = make_epoch_bar(cfg.epochs_per_period, period.label, period.index, n_periods)
        last_loss = 0.0; opt_step = 0

        for epoch in epoch_bar:
            self._columns[-1].train(); total = n = 0; accum_loss = 0.0
            batch_bar = make_batch_bar(dl, epoch, cfg.epochs_per_period)
            for ms, batch in enumerate(batch_bar, 1):
                ids    = batch["input_ids"].to(self._device)
                mask   = batch.get("attention_mask")
                if mask is not None: mask = mask.to(self._device)
                labels = batch["labels"].to(self._device)
                prev   = self._get_prev_hiddens(ids, mask)
                out    = self._columns[-1](ids, labels, prev, mask)
                if not torch.isfinite(out.loss): continue
                (out.loss / accum).backward(); accum_loss += out.loss.item()
                if ms % accum == 0 or ms == len(dl):
                    trainable = [p for p in self._columns[-1].parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable, max_grad)
                    self._optimizer.step(); self._optimizer.zero_grad(set_to_none=True)
                    if scheduler: scheduler.step()
                    opt_step += 1; sl = accum_loss / accum; accum_loss = 0.0
                    total += sl; n += 1
                    if loss_logger and opt_step % log_every == 0:
                        loss_logger(period.label, epoch, opt_step, sl)
            last_loss = total / max(n, 1)
            if loss_logger: loss_logger(period.label, epoch, opt_step, last_loss)
        return last_loss

    def on_period_end(self, period: Period): pass


def main():
    p = argparse.ArgumentParser(description="B7 PNN baseline")
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", default=None); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    with open(args.config) as f: cfg_dict = yaml.safe_load(f) or {}
    if args.dataset: cfg_dict["dataset"] = args.dataset
    if args.seed:    cfg_dict["seed"]    = args.seed
    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items() if k in INCAConfig.__dataclass_fields__})
    BaselineRunner(cfg, B7PNN(cfg=cfg), device=args.device).run()

if __name__ == "__main__": main()

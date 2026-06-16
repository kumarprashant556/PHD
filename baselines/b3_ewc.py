"""B3 — Online Elastic Weight Consolidation (EWC) baseline (Trainer-based).

Computes diagonal Fisher after each period and adds a quadratic regularisation
penalty to the loss so weights important for earlier periods aren't overwritten.

Memory cost: O(2 × params) regardless of the number of periods (single
accumulated snapshot with gamma decay).  Fisher + theta* stored on CPU.

Implementation:
  - ``EWCTrainer`` subclasses Seq2SeqTrainer and overrides ``compute_loss`` to
    add the EWC penalty.
  - Fisher is refreshed in ``on_period_end`` using the trainer's collator on a
    small sample of the just-finished period.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer

# Repo root on sys.path so `baselines._runtime` resolves whether invoked as
# `python baselines/b3_ewc.py` (direct) or via the sweep launcher.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from baselines._runtime import (
    INCAConfig, TrainerRunner, TokenizedDataset,
    model_dtype, _trainer_kwargs,
)


# ── Custom Trainer with EWC penalty ───────────────────────────────────────────

class EWCTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with an EWC penalty added to ``compute_loss``."""

    def __init__(self, *args, ewc_baseline=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ewc_baseline = ewc_baseline

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss + self._ewc_baseline._ewc_penalty()
        return (loss, outputs) if return_outputs else loss


# ── Fisher computation (diagonal) ─────────────────────────────────────────────

def _compute_fisher(
    model,
    data_loader,
    device: str,
    max_batches: int = 200,
) -> Dict[str, torch.Tensor]:
    fisher: Dict[str, torch.Tensor] = {
        n: torch.zeros_like(p, device=device)
        for n, p in model.named_parameters() if p.requires_grad
    }
    model.eval()
    seen = 0
    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        model.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        out = model(**batch)
        if not torch.isfinite(out.loss):
            continue
        out.loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        seen += 1
    model.zero_grad()
    if seen > 0:
        for n in fisher:
            fisher[n] /= seen
    return fisher


# ── B3 baseline ───────────────────────────────────────────────────────────────

@dataclass
class B3EWC:
    cfg:                 INCAConfig
    lambda_ewc:          float = 100.0
    fisher_max_batches:  int   = 200
    fisher_gamma:        float = 0.9     # Online: F_accum = gamma*F_prev + F_new
    name:                str   = "b3_ewc"

    _model:      Any = field(init=False, default=None)
    _tokenizer:  Any = field(init=False, default=None)
    _device:     str = field(init=False, default="")
    # Online EWC: single accumulated snapshot on CPU.
    _fisher_cpu: Dict = field(init=False, default_factory=dict)
    _theta_cpu:  Dict = field(init=False, default_factory=dict)

    def build_model(self, tokenizer, device: str):
        self._tokenizer = tokenizer
        self._device    = device
        load_kw = {"dtype": model_dtype(device, self.cfg)}
        if "cuda" in device:
            load_kw["device_map"] = "auto"
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
        if "device_map" not in load_kw:
            model = model.to(device)
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self._model = model
        return model

    def scoring_model(self):
        return self._model

    def on_period_start(self, period_label, period_idx):
        pass

    def _ewc_penalty(self) -> torch.Tensor:
        if not self._fisher_cpu:
            return torch.tensor(0.0, device=self._device)
        penalty = torch.tensor(0.0, device=self._device)
        for n, p in self._model.named_parameters():
            if n in self._fisher_cpu and p.requires_grad:
                f = self._fisher_cpu[n].to(p.device)
                t = self._theta_cpu[n].to(p.device)
                penalty = penalty + (f * (p - t) ** 2).sum()
        return (self.lambda_ewc / 2.0) * penalty

    def make_trainer(self, args, raw_items, tokenizer, period_label, period_idx):
        max_in = self.cfg.max_input_length
        max_lb = getattr(self.cfg, "max_target_length", max_in)
        train_ds = TokenizedDataset(raw_items, tokenizer, max_in, max_lb)
        collator = DataCollatorForSeq2Seq(
            tokenizer, model=self._model, padding="longest", label_pad_token_id=-100,
        )
        return EWCTrainer(
            model=self._model,
            args=args,
            train_dataset=train_ds,
            data_collator=collator,
            ewc_baseline=self,
            **_trainer_kwargs(self._model, tokenizer),
        )

    def on_period_end(self, period_label, period_idx, raw_items):
        """Refresh Fisher (online accumulation with gamma decay) and theta*."""
        max_in = self.cfg.max_input_length
        max_lb = getattr(self.cfg, "max_target_length", max_in)
        # Use a small slice for Fisher — full set is overkill and slow.
        sample = raw_items[: max(self.fisher_max_batches * self.cfg.batch_size, 1)]
        ds = TokenizedDataset(sample, self._tokenizer, max_in, max_lb)
        collator = DataCollatorForSeq2Seq(
            self._tokenizer, model=self._model, padding="longest", label_pad_token_id=-100,
        )
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=collator)
        new_fisher = _compute_fisher(self._model, dl, self._device, self.fisher_max_batches)

        # Online accumulation: F = gamma * F_prev + F_new
        for n, f_new in new_fisher.items():
            f_prev = self._fisher_cpu.get(n, torch.zeros_like(f_new, device="cpu"))
            self._fisher_cpu[n] = self.fisher_gamma * f_prev + f_new.cpu()

        # Update theta* to current weights (on CPU).
        self._theta_cpu = {
            n: p.detach().cpu().clone()
            for n, p in self._model.named_parameters() if p.requires_grad
        }


def main() -> None:
    p = argparse.ArgumentParser(description="B3 Online EWC baseline (Trainer-based)")
    p.add_argument("--config", required=True)
    p.add_argument("--lambda_ewc", type=float, default=100.0)
    p.add_argument("--fisher_max_batches", type=int, default=200)
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
        B3EWC(cfg=cfg, lambda_ewc=args.lambda_ewc,
              fisher_max_batches=args.fisher_max_batches),
        device=args.device,
    ).run()


if __name__ == "__main__":
    main()

"""B3 — Elastic Weight Consolidation (EWC). Adapted from ContinualAI/avalanche.

Computes diagonal Fisher after each period and adds a regularisation penalty
to prevent overwriting weights important for earlier periods.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch, yaml
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM

from runner import (
    INCAConfig, Period, make_loader, model_dtype,
    seq2seq_train_loop, BaselineRunner,
)


def compute_fisher(model, data_loader, device, max_batches=200):
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if p.requires_grad}
    model.eval(); seen = 0
    for i, batch in enumerate(data_loader):
        if i >= max_batches: break
        model.zero_grad()
        ids    = batch["input_ids"].to(device)
        mask   = batch.get("attention_mask")
        if mask is not None: mask = mask.to(device)
        labels = batch["labels"].to(device)
        out    = model(input_ids=ids, attention_mask=mask, labels=labels)
        if not torch.isfinite(out.loss): continue
        out.loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        seen += 1
    model.zero_grad()
    if seen > 0:
        for n in fisher: fisher[n] /= seen
    return fisher


@dataclass
class B3EWC:
    cfg:                 INCAConfig
    lambda_ewc:          float = 100.0
    fisher_max_batches:  int   = 200
    fisher_gamma:        float = 0.9   # Online EWC decay: accumulated = gamma*old + new
    name:                str   = "b3_ewc"
    extras:              Dict[str, Any] = field(default_factory=dict)

    _model:      Any  = field(init=False, default=None)
    _optimizer:  Any  = field(init=False, default=None)
    _tokenizer:  Any  = field(init=False, default=None)
    _device:     str  = field(init=False, default="")
    # Online EWC: single accumulated snapshot instead of per-period list.
    # Fisher and theta_star stored on CPU to save GPU memory.
    _fisher_cpu: Dict = field(init=False, default_factory=dict)   # accumulated Fisher (CPU)
    _theta_cpu:  Dict = field(init=False, default_factory=dict)   # reference params (CPU)
    _snapshots:  List = field(init=False, default_factory=list)   # legacy field kept for compat

    def build_model(self, tokenizer, device: str):
        self._tokenizer = tokenizer
        self._device    = device
        load_kw = {"dtype": model_dtype(device)}
        if "cuda" in device: load_kw["device_map"] = "auto"
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
        if "device_map" not in load_kw: model = model.to(device)
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self._model     = model
        self._optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.extras.update(lambda_ewc=self.lambda_ewc)
        return model

    def scoring_model(self): return self._model
    def on_period_start(self, period: Period): pass

    def _ewc_penalty(self):
        """Online EWC penalty: single accumulated (Fisher, theta*) snapshot on CPU."""
        if not self._fisher_cpu:
            return torch.tensor(0.0, device=self._device)
        penalty = torch.tensor(0.0, device=self._device)
        for n, p in self._model.named_parameters():
            if n in self._fisher_cpu and p.requires_grad:
                f    = self._fisher_cpu[n].to(p.device)
                t    = self._theta_cpu[n].to(p.device)
                penalty = penalty + (f * (p - t) ** 2).sum()
        return (self.lambda_ewc / 2.0) * penalty

    def train_period(self, period, scheduler=None, loss_logger=None, text_logger=None):
        cfg = self.cfg
        dl  = make_loader(period.train_items, self._tokenizer, batch_size=cfg.batch_size,
                          max_seq_len=cfg.max_input_length,
                          max_answer_len=getattr(cfg, "max_target_length", cfg.max_input_length))
        return seq2seq_train_loop(
            model=self._model, optimizer=self._optimizer, dataloader=dl,
            device=self._device, cfg=cfg, period_label=period.label,
            period_idx=period.index, n_periods=getattr(cfg, "max_periods", 99),
            extra_loss_fn=self._ewc_penalty,
            scheduler=scheduler, loss_logger=loss_logger)

    def on_period_end(self, period: Period):
        """Online EWC: accumulate Fisher with gamma decay; store theta* and Fisher on CPU.

        Memory cost: always O(2 × params), regardless of the number of periods.
        Each period's Fisher is: F_accum = gamma * F_accum_prev + F_new
        theta_star is updated to the current weights after each period.
        """
        cfg = self.cfg
        dl  = make_loader(period.train_items, self._tokenizer, batch_size=cfg.batch_size,
                          max_seq_len=cfg.max_input_length, shuffle=False,
                          max_answer_len=getattr(cfg, "max_target_length", cfg.max_input_length))
        new_fisher = compute_fisher(self._model, dl, self._device, self.fisher_max_batches)

        # Online accumulation: F = gamma * F_prev + F_new  (stays as one snapshot)
        for n, f_new in new_fisher.items():
            f_prev = self._fisher_cpu.get(n, torch.zeros_like(f_new, device="cpu"))
            self._fisher_cpu[n] = self.fisher_gamma * f_prev + f_new.cpu()

        # Update theta* to current weights (on CPU to save GPU memory)
        self._theta_cpu = {
            n: p.detach().cpu().clone()
            for n, p in self._model.named_parameters() if p.requires_grad
        }


def main():
    p = argparse.ArgumentParser(description="B3 EWC baseline")
    p.add_argument("--config", required=True); p.add_argument("--lambda_ewc", type=float, default=100.0)
    p.add_argument("--fisher_max_batches", type=int, default=200)
    p.add_argument("--dataset", default=None); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    with open(args.config) as f: cfg_dict = yaml.safe_load(f) or {}
    if args.dataset: cfg_dict["dataset"] = args.dataset
    if args.seed:    cfg_dict["seed"]    = args.seed
    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items() if k in INCAConfig.__dataclass_fields__})
    BaselineRunner(cfg, B3EWC(cfg=cfg, lambda_ewc=args.lambda_ewc,
                              fisher_max_batches=args.fisher_max_batches), device=args.device).run()

if __name__ == "__main__": main()

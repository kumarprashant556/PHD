"""B3 — Elastic Weight Consolidation (EWC). Adapted from ContinualAI/avalanche.

Supports both seq2seq (FLAN-T5) and causal-LM via cfg.model_type.
Fisher is computed on the seq2seq CE loss (encoder+decoder) for T5.
"""

from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from Phase0.common.config import Phase0Config, load_config, model_dtype
from Phase0.common.harness import Period, make_loader
from Phase0.common.progress import standard_train_loop, seq2seq_train_loop
from Phase0.common.runner import BaselineRunner


def compute_fisher(
    model, data_loader, device: str, max_batches: int = 200,
    model_type: str = "seq2seq",
) -> Dict[str, torch.Tensor]:
    """Diagonal Fisher via backprop through the model's CE loss.

    For seq2seq, forward pass uses (input_ids, attention_mask, labels).
    For causal LM, forward pass uses (input_ids, labels=input_ids).
    """
    fisher = {n: torch.zeros_like(p, device=device)
              for n, p in model.named_parameters() if p.requires_grad}
    was_training = model.training
    model.eval()
    seen = 0
    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break
        model.zero_grad()
        if model_type == "seq2seq":
            ids    = batch["input_ids"].to(device)
            mask   = batch.get("attention_mask")
            if mask is not None: mask = mask.to(device)
            labels = batch["labels"].to(device)
            out    = model(input_ids=ids, attention_mask=mask, labels=labels)
        else:
            ids = batch["input_ids"].to(device)
            out = model(input_ids=ids, labels=ids.clone())
        if not torch.isfinite(out.loss):
            continue
        out.loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        seen += 1
    model.zero_grad()
    if was_training:
        model.train()
    if seen > 0:
        for n in fisher:
            fisher[n] /= seen
    return fisher


@dataclass
class B3EWC:
    cfg: Phase0Config
    lambda_ewc: float = 100.0
    fisher_max_batches: int = 200
    name: str = "b3_ewc"
    extras: Dict[str, Any] = field(default_factory=dict)

    _model: Any = field(init=False, default=None)
    _optimizer: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device: str = field(init=False, default="")
    _snapshots: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = \
        field(init=False, default_factory=list)

    def build_model(self, tokenizer, device: str) -> torch.nn.Module:
        self._tokenizer = tokenizer
        self._device    = device
        load_kw: Dict[str, Any] = {"torch_dtype": model_dtype(device)}
        if "cuda" in device:
            load_kw["device_map"] = "auto"
        if self.cfg.model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kw)
        if "device_map" not in load_kw:
            model = model.to(device)
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self._model     = model
        self._optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
        )
        self.extras.update(lambda_ewc=self.lambda_ewc,
                           fisher_max_batches=self.fisher_max_batches)
        return model

    def scoring_model(self) -> torch.nn.Module:
        return self._model

    def on_period_start(self, period: Period) -> None:
        return None

    def _ewc_penalty(self) -> torch.Tensor:
        if not self._snapshots:
            return torch.tensor(0.0, device=self._device)
        penalty = torch.tensor(0.0, device=self._device)
        for fisher, theta_star in self._snapshots:
            for n, p in self._model.named_parameters():
                if n in fisher and p.requires_grad:
                    penalty = penalty + (
                        fisher[n].to(p.device) * (p - theta_star[n].to(p.device)) ** 2
                    ).sum()
        return (self.lambda_ewc / 2.0) * penalty

    def train_period(self, period: Period,
                     scheduler=None, loss_logger=None, text_logger=None) -> float:
        cfg  = self.cfg
        dl   = make_loader(
            period.train_items, self._tokenizer,
            batch_size=cfg.batch_size, max_seq_len=cfg.max_seq_len,
            shuffle=True, model_type=cfg.model_type,
            max_answer_len=cfg.max_answer_len,
        )
        loop = seq2seq_train_loop if cfg.model_type == "seq2seq" else standard_train_loop
        return loop(
            model=self._model, optimizer=self._optimizer, dataloader=dl,
            device=self._device, cfg=cfg,
            period_label=period.label, period_idx=period.index,
            n_periods=cfg.max_periods,
            extra_loss_fn=self._ewc_penalty,
            scheduler=scheduler, loss_logger=loss_logger,
        )

    def on_period_end(self, period: Period) -> None:
        cfg = self.cfg
        dl  = make_loader(
            period.train_items, self._tokenizer,
            batch_size=cfg.batch_size, max_seq_len=cfg.max_seq_len,
            shuffle=False, model_type=cfg.model_type,
            max_answer_len=cfg.max_answer_len,
        )
        fisher = compute_fisher(
            self._model, dl, self._device,
            self.fisher_max_batches, model_type=cfg.model_type,
        )
        theta_star = {n: p.detach().clone()
                      for n, p in self._model.named_parameters() if p.requires_grad}
        self._snapshots.append((fisher, theta_star))


def main() -> None:
    p = argparse.ArgumentParser(description="B3 EWC baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--lambda_ewc", type=float, default=100.0)
    p.add_argument("--fisher_max_batches", type=int, default=200)
    p.add_argument("--model_type", default=None)
    args, _ = p.parse_known_args()
    ov = {k: getattr(args, k) for k in ("model_type",) if getattr(args, k) is not None}
    cfg = load_config(args.config, overrides=ov)
    BaselineRunner(cfg, B3EWC(cfg=cfg, lambda_ewc=args.lambda_ewc,
                              fisher_max_batches=args.fisher_max_batches)).run()


if __name__ == "__main__":
    main()

"""B1 — Naive sequential fine-tuning baseline (forgetting floor).

Supports both seq2seq (FLAN-T5) and causal-LM (pythia/GPT) via cfg.model_type.
"""

from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.optimization import Adafactor

from Phase0.common.config import Phase0Config, load_config, model_dtype
from Phase0.common.harness import Period, make_loader
from Phase0.common.progress import standard_train_loop, seq2seq_train_loop
from Phase0.common.runner import BaselineRunner


def _load_model(cfg: Phase0Config, device: str):
    load_kw: Dict[str, Any] = {"torch_dtype": model_dtype(device)}
    if "cuda" in device:
        load_kw["device_map"] = "auto"
    if cfg.model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, **load_kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **load_kw)
    if "device_map" not in load_kw:
        model = model.to(device)
    # Gradient checkpointing: recomputes activations during backward to save VRAM.
    # Adds ~20% compute overhead but can halve activation memory.
    if getattr(cfg, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        # Required so that inputs always need grad (enables checkpointing on first layer)
        model.enable_input_require_grads()
        print(f"  [b1] gradient checkpointing enabled")
    return model


@dataclass
class B1NaiveFinetune:
    cfg: Phase0Config
    name: str = "b1_finetune"
    extras: Dict[str, Any] = field(default_factory=dict)

    _model: Any = field(init=False, default=None)
    _optimizer: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device: str = field(init=False, default="")

    def build_model(self, tokenizer, device: str) -> torch.nn.Module:
        self._tokenizer = tokenizer
        self._device    = device
        model = _load_model(self.cfg, device)
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self._model = model
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if getattr(self.cfg, "use_adafactor", False):
            # Adafactor: factored second moments + no first moment = ~3–4× lower
            # optimizer memory than AdamW.  Use explicit LR (relative_step=False).
            self._optimizer = Adafactor(
                params,
                lr=self.cfg.lr,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
                weight_decay=self.cfg.weight_decay,
            )
            print(f"  [b1] using Adafactor optimizer (lr={self.cfg.lr})")
        else:
            self._optimizer = AdamW(
                params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
            )
        return model

    def scoring_model(self) -> torch.nn.Module:
        return self._model

    def on_period_start(self, period: Period) -> None:
        return None

    def train_period(self, period: Period,
                     scheduler=None, loss_logger=None, text_logger=None) -> float:
        cfg = self.cfg
        dl  = make_loader(
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
            scheduler=scheduler, loss_logger=loss_logger, text_logger=text_logger,
        )

    def on_period_end(self, period: Period) -> None:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="B1 Naive fine-tune")
    p.add_argument("--config", default=None)
    p.add_argument("--dataset", default=None); p.add_argument("--model_name", default=None)
    p.add_argument("--model_type", default=None)
    p.add_argument("--max_periods", type=int, default=None)
    p.add_argument("--epochs_per_period", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--max_train_probes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    ov = {k: getattr(args, k) for k in ("dataset", "model_name", "model_type",
          "max_periods", "epochs_per_period", "batch_size", "lr",
          "max_seq_len", "max_train_probes", "seed")
          if getattr(args, k) is not None}
    cfg = load_config(args.config, overrides=ov)
    BaselineRunner(cfg, B1NaiveFinetune(cfg=cfg)).run()


if __name__ == "__main__":
    main()

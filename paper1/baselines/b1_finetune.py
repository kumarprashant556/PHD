"""B1 — Naive sequential fine-tuning baseline (Trainer-based).

Fine-tunes FLAN-T5 on each period one at a time with no memory or protection.
The lower bound for CL comparisons (maximum catastrophic forgetting).
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml
from transformers import AutoModelForSeq2SeqLM

# Repo root on sys.path so `baselines._runtime` resolves whether invoked as
# `python baselines/b1_finetune.py` (direct) or via the sweep launcher.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from baselines._runtime import (
    INCAConfig, TrainerRunner, TokenizedDataset,
    model_dtype, standard_trainer,
)


@dataclass
class B1NaiveFinetune:
    cfg:  INCAConfig
    name: str = "b1_finetune"

    _model:     Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device:    str = field(init=False, default="")

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

    def on_period_start(self, period_label: str, period_idx: int) -> None:
        pass

    def on_period_end(self, period_label, period_idx, raw_items):
        pass

    def make_trainer(self, args, raw_items, tokenizer, period_label, period_idx):
        train_ds = TokenizedDataset(
            raw_items, tokenizer,
            self.cfg.max_input_length,
            getattr(self.cfg, "max_target_length", self.cfg.max_input_length),
        )
        return standard_trainer(self._model, args, train_ds, tokenizer)


def main() -> None:
    p = argparse.ArgumentParser(description="B1 Naive fine-tune (Trainer-based)")
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
    TrainerRunner(cfg, B1NaiveFinetune(cfg=cfg), device=args.device).run()


if __name__ == "__main__":
    main()

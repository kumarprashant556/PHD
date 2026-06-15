"""B1 — Naive sequential fine-tuning baseline (forgetting floor).

Fine-tunes FLAN-T5 on each period one at a time with no memory or protection.
Shows maximum catastrophic forgetting — the lower bound for CL comparisons.
"""

from __future__ import annotations
import argparse, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import torch, yaml
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM
from transformers.optimization import Adafactor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from models.inca.config import INCAConfig
from training.cl_runner import (
    Period, make_loader, model_dtype,
    seq2seq_train_loop, BaselineRunner,
)


@dataclass
class B1NaiveFinetune:
    cfg:    INCAConfig
    name:   str = "b1_finetune"
    extras: Dict[str, Any] = field(default_factory=dict)

    _model:     Any = field(init=False, default=None)
    _optimizer: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device:    str = field(init=False, default="")

    def build_model(self, tokenizer, device: str):
        self._tokenizer = tokenizer
        self._device    = device
        load_kw = {"torch_dtype": model_dtype(device)}
        if "cuda" in device: load_kw["device_map"] = "auto"
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
        if "device_map" not in load_kw: model = model.to(device)
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if getattr(self.cfg, "gradient_checkpointing", False):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        params = [p for p in model.parameters() if p.requires_grad]
        if getattr(self.cfg, "use_adafactor", False):
            self._optimizer = Adafactor(params, lr=self.cfg.lr, relative_step=False,
                                        scale_parameter=False, warmup_init=False,
                                        weight_decay=self.cfg.weight_decay)
        else:
            self._optimizer = AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self._model = model
        return model

    def scoring_model(self): return self._model
    def on_period_start(self, period: Period): pass
    def on_period_end(self, period: Period): pass

    def train_period(self, period, scheduler=None, loss_logger=None, text_logger=None):
        cfg = self.cfg
        dl  = make_loader(period.train_items, self._tokenizer, batch_size=cfg.batch_size,
                          max_seq_len=cfg.max_input_length,
                          max_answer_len=getattr(cfg, "max_target_length", cfg.max_input_length))
        return seq2seq_train_loop(
            model=self._model, optimizer=self._optimizer, dataloader=dl,
            device=self._device, cfg=cfg, period_label=period.label,
            period_idx=period.index, n_periods=getattr(cfg, "max_periods", 99),
            scheduler=scheduler, loss_logger=loss_logger, text_logger=text_logger)


def main():
    p = argparse.ArgumentParser(description="B1 Naive fine-tune")
    p.add_argument("--config", required=True); p.add_argument("--dataset", default=None)
    p.add_argument("--seed", type=int, default=None); p.add_argument("--device", default=None)
    args = p.parse_args()
    with open(args.config) as f: cfg_dict = yaml.safe_load(f) or {}
    if args.dataset: cfg_dict["dataset"] = args.dataset
    if args.seed:    cfg_dict["seed"]    = args.seed
    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items() if k in INCAConfig.__dataclass_fields__})
    BaselineRunner(cfg, B1NaiveFinetune(cfg=cfg), device=args.device).run()

if __name__ == "__main__": main()

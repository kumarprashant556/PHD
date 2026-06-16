"""B5 — LoRA Mixture-of-Experts baseline (Trainer-based).

Adds a new LoRA adapter (expert) per period via PEFT, and learns a soft gate
over them.  A wrapper exposes ``forward`` that adds the gate regulariser to
``out.loss`` so plain ``Seq2SeqTrainer`` can train both peft adapters + gate.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoModelForSeq2SeqLM

# Repo root on sys.path so `baselines._runtime` resolves whether invoked as
# `python baselines/b5_lora_moe.py` (direct) or via the sweep launcher.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from baselines._runtime import (
    INCAConfig, TrainerRunner, TokenizedDataset,
    model_dtype, standard_trainer,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT = True
except ImportError:
    _PEFT = False


def _require_peft():
    if not _PEFT:
        raise ImportError("B5 requires peft: pip install peft")


class ExpertGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1))

    def add_expert(self):
        old = self.logits.data
        self.logits = nn.Parameter(torch.cat([old, torch.zeros(1, device=old.device)]))

    def weights(self):
        return F.softmax(self.logits, dim=0)


class LoRAGateWrapper(nn.Module):
    """PEFT model + ExpertGate.  Adds gate-regulariser to loss in forward."""

    def __init__(self, peft_model, gate: ExpertGate, gate_reg_weight: float):
        super().__init__()
        self.peft = peft_model
        self.gate = gate
        self.gate_reg_weight = gate_reg_weight

    def forward(self, **kwargs):
        out = self.peft(**kwargs)
        if kwargs.get("labels") is not None:
            w = self.gate.weights()
            # Push the latest expert's weight toward 1 (gives gate real grad).
            gate_reg = -torch.log(w[-1] + 1e-8)
            out.loss = out.loss + self.gate_reg_weight * gate_reg
        return out

    def generate(self, **kwargs):
        return self.peft.generate(**kwargs)

    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.peft, "gradient_checkpointing_enable"):
            self.peft.gradient_checkpointing_enable(**kw)

    def gradient_checkpointing_disable(self):
        if hasattr(self.peft, "gradient_checkpointing_disable"):
            self.peft.gradient_checkpointing_disable()


@dataclass
class B5LoRAMoE:
    cfg:             INCAConfig
    lora_rank:       int   = 8
    lora_alpha:      float = 16.0
    lora_dropout:    float = 0.05
    gate_reg_weight: float = 0.05
    name:            str   = "b5_lora_moe"

    _wrapper:      Any = field(init=False, default=None)
    _peft:         Any = field(init=False, default=None)
    _gate:         Any = field(init=False, default=None)
    _tokenizer:    Any = field(init=False, default=None)
    _device:       str = field(init=False, default="")
    _expert_names: List[str] = field(init=False, default_factory=list)

    def build_model(self, tokenizer, device: str):
        _require_peft()
        self._tokenizer = tokenizer
        self._device    = device
        load_kw = {"dtype": model_dtype(device, self.cfg)}
        if "cuda" in device:
            load_kw["device_map"] = "auto"
        base = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
        if "device_map" not in load_kw:
            base = base.to(device)
        if hasattr(base.config, "pad_token_id"):
            base.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        lora_cfg = LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout, bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        self._peft = get_peft_model(base, lora_cfg, adapter_name="expert_0")
        self._expert_names = ["expert_0"]
        self._gate = ExpertGate().to(device)
        self._wrapper = LoRAGateWrapper(self._peft, self._gate, self.gate_reg_weight).to(device)
        return self._wrapper

    def scoring_model(self):
        # If only one expert, return wrapper as-is.  With >1 experts, merge them
        # using the learned gate weights so generate() reflects the full mixture.
        if len(self._expert_names) > 1:
            w = self._gate.weights().detach().tolist()
            if "merged_eval" in self._peft.peft_config:
                self._peft.delete_adapter("merged_eval")
            self._peft.add_weighted_adapter(self._expert_names, w, "merged_eval", "linear")
            self._peft.set_adapter("merged_eval")
        return self._wrapper

    def on_period_start(self, period_label, period_idx):
        if period_idx == 0:
            return
        # Freeze old experts, add a new one, activate it.
        for n, p in self._peft.named_parameters():
            p.requires_grad = False
        new_name = f"expert_{period_idx}"
        lora_cfg = LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout, bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        self._peft.add_adapter(new_name, lora_cfg)
        self._peft.set_adapter(new_name)
        self._expert_names.append(new_name)
        # Grow the gate; re-attach so optimizer in next Trainer sees new param.
        self._gate.add_expert()
        self._gate = self._gate.to(self._device)
        self._wrapper.gate = self._gate
        if "merged_eval" in self._peft.peft_config:
            self._peft.delete_adapter("merged_eval")

    def on_period_end(self, period_label, period_idx, raw_items):
        if "merged_eval" in self._peft.peft_config:
            self._peft.delete_adapter("merged_eval")

    def make_trainer(self, args, raw_items, tokenizer, period_label, period_idx):
        # Make sure the latest expert is the active one before training.
        self._peft.set_adapter(self._expert_names[-1])
        max_in = self.cfg.max_input_length
        max_lb = getattr(self.cfg, "max_target_length", max_in)
        train_ds = TokenizedDataset(raw_items, tokenizer, max_in, max_lb)
        return standard_trainer(self._wrapper, args, train_ds, tokenizer)


def main() -> None:
    p = argparse.ArgumentParser(description="B5 LoRA-MoE baseline (Trainer-based)")
    p.add_argument("--config", required=True)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
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
        B5LoRAMoE(cfg=cfg, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha),
        device=args.device,
    ).run()


if __name__ == "__main__":
    main()

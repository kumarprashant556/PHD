"""B5 — LoRA Mixture-of-Experts. Powered by HuggingFace PEFT.

Adds a new LoRA adapter (expert) per period and learns a soft gate over them.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch, torch.nn as nn, torch.nn.functional as F, yaml
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM

from runner import (
    INCAConfig, Period, make_loader, model_dtype,
    make_epoch_bar, make_batch_bar, seq2seq_train_loop, BaselineRunner,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT = True
except ImportError:
    _PEFT = False


def _require_peft():
    if not _PEFT:
        raise ImportError("B5 requires peft: pip install peft --break-system-packages")


class ExpertGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1))

    def add_expert(self):
        old = self.logits.data
        self.logits = nn.Parameter(torch.cat([old, torch.zeros(1, device=old.device)]))

    def weights(self): return F.softmax(self.logits, dim=0)


@dataclass
class B5LoRAMoE:
    cfg:             INCAConfig
    lora_rank:       int   = 8
    lora_alpha:      float = 16.0
    lora_dropout:    float = 0.05
    gate_reg_weight: float = 0.05  # weight of gate regularization loss
    name:            str   = "b5_lora_moe"
    extras:          Dict[str, Any] = field(default_factory=dict)

    _model:        Any  = field(init=False, default=None)
    _gate:         Any  = field(init=False, default=None)
    _optimizer:    Any  = field(init=False, default=None)
    _tokenizer:    Any  = field(init=False, default=None)
    _device:       str  = field(init=False, default="")
    _expert_names: List = field(init=False, default_factory=list)

    def build_model(self, tokenizer, device: str):
        _require_peft()
        self._tokenizer = tokenizer
        self._device    = device
        load_kw = {"dtype": model_dtype(device)}
        if "cuda" in device: load_kw["device_map"] = "auto"
        base = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
        if "device_map" not in load_kw: base = base.to(device)
        if hasattr(base.config, "pad_token_id"):
            base.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        lora_cfg = LoraConfig(r=self.lora_rank, lora_alpha=self.lora_alpha,
                              lora_dropout=self.lora_dropout, bias="none",
                              task_type=TaskType.SEQ_2_SEQ_LM)
        self._model        = get_peft_model(base, lora_cfg, adapter_name="expert_0")
        self._expert_names = ["expert_0"]
        self._gate         = ExpertGate().to(device)
        self._setup_optimizer()
        self.extras.update(lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)
        return self._model

    def _setup_optimizer(self):
        active    = self._expert_names[-1]
        trainable = [p for n, p in self._model.named_parameters() if active in n and p.requires_grad]
        trainable += list(self._gate.parameters())
        self._optimizer = AdamW(trainable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def scoring_model(self):
        if len(self._expert_names) == 1: return self._model
        w = self._gate.weights().detach().tolist()
        if "merged_eval" in self._model.peft_config:
            self._model.delete_adapter("merged_eval")
        self._model.add_weighted_adapter(self._expert_names, w, "merged_eval", "linear")
        self._model.set_adapter("merged_eval")
        return self._model

    def on_period_start(self, period: Period):
        if period.index == 0: return
        for n, p in self._model.named_parameters(): p.requires_grad = False
        new_name = f"expert_{period.index}"
        lora_cfg = LoraConfig(r=self.lora_rank, lora_alpha=self.lora_alpha,
                              lora_dropout=self.lora_dropout, bias="none",
                              task_type=TaskType.SEQ_2_SEQ_LM)
        self._model.add_adapter(new_name, lora_cfg)
        self._model.set_adapter(new_name)
        self._expert_names.append(new_name)
        self._gate.add_expert(); self._gate = self._gate.to(self._device)
        self._setup_optimizer()
        if "merged_eval" in self._model.peft_config:
            self._model.delete_adapter("merged_eval")

    def train_period(self, period, scheduler=None, loss_logger=None, text_logger=None):
        cfg      = self.cfg
        max_seq  = cfg.max_input_length
        max_ans  = getattr(cfg, "max_target_length", max_seq)
        accum    = max(1, getattr(cfg, "grad_accum_steps", 1))
        log_every = max(1, getattr(cfg, "log_every_n_steps", 50))
        max_grad  = getattr(cfg, "max_grad_norm", 1.0)
        n_experts = len(self._expert_names)

        self._model.set_adapter(self._expert_names[-1])
        dl = make_loader(period.train_items, self._tokenizer, batch_size=cfg.batch_size,
                         max_seq_len=max_seq, max_answer_len=max_ans)
        if len(dl) == 0: return 0.0

        n_periods = getattr(cfg, "max_periods", 99)
        epoch_bar = make_epoch_bar(cfg.epochs_per_period, period.label, period.index, n_periods)
        last_loss = 0.0; opt_step = 0

        for epoch in epoch_bar:
            self._model.train(); total = n = 0; accum_loss = 0.0
            batch_bar = make_batch_bar(dl, epoch, cfg.epochs_per_period)
            for ms, batch in enumerate(batch_bar, 1):
                ids    = batch["input_ids"].to(self._device)
                mask   = batch.get("attention_mask")
                if mask is not None: mask = mask.to(self._device)
                labels = batch["labels"].to(self._device)
                out    = self._model(input_ids=ids, attention_mask=mask, labels=labels)
                if not torch.isfinite(out.loss): continue
                # Gate regularization: push latest expert weight toward 1 via
                # -log(w[-1]).  This gives gate parameters real gradients so the
                # gate learns to balance recency vs. history across periods.
                gate_w = self._gate.weights()
                gate_reg = -torch.log(gate_w[-1] + 1e-8)
                total_loss = out.loss + self.gate_reg_weight * gate_reg
                (total_loss / accum).backward()
                accum_loss += out.loss.item()
                if ms % accum == 0 or ms == len(dl):
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_grad)
                    self._optimizer.step(); self._optimizer.zero_grad(set_to_none=True)
                    if scheduler: scheduler.step()
                    opt_step += 1; sl = accum_loss / accum; accum_loss = 0.0
                    total += sl; n += 1
                    if loss_logger and opt_step % log_every == 0:
                        loss_logger(period.label, epoch, opt_step, sl)
            last_loss = total / max(n, 1)
            if loss_logger: loss_logger(period.label, epoch, opt_step, last_loss)
        return last_loss

    def on_period_end(self, period: Period):
        if "merged_eval" in self._model.peft_config:
            self._model.delete_adapter("merged_eval")


def main():
    p = argparse.ArgumentParser(description="B5 LoRA-MoE baseline")
    p.add_argument("--config", required=True); p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--dataset", default=None); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    with open(args.config) as f: cfg_dict = yaml.safe_load(f) or {}
    if args.dataset: cfg_dict["dataset"] = args.dataset
    if args.seed:    cfg_dict["seed"]    = args.seed
    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items() if k in INCAConfig.__dataclass_fields__})
    BaselineRunner(cfg, B5LoRAMoE(cfg=cfg, lora_rank=args.lora_rank,
                                  lora_alpha=args.lora_alpha), device=args.device).run()

if __name__ == "__main__": main()

"""B5 — LoRA Mixture-of-Experts. Powered by HuggingFace PEFT.

Supports seq2seq (FLAN-T5) and causal-LM via cfg.model_type.
For seq2seq, task_type is set to SEQ_2_SEQ_LM automatically.
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
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from Phase0.common.config import Phase0Config, load_config, model_dtype
from Phase0.common.harness import Period, make_loader
from Phase0.common.progress import make_epoch_bar, make_batch_bar
from Phase0.common.runner import BaselineRunner

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


def _require_peft():
    if not _PEFT_AVAILABLE:
        raise ImportError("B5 requires peft: pip install peft --break-system-packages")


class ExpertGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1))

    def add_expert(self):
        old = self.logits.data
        self.logits = nn.Parameter(torch.cat([old, torch.zeros(1, device=old.device)]))

    def weights(self):
        return F.softmax(self.logits, dim=0)


@dataclass
class B5LoRAMoE:
    cfg: Phase0Config
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    name: str = "b5_lora_moe"
    extras: Dict[str, Any] = field(default_factory=dict)

    _model: Any = field(init=False, default=None)
    _gate: ExpertGate = field(init=False, default=None)
    _optimizer: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device: str = field(init=False, default="")
    _expert_names: List[str] = field(init=False, default_factory=list)

    def build_model(self, tokenizer, device: str):
        _require_peft()
        self._tokenizer = tokenizer
        self._device    = device
        load_kw: Dict[str, Any] = {"torch_dtype": model_dtype(device)}
        if "cuda" in device:
            load_kw["device_map"] = "auto"

        if self.cfg.model_type == "seq2seq":
            base       = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
            task_type  = TaskType.SEQ_2_SEQ_LM
        else:
            base       = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kw)
            task_type  = TaskType.CAUSAL_LM

        if "device_map" not in load_kw:
            base = base.to(device)
        if hasattr(base.config, "pad_token_id"):
            base.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        lora_cfg = LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout, bias="none", task_type=task_type,
        )
        self._model        = get_peft_model(base, lora_cfg, adapter_name="expert_0")
        self._expert_names = ["expert_0"]
        self._gate         = ExpertGate().to(device)
        self._setup_optimizer()
        self.extras.update(lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)
        return self._model

    def _setup_optimizer(self):
        active    = self._expert_names[-1]
        trainable = [p for n, p in self._model.named_parameters()
                     if active in n and p.requires_grad]
        trainable += list(self._gate.parameters())
        self._optimizer = AdamW(trainable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def scoring_model(self):
        if len(self._expert_names) == 1:
            return self._model
        w = self._gate.weights().detach().tolist()
        if "merged_eval" in self._model.peft_config:
            self._model.delete_adapter("merged_eval")
        self._model.add_weighted_adapter(
            adapters=self._expert_names, weights=w,
            adapter_name="merged_eval", combination_type="linear",
        )
        self._model.set_adapter("merged_eval")
        return self._model

    def on_period_start(self, period: Period) -> None:
        if period.index == 0:
            return
        for n, p in self._model.named_parameters():
            p.requires_grad = False
        new_name = f"expert_{period.index}"
        task_type = (TaskType.SEQ_2_SEQ_LM if self.cfg.model_type == "seq2seq"
                     else TaskType.CAUSAL_LM)
        lora_cfg = LoraConfig(
            r=self.lora_rank, lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout, bias="none", task_type=task_type,
        )
        self._model.add_adapter(new_name, lora_cfg)
        self._model.set_adapter(new_name)
        self._expert_names.append(new_name)
        self._gate.add_expert()
        self._gate = self._gate.to(self._device)
        self._setup_optimizer()
        if "merged_eval" in self._model.peft_config:
            self._model.delete_adapter("merged_eval")

    def train_period(self, period: Period,
                     scheduler=None, loss_logger=None, text_logger=None) -> float:
        cfg       = self.cfg
        active    = self._expert_names[-1]
        n_experts = len(self._expert_names)
        self._model.set_adapter(active)

        dl = make_loader(
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
            self._model.train()
            total, n   = 0.0, 0
            accum_loss = 0.0
            batch_bar  = make_batch_bar(dl, epoch, n_epochs)

            for micro_step, batch in enumerate(batch_bar, 1):
                if cfg.model_type == "seq2seq":
                    ids    = batch["input_ids"].to(self._device)
                    mask   = batch.get("attention_mask")
                    if mask is not None: mask = mask.to(self._device)
                    labels = batch["labels"].to(self._device)
                    out    = self._model(input_ids=ids, attention_mask=mask, labels=labels)
                else:
                    ids    = batch["input_ids"].to(self._device)
                    mask   = batch.get("attention_mask")
                    if mask is not None: mask = mask.to(self._device)
                    labels = ids.clone()
                    if mask is not None: labels[mask == 0] = -100
                    out    = self._model(input_ids=ids, attention_mask=mask, labels=labels)

                if not torch.isfinite(out.loss):
                    continue
                scaled_loss = out.loss / accum_steps
                scaled_loss.backward()
                accum_loss += out.loss.item()

                is_accum_step = (
                    (micro_step % accum_steps == 0) or (micro_step == len(dl))
                )
                if is_accum_step:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), cfg.max_grad_norm,
                    )
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
                        experts=n_experts,
                        lr=f"{self._optimizer.param_groups[0]['lr']:.2e}",
                    )

            last_loss = total / max(n, 1)
            epoch_bar.set_postfix(avg_loss=f"{last_loss:.4f}", experts=n_experts)
            if loss_logger is not None:
                loss_logger(period.label, epoch, opt_step, last_loss)

        return last_loss

    def on_period_end(self, period: Period) -> None:
        if "merged_eval" in self._model.peft_config:
            self._model.delete_adapter("merged_eval")


def main() -> None:
    p = argparse.ArgumentParser(description="B5 LoRA-MoE baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--model_type", default=None)
    args, _ = p.parse_known_args()
    ov  = {k: getattr(args, k) for k in ("model_type",) if getattr(args, k) is not None}
    cfg = load_config(args.config, overrides=ov)
    BaselineRunner(cfg, B5LoRAMoE(cfg=cfg, lora_rank=args.lora_rank,
                                   lora_alpha=args.lora_alpha)).run()


if __name__ == "__main__":
    main()

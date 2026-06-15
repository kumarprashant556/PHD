"""B2 — Experience Replay baseline.

Supports both seq2seq (FLAN-T5) and causal-LM (pythia/GPT) via cfg.model_type.
The replay buffer stores QA probe items (seq2seq) or text docs (causal).
"""

from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
import argparse, random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from Phase0.common.config import Phase0Config, load_config, model_dtype
from Phase0.common.harness import Period, TextDataset, Seq2SeqDataset
from Phase0.common.progress import make_epoch_bar, make_batch_bar
from Phase0.common.runner import BaselineRunner


@dataclass
class B2ReplayOnly:
    cfg: Phase0Config
    buffer_size: int = 2000
    replay_ratio: float = 0.5
    name: str = "b2_replay"
    extras: Dict[str, Any] = field(default_factory=dict)

    _model: Any = field(init=False, default=None)
    _optimizer: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device: str = field(init=False, default="")
    _buffer: List[Dict[str, Any]] = field(init=False, default_factory=list)
    _rng: Any = field(init=False, default=None)

    def build_model(self, tokenizer, device: str):
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
        self._rng = random.Random(self.cfg.seed)
        self.extras.update(buffer_size=self.buffer_size, replay_ratio=self.replay_ratio)
        return model

    def scoring_model(self): return self._model
    def on_period_start(self, period: Period) -> None: return None

    def _make_ds(self, items):
        cfg = self.cfg
        if cfg.model_type == "seq2seq":
            return Seq2SeqDataset(items, self._tokenizer,
                                  max_input_len=cfg.max_seq_len,
                                  max_answer_len=cfg.max_answer_len)
        return TextDataset(items, self._tokenizer, cfg.max_seq_len)

    def _forward(self, batch):
        """Run forward pass and return CE loss for either model_type."""
        if self.cfg.model_type == "seq2seq":
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
        return out.loss

    def train_period(self, period: Period,
                     scheduler=None, loss_logger=None, text_logger=None) -> float:
        cfg    = self.cfg
        cur_bs = max(1, int(cfg.batch_size * (1 - self.replay_ratio)))
        rep_bs = cfg.batch_size - cur_bs

        ds = self._make_ds(list(period.train_items))
        dl = DataLoader(ds, batch_size=cur_bs, shuffle=True, drop_last=True)
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
                cur_loss = self._forward(batch)
                if not torch.isfinite(cur_loss):
                    continue
                losses = [cur_loss]

                if self._buffer and rep_bs > 0:
                    samples = self._rng.sample(self._buffer, min(rep_bs, len(self._buffer)))
                    rep_ds = self._make_ds(samples)
                    if len(rep_ds) > 0:
                        rep_items = [rep_ds[i] for i in range(len(rep_ds))]
                        rep_batch = {k: torch.stack([x[k] for x in rep_items])
                                     for k in rep_items[0]}
                        rep_loss = self._forward(rep_batch)
                        if torch.isfinite(rep_loss):
                            losses.append(rep_loss)

                raw_loss    = sum(losses) / len(losses)
                scaled_loss = raw_loss / accum_steps
                scaled_loss.backward()
                accum_loss += cur_loss.item()

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
                        cur=f"{step_loss:.4f}", avg=f"{total/n:.4f}",
                        buf=len(self._buffer),
                        lr=f"{self._optimizer.param_groups[0]['lr']:.2e}",
                    )

            last_loss = total / max(n, 1)
            epoch_bar.set_postfix(avg_loss=f"{last_loss:.4f}")
            if loss_logger is not None:
                loss_logger(period.label, epoch, opt_step, last_loss)

        return last_loss

    def on_period_end(self, period: Period) -> None:
        self._buffer.extend(period.train_items)
        if len(self._buffer) > self.buffer_size:
            self._rng.shuffle(self._buffer)
            self._buffer = self._buffer[:self.buffer_size]


def main() -> None:
    p = argparse.ArgumentParser(description="B2 Replay baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--buffer_size", type=int, default=2000)
    p.add_argument("--replay_ratio", type=float, default=0.5)
    p.add_argument("--model_type", default=None)
    args, _ = p.parse_known_args()
    ov = {k: getattr(args, k) for k in ("model_type",) if getattr(args, k) is not None}
    cfg = load_config(args.config, overrides=ov)
    BaselineRunner(cfg, B2ReplayOnly(cfg=cfg, buffer_size=args.buffer_size,
                                     replay_ratio=args.replay_ratio)).run()


if __name__ == "__main__":
    main()

"""B2 — Experience Replay baseline.

Maintains a buffer of past examples and mixes them into each period's training.
"""

from __future__ import annotations
import argparse, random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch, yaml
from torch.optim import AdamW

from runner import (
    INCAConfig, Period, make_loader, model_dtype, Seq2SeqDataset,
    make_epoch_bar, make_batch_bar, seq2seq_train_loop, BaselineRunner,
    _pre_tokenize, _make_collate,
)
from transformers import AutoModelForSeq2SeqLM


@dataclass
class B2ReplayOnly:
    cfg:          INCAConfig
    buffer_size:  int   = 2000
    replay_ratio: float = 0.5
    name:         str   = "b2_replay"
    extras:       Dict[str, Any] = field(default_factory=dict)

    _model:     Any  = field(init=False, default=None)
    _optimizer: Any  = field(init=False, default=None)
    _tokenizer: Any  = field(init=False, default=None)
    _device:    str  = field(init=False, default="")
    _buffer:    List = field(init=False, default_factory=list)
    _rng:       Any  = field(init=False, default=None)

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
        self._rng = random.Random(getattr(self.cfg, "seed", 42))
        self.extras.update(buffer_size=self.buffer_size, replay_ratio=self.replay_ratio)
        return model

    def scoring_model(self): return self._model
    def on_period_start(self, period: Period): pass

    def _forward(self, batch):
        ids    = batch["input_ids"].to(self._device)
        mask   = batch.get("attention_mask")
        if mask is not None: mask = mask.to(self._device)
        labels = batch["labels"].to(self._device)
        return self._model(input_ids=ids, attention_mask=mask, labels=labels).loss

    def train_period(self, period, scheduler=None, loss_logger=None, text_logger=None):
        cfg    = self.cfg
        cur_bs = max(1, int(cfg.batch_size * (1 - self.replay_ratio)))
        rep_bs = cfg.batch_size - cur_bs
        max_seq  = cfg.max_input_length
        max_ans  = getattr(cfg, "max_target_length", max_seq)
        accum    = max(1, getattr(cfg, "grad_accum_steps", 1))
        log_every = max(1, getattr(cfg, "log_every_n_steps", 50))
        max_grad  = getattr(cfg, "max_grad_norm", 1.0)

        # Pre-tokenize all items once (avoids per-batch tokenizer overhead)
        pad_id  = self._tokenizer.pad_token_id or 0
        collate = _make_collate(pad_id)
        encoded = _pre_tokenize(period.train_items, self._tokenizer, max_seq, max_ans)
        from torch.utils.data import DataLoader
        dl = DataLoader(Seq2SeqDataset(encoded), batch_size=cur_bs,
                        shuffle=True, drop_last=False, collate_fn=collate)
        if len(dl) == 0: return 0.0

        epoch_bar = make_epoch_bar(cfg.epochs_per_period, period.label,
                                   period.index, getattr(cfg, "max_periods", 99))
        last_loss = 0.0; opt_step = 0

        for epoch in epoch_bar:
            self._model.train()
            total = n = 0; accum_loss = 0.0
            batch_bar = make_batch_bar(dl, epoch, cfg.epochs_per_period)

            for ms, batch in enumerate(batch_bar, 1):
                cur_loss = self._forward(batch)
                if not torch.isfinite(cur_loss): continue
                losses = [cur_loss]

                n_cur = batch["input_ids"].shape[0]
                n_rep = 0
                if self._buffer and rep_bs > 0:
                    samples = self._rng.sample(self._buffer, min(rep_bs, len(self._buffer)))
                    rep_enc = _pre_tokenize(samples, self._tokenizer, max_seq, max_ans)
                    if rep_enc:
                        rb = collate(rep_enc)
                        rb = {k: v.to(self._device) for k, v in rb.items()}
                        rl = self._forward(rb)
                        if torch.isfinite(rl):
                            losses.append(rl)
                            n_rep = len(rep_enc)

                # Weight losses by actual batch sizes, not by number of loss terms.
                # Old code: sum/len gave replay 50% weight regardless of replay_ratio.
                total_n = n_cur + n_rep
                if n_rep > 0 and len(losses) == 2:
                    raw = (losses[0] * n_cur + losses[1] * n_rep) / total_n
                else:
                    raw = losses[0]
                (raw / accum).backward()
                accum_loss += cur_loss.item()

                if ms % accum == 0 or ms == len(dl):
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_grad)
                    self._optimizer.step()
                    self._optimizer.zero_grad(set_to_none=True)
                    if scheduler: scheduler.step()
                    opt_step += 1; sl = accum_loss / accum; accum_loss = 0.0
                    total += sl; n += 1
                    if loss_logger and opt_step % log_every == 0:
                        loss_logger(period.label, epoch, opt_step, sl)

            last_loss = total / max(n, 1)
            if loss_logger: loss_logger(period.label, epoch, opt_step, last_loss)

        return last_loss

    def on_period_end(self, period: Period):
        self._buffer.extend(period.train_items)
        if len(self._buffer) > self.buffer_size:
            self._rng.shuffle(self._buffer)
            self._buffer = self._buffer[:self.buffer_size]


def main():
    p = argparse.ArgumentParser(description="B2 Replay baseline")
    p.add_argument("--config", required=True); p.add_argument("--buffer_size", type=int, default=2000)
    p.add_argument("--replay_ratio", type=float, default=0.5)
    p.add_argument("--dataset", default=None); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    with open(args.config) as f: cfg_dict = yaml.safe_load(f) or {}
    if args.dataset: cfg_dict["dataset"] = args.dataset
    if args.seed:    cfg_dict["seed"]    = args.seed
    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items() if k in INCAConfig.__dataclass_fields__})
    BaselineRunner(cfg, B2ReplayOnly(cfg=cfg, buffer_size=args.buffer_size,
                                    replay_ratio=args.replay_ratio), device=args.device).run()

if __name__ == "__main__": main()

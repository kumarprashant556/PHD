"""B2 — Experience Replay baseline (Trainer-based).

Maintains a buffer of past period items.  Each period's training set is mixed
with replay items at the configured ``replay_ratio`` before being handed to
``Seq2SeqTrainer``.

Mixing the replay items into the train Dataset (rather than interleaving at
batch time) is the simpler design: Trainer's shuffling distributes them
uniformly over the epoch with the right ratio in expectation.
"""

from __future__ import annotations
import argparse
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml
from transformers import AutoModelForSeq2SeqLM

# Repo root on sys.path so `baselines._runtime` resolves whether invoked as
# `python baselines/b2_replay.py` (direct) or via the sweep launcher.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from baselines._runtime import (
    INCAConfig, TrainerRunner, TokenizedDataset,
    model_dtype, standard_trainer,
)


@dataclass
class B2Replay:
    cfg:          INCAConfig
    buffer_size:  int   = 2000
    replay_ratio: float = 0.5      # fraction of training set drawn from buffer
    name:         str   = "b2_replay"

    _model:     Any  = field(init=False, default=None)
    _tokenizer: Any  = field(init=False, default=None)
    _device:    str  = field(init=False, default="")
    _buffer:    List[Dict[str, Any]] = field(init=False, default_factory=list)
    _rng:       Any  = field(init=False, default=None)

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
        self._rng = random.Random(getattr(self.cfg, "seed", 42))
        return model

    def scoring_model(self):
        return self._model

    def on_period_start(self, period_label, period_idx):
        pass

    def make_trainer(self, args, raw_items, tokenizer, period_label, period_idx):
        # Mix replay items into the period's training set.
        # replay_ratio = fraction of the mixed set drawn from the buffer.
        train_items = list(raw_items)
        if self._buffer and self.replay_ratio > 0:
            n_replay = int(len(train_items) * self.replay_ratio / max(1e-9, 1 - self.replay_ratio))
            n_replay = min(n_replay, len(self._buffer))
            if n_replay > 0:
                train_items += self._rng.sample(self._buffer, n_replay)

        max_in = self.cfg.max_input_length
        max_lb = getattr(self.cfg, "max_target_length", max_in)
        train_ds = TokenizedDataset(train_items, tokenizer, max_in, max_lb)
        return standard_trainer(self._model, args, train_ds, tokenizer)

    def on_period_end(self, period_label, period_idx, raw_items):
        # Reservoir-style cap on buffer.
        self._buffer.extend(raw_items)
        if len(self._buffer) > self.buffer_size:
            self._rng.shuffle(self._buffer)
            self._buffer = self._buffer[: self.buffer_size]


def main() -> None:
    p = argparse.ArgumentParser(description="B2 Replay baseline (Trainer-based)")
    p.add_argument("--config", required=True)
    p.add_argument("--buffer_size", type=int, default=2000)
    p.add_argument("--replay_ratio", type=float, default=0.5)
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
        B2Replay(cfg=cfg, buffer_size=args.buffer_size, replay_ratio=args.replay_ratio),
        device=args.device,
    ).run()


if __name__ == "__main__":
    main()

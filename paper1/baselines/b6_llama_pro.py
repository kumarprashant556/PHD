"""B6 — LLaMA-Pro-style block expansion baseline (Trainer-based).

Appends a zero-initialised copy of the last encoder block at each new period
and freezes every previous block.  Plain Seq2SeqTrainer trains the (single)
new block per period.

Memory tracking: a MemoryTracker instance is attached at build_model() time
and records per-period peak_train_mb, infer_mb, param_delta, wall_time_s.
Call save_tracker(path) after TrainerRunner.run() to persist memory_log.json.
"""

from __future__ import annotations
import argparse
import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import yaml
from transformers import AutoModelForSeq2SeqLM

# Repo root on sys.path so `baselines._runtime` resolves whether invoked as
# `python baselines/b6_llama_pro.py` (direct) or via the sweep launcher.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from baselines._runtime import (
    INCAConfig, TrainerRunner, TokenizedDataset,
    model_dtype, standard_trainer,
)
from training.memory_tracker import MemoryTracker


def _get_layer_list(model):
    if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        return model.encoder.block
    if hasattr(model, "model") and hasattr(model.model, "encoder") \
            and hasattr(model.model.encoder, "block"):
        return model.model.encoder.block
    raise AttributeError(f"Cannot locate encoder block list in {type(model).__name__}")


def _zero_output_projections(block):
    """Zero ALL output projections so the new block is identity-preserving."""
    found = False
    for path in ("layer.0.SelfAttention.o", "layer.1.DenseReluDense.wo",
                 "self_attn.o_proj", "mlp.down_proj"):
        parts = path.split(".")
        try:
            m = block
            for part in parts:
                m = getattr(m, part)
            with torch.no_grad():
                if hasattr(m, "weight"):
                    m.weight.zero_()
                if hasattr(m, "bias") and m.bias is not None:
                    m.bias.zero_()
            found = True
        except AttributeError:
            continue
    if not found:
        warnings.warn(f"Could not zero-init output projections in {type(block).__name__}")


@dataclass
class B6LLaMAProExpansion:
    cfg:                      INCAConfig
    initial_trainable_blocks: int = 1
    name:                     str = "b6_llama_pro"

    _model:        Any          = field(init=False, default=None)
    _tokenizer:    Any          = field(init=False, default=None)
    _device:       str          = field(init=False, default="")
    _mem_tracker:  MemoryTracker = field(init=False, default=None)  # type: ignore[assignment]

    def build_model(self, tokenizer, device: str):
        self._tokenizer  = tokenizer
        self._device     = device
        self._mem_tracker = MemoryTracker(device=device, method=self.name)
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

    def on_period_start(self, period_label, period_idx):
        # Memory tracker: start timing + snap param count BEFORE any topology change
        if self._mem_tracker is not None and self._model is not None:
            self._mem_tracker.period_start(period_label, self._model)
        if period_idx == 0:
            # First period: freeze everything except the top N blocks.
            for p in self._model.parameters():
                p.requires_grad = False
            layers = _get_layer_list(self._model)
            for block in layers[-self.initial_trainable_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
        else:
            # Subsequent periods: copy + zero-init the last block, append,
            # freeze everything else.
            layers    = _get_layer_list(self._model)
            new_block = copy.deepcopy(layers[-1]).to(self._device)
            _zero_output_projections(new_block)
            for p in self._model.parameters():
                p.requires_grad = False
            layers.append(new_block)
            for p in layers[-1].parameters():
                p.requires_grad = True
            # Keep config depth in sync — update the main model config …
            for attr in ("num_layers", "num_hidden_layers", "n_layer"):
                if hasattr(self._model.config, attr):
                    setattr(self._model.config, attr,
                            getattr(self._model.config, attr) + 1)
                    break
            # … AND the encoder-stack config (T5 keeps a *separate* config
            # object on encoder/decoder; get_head_mask uses it to set the
            # size of the head_mask list — if it's stale the list is too
            # short and head_mask[i] crashes on the newly-added block).
            if hasattr(self._model, "encoder") and hasattr(self._model.encoder, "config"):
                enc_cfg = self._model.encoder.config
                for attr in ("num_layers", "num_hidden_layers", "n_layer"):
                    if hasattr(enc_cfg, attr):
                        setattr(enc_cfg, attr, getattr(enc_cfg, attr) + 1)
                        break

    def on_period_end(self, period_label, period_idx, raw_items):
        # Memory tracker: record peak memory + params after training
        # acc_delta is unavailable here (eval happens after this hook in runner.py);
        # the memory_log.json value will be 0.0 — enrich offline from regret_matrix.csv.
        if self._mem_tracker is not None and self._model is not None:
            self._mem_tracker.period_end(period_label, self._model, acc_delta=0.0)

    def save_tracker(self, path: "str | _Path") -> None:
        """Persist memory_log.json after TrainerRunner.run() completes."""
        if self._mem_tracker is not None:
            self._mem_tracker.save(path)

    def make_trainer(self, args, raw_items, tokenizer, period_label, period_idx):
        max_in = self.cfg.max_input_length
        max_lb = getattr(self.cfg, "max_target_length", max_in)
        train_ds = TokenizedDataset(raw_items, tokenizer, max_in, max_lb)
        return standard_trainer(self._model, args, train_ds, tokenizer)


def main() -> None:
    p = argparse.ArgumentParser(description="B6 LLaMA-Pro expansion (Trainer-based)")
    p.add_argument("--config", required=True)
    p.add_argument("--initial_trainable_blocks", type=int, default=1)
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
    expansion = B6LLaMAProExpansion(
        cfg=cfg, initial_trainable_blocks=args.initial_trainable_blocks
    )
    runner = TrainerRunner(cfg, expansion, device=args.device)
    runner.run()
    # Save memory log alongside the run results
    out_dir = _Path(getattr(cfg, "out_dir", "results"))
    expansion.save_tracker(out_dir / "memory_log.json")


if __name__ == "__main__":
    main()

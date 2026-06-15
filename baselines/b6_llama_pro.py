"""B6 — LLaMA-Pro-style block expansion. Adapted from TencentARC/LLaMA-Pro.

Appends a zero-initialised copy of the last encoder block at each new period.
Only the new block's parameters are trained; earlier blocks are frozen.
"""

from __future__ import annotations
import argparse, copy, warnings
from dataclasses import dataclass, field
from typing import Any, Dict

import torch, torch.nn as nn, yaml
from torch.optim import AdamW
from transformers import AutoModelForSeq2SeqLM

from runner import (
    INCAConfig, Period, make_loader, model_dtype,
    seq2seq_train_loop, BaselineRunner,
)


def _get_layer_list(model):
    """Return the encoder block list (T5 architecture)."""
    if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        return model.encoder.block
    if hasattr(model, "model") and hasattr(model.model, "encoder") \
            and hasattr(model.model.encoder, "block"):
        return model.model.encoder.block
    raise AttributeError(f"Cannot locate encoder block list in {type(model).__name__}")


def _zero_output_projections(block):
    """Zero-init ALL output projections so the new block is identity-preserving.

    T5 encoder blocks have TWO relevant projections:
      - Self-attention output  (layer.0.SelfAttention.o)
      - FFN output             (layer.1.DenseReluDense.wo)
    Both must be zeroed, or the block is not function-preserving.

    The original code returned after the first match, so FFN output was never
    zeroed in T5.  Fixed: remove early return so ALL paths are attempted.
    """
    found = False
    for path in ("layer.0.SelfAttention.o", "layer.1.DenseReluDense.wo",
                 "self_attn.o_proj", "mlp.down_proj"):
        parts = path.split(".")
        try:
            m = block
            for part in parts:
                m = getattr(m, part)
            with torch.no_grad():
                if hasattr(m, "weight"): m.weight.zero_()
                if hasattr(m, "bias") and m.bias is not None: m.bias.zero_()
            found = True
            # Do NOT return — keep looping to zero ALL matching projections.
        except AttributeError:
            continue
    if not found:
        warnings.warn(f"Could not zero-init output projections in {type(block).__name__}")


@dataclass
class B6LLaMAProExpansion:
    cfg:                       INCAConfig
    initial_trainable_blocks:  int = 1
    name:                      str = "b6_llama_pro"
    extras:                    Dict[str, Any] = field(default_factory=dict)

    _model:        Any  = field(init=False, default=None)
    _optimizer:    Any  = field(init=False, default=None)
    _tokenizer:    Any  = field(init=False, default=None)
    _device:       str  = field(init=False, default="")
    _first_period: bool = field(init=False, default=True)

    def build_model(self, tokenizer, device: str):
        self._tokenizer = tokenizer
        self._device    = device
        load_kw = {"dtype": model_dtype(device)}
        if "cuda" in device: load_kw["device_map"] = "auto"
        model = AutoModelForSeq2SeqLM.from_pretrained(self.cfg.model_name, **load_kw)
        if "device_map" not in load_kw: model = model.to(device)
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self._model = model; self._first_period = True
        return model

    def scoring_model(self): return self._model
    def on_period_end(self, period: Period): pass

    def on_period_start(self, period: Period):
        if self._first_period:
            for p in self._model.parameters(): p.requires_grad = False
            layers = _get_layer_list(self._model)
            for block in layers[-self.initial_trainable_blocks:]:
                for p in block.parameters(): p.requires_grad = True
            self._first_period = False
        else:
            layers    = _get_layer_list(self._model)
            new_block = copy.deepcopy(layers[-1]).to(self._device)
            _zero_output_projections(new_block)
            for p in self._model.parameters(): p.requires_grad = False
            layers.append(new_block)
            for p in layers[-1].parameters(): p.requires_grad = True
            # Bump config depth
            for attr in ("num_layers", "num_hidden_layers", "n_layer"):
                if hasattr(self._model.config, attr):
                    setattr(self._model.config, attr,
                            getattr(self._model.config, attr) + 1)
                    break

        trainable      = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = AdamW(trainable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def train_period(self, period, scheduler=None, loss_logger=None, text_logger=None):
        cfg = self.cfg
        dl  = make_loader(period.train_items, self._tokenizer, batch_size=cfg.batch_size,
                          max_seq_len=cfg.max_input_length,
                          max_answer_len=getattr(cfg, "max_target_length", cfg.max_input_length))
        return seq2seq_train_loop(
            model=self._model, optimizer=self._optimizer, dataloader=dl,
            device=self._device, cfg=cfg, period_label=period.label,
            period_idx=period.index, n_periods=getattr(cfg, "max_periods", 99),
            scheduler=scheduler, loss_logger=loss_logger)


def main():
    p = argparse.ArgumentParser(description="B6 LLaMA-Pro expansion")
    p.add_argument("--config", required=True)
    p.add_argument("--initial_trainable_blocks", type=int, default=1)
    p.add_argument("--dataset", default=None); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    with open(args.config) as f: cfg_dict = yaml.safe_load(f) or {}
    if args.dataset: cfg_dict["dataset"] = args.dataset
    if args.seed:    cfg_dict["seed"]    = args.seed
    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items() if k in INCAConfig.__dataclass_fields__})
    BaselineRunner(cfg, B6LLaMAProExpansion(cfg=cfg,
        initial_trainable_blocks=args.initial_trainable_blocks), device=args.device).run()

if __name__ == "__main__": main()

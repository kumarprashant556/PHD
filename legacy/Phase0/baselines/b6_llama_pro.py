"""B6 — LLaMA-Pro-style block expansion. Adapted from TencentARC/LLaMA-Pro.

Supports seq2seq (FLAN-T5 encoder or decoder blocks) and causal-LM.
For seq2seq the expansion applies to the last encoder transformer block.
"""

from __future__ import annotations
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
import argparse, copy, warnings
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from Phase0.common.config import Phase0Config, load_config, model_dtype
from Phase0.common.harness import Period, make_loader
from Phase0.common.progress import standard_train_loop, seq2seq_train_loop
from Phase0.common.runner import BaselineRunner


def _get_layer_list(model: nn.Module, model_type: str = "causal") -> nn.ModuleList:
    """Return the list of transformer blocks to expand."""
    # Seq2seq: expand encoder blocks (T5 architecture)
    if model_type == "seq2seq":
        if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
            return model.encoder.block
        if hasattr(model, "model") and hasattr(model.model, "encoder") \
                and hasattr(model.model.encoder, "block"):
            return model.model.encoder.block
    # Causal / fallback
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer"):
        t = model.transformer
        if hasattr(t, "h"):       return t.h
        if hasattr(t, "layers"):  return t.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder") \
            and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise AttributeError(f"Cannot locate block list in {type(model).__name__}")


def _zero_init_output_projections(block: nn.Module) -> None:
    _CANDIDATES = [
        ("self_attn.o_proj",       "mlp.down_proj"),
        ("attn.c_proj",            "mlp.c_proj"),
        ("self_attn.out_proj",     "fc2"),
        ("self_attention.dense",   "mlp.dense_4h_to_h"),
        ("attention.dense",        "mlp.dense_4h_to_h"),
        ("layer.0.SelfAttention.o", "layer.1.DenseReluDense.wo"),  # T5 block
    ]
    def _zero(mod):
        with torch.no_grad():
            for attr in ("weight", "bias"):
                t = getattr(mod, attr, None)
                if t is not None:
                    t.zero_()

    def _resolve(parent, path):
        m = parent
        for part in path.split("."): m = getattr(m, part)
        return m

    for ap, mp in _CANDIDATES:
        try:
            _zero(_resolve(block, ap))
            _zero(_resolve(block, mp))
            return
        except AttributeError:
            continue
    warnings.warn(f"Could not zero-init output projections in {type(block).__name__}")


@dataclass
class B6LLaMAProExpansion:
    cfg: Phase0Config
    initial_trainable_blocks: int = 1
    name: str = "b6_llama_pro"
    extras: Dict[str, Any] = field(default_factory=dict)

    _model: Any = field(init=False, default=None)
    _optimizer: Any = field(init=False, default=None)
    _tokenizer: Any = field(init=False, default=None)
    _device: str = field(init=False, default="")
    _first_period: bool = field(init=False, default=True)

    def build_model(self, tokenizer, device: str) -> torch.nn.Module:
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
        self._model       = model
        self._first_period = True
        return model

    def scoring_model(self) -> torch.nn.Module:
        return self._model

    def on_period_start(self, period: Period) -> None:
        mt = self.cfg.model_type
        if self._first_period:
            for p in self._model.parameters():
                p.requires_grad = False
            layers = _get_layer_list(self._model, mt)
            for block in layers[-self.initial_trainable_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True
            self._first_period = False
        else:
            layers    = _get_layer_list(self._model, mt)
            new_block = copy.deepcopy(layers[-1]).to(self._device)
            _zero_init_output_projections(new_block)
            for p in self._model.parameters():
                p.requires_grad = False
            layers.append(new_block)
            for p in layers[-1].parameters():
                p.requires_grad = True
            # Update config depth counter
            for attr in ("num_hidden_layers", "n_layer", "num_layers",
                         "num_layers", "num_heads"):
                if hasattr(self._model.config, attr):
                    setattr(self._model.config, attr,
                            getattr(self._model.config, attr) + 1)
                    break

        trainable      = [p for p in self._model.parameters() if p.requires_grad]
        self._optimizer = AdamW(trainable, lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)

    def train_period(self, period: Period,
                     scheduler=None, loss_logger=None, text_logger=None) -> float:
        cfg  = self.cfg
        dl   = make_loader(
            period.train_items, self._tokenizer,
            batch_size=cfg.batch_size, max_seq_len=cfg.max_seq_len,
            shuffle=True, model_type=cfg.model_type,
            max_answer_len=cfg.max_answer_len,
        )
        loop = seq2seq_train_loop if cfg.model_type == "seq2seq" else standard_train_loop
        return loop(
            model=self._model, optimizer=self._optimizer, dataloader=dl,
            device=self._device, cfg=cfg,
            period_label=period.label, period_idx=period.index,
            n_periods=cfg.max_periods,
            scheduler=scheduler, loss_logger=loss_logger,
        )

    def on_period_end(self, period: Period) -> None:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="B6 LLaMA-Pro block expansion")
    p.add_argument("--config", default=None)
    p.add_argument("--initial_trainable_blocks", type=int, default=1)
    p.add_argument("--model_type", default=None)
    args, _ = p.parse_known_args()
    ov  = {k: getattr(args, k) for k in ("model_type",) if getattr(args, k) is not None}
    cfg = load_config(args.config, overrides=ov)
    BaselineRunner(cfg, B6LLaMAProExpansion(
        cfg=cfg, initial_trainable_blocks=args.initial_trainable_blocks)).run()


if __name__ == "__main__":
    main()

"""Shared runtime for the B1-B7 baselines.

The public API surface used by every ``baselines/bN_*.py`` file:

    INCAConfig            ← re-exported from models.inca.config
    TrainerRunner         ← orchestrates the per-period loop
    Baseline (Protocol)   ← interface every baseline implements
    TokenizedDataset      ← pre-tokenized HF dataset
    standard_trainer      ← Seq2SeqTrainer factory used by B1/B4/B5/B6/B7
    model_dtype           ← pick fp32/bf16/fp16 by config + device
    _trainer_kwargs       ← compat shim for tokenizer= vs processing_class=

Sub-modules are split by concern; each is small enough to read in one screen:

    precision.py        — model_dtype, autocast_dtype
    data.py             — CC-News v2 loaders, TokenizedDataset
    eval.py             — eval_cloze_accuracy, pretty_matrix
    logging_setup.py    — setup_logging, TrainerLogCallback
    trainer_factory.py  — standard_trainer, build_training_args, _trainer_kwargs
    runner.py           — Baseline protocol, TrainerRunner
"""
from __future__ import annotations

# ── Repo root on sys.path so submodules can `from models.inca.config import ...`
# and `from evaluation.metrics import ...` regardless of how they're invoked.
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Re-export the public API.
from models.inca.config import INCAConfig                                  # noqa: E402

from .precision        import model_dtype, autocast_dtype                  # noqa: E402
from .data             import TokenizedDataset, load_cc_news_v2, load_probes  # noqa: E402
from .eval             import eval_cloze_accuracy, pretty_matrix           # noqa: E402
from .logging_setup    import LOGGER_NAME, setup_logging, TrainerLogCallback  # noqa: E402
from .trainer_factory  import standard_trainer, build_training_args, _trainer_kwargs  # noqa: E402
from .runner           import Baseline, TrainerRunner                      # noqa: E402

# Back-compat alias used by some baselines.
BaselineRunner = TrainerRunner

# Underscore-prefixed back-compat aliases for legacy importers.
_load_cc_news_v2 = load_cc_news_v2
_load_probes     = load_probes
_autocast_dtype  = autocast_dtype
_pretty_matrix   = pretty_matrix

__all__ = [
    "INCAConfig",
    "Baseline", "TrainerRunner", "BaselineRunner",
    "TokenizedDataset",
    "standard_trainer", "build_training_args",
    "eval_cloze_accuracy", "pretty_matrix",
    "model_dtype", "autocast_dtype",
    "setup_logging", "TrainerLogCallback", "LOGGER_NAME",
    "load_cc_news_v2", "load_probes",
    "_trainer_kwargs",
]

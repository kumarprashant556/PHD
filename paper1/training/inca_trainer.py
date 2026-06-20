"""INCA-v2 training loop.

Orchestrates:
  * INCALayerManager          — block-chain growth / freeze
  * INCAPlateauDetector       — multi-signal consensus saturation (T1.1)
  * CKAMonitor                — representational-stability signal (T1.5)
  * INCAReplayBuffer          — study-schedule replay (T1.4)
  * T1.2 early-stop relabelling — timeout → PERIOD_LEARNED or BLOCK_FULL
  * T1.3 replay drift check   — early BLOCK_FULL if past-period accuracy drops

Usage (from repo root)
----------------------
    python scripts/train_inca.py --config configs/inca.yaml
    python scripts/train_inca.py --config configs/inca.yaml --dataset cc_news
    python scripts/train_inca.py --config configs/inca.yaml --dry-run
"""

from __future__ import annotations

# ── Crash hardening: must run before ANY other import ─────────────────────────
#
# Root-cause chain that has been observed:
#   1. A prior job crashes with SIGSEGV while Python is writing a .pyc file.
#      The file is left on disk in a partially-written, corrupt state.
#   2. The next job starts.  Python reads the corrupt .pyc, detects a checksum
#      mismatch, and falls back to compiling the source with compile().
#   3. compile() stack-overflows inside CPython's C code while processing numpy
#      or pandas source files that have deeply-nested type annotations.
#      The default Linux stack limit (8 MB) is too small for Python 3.12's
#      new bytecode compiler on these files. SIGSEGV.
#
# Fixes applied here (in order, before any third-party import):
#   A. faulthandler: print full Python+C traceback on SIGSEGV instead of
#      just "Segmentation fault" — allows future crash diagnosis.
#   B. resource.setrlimit(RLIMIT_STACK): raise the C stack limit to 256 MB
#      in this process, so compile() no longer overflows.
#   C. Pre-compile pandas + numpy in a subprocess that runs with
#      `ulimit -s unlimited` (bash shell limit, inherited before Python
#      starts).  This regenerates all .pyc files so compile() is NEVER
#      called inside our process for these packages.
#   D. CUDA_MODULE_LOADING=LAZY: defer CUDA kernel loading to first use.
#      Prevents CUDA-driver SIGSEGV when zombie processes from prior crashes
#      still hold /dev/nvidia* file descriptors.
#   E. TOKENIZERS_PARALLELISM=false + RAYON_NUM_THREADS=1: keep the Rust
#      tokenizer single-threaded. Rayon worker panics kill the process with
#      uncatchable SIGSEGV.

import faulthandler as _faulthandler
import os as _os
import resource as _resource
import subprocess as _subprocess
import sys as _sys
import sysconfig as _sysconfig

# A. Enable fault handler
_faulthandler.enable(file=_sys.stderr, all_threads=True)

# B. Raise stack limit to 256 MB in this process
try:
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_STACK)
    _target = 256 * 1024 * 1024          # 256 MB
    _new_soft = (_resource.RLIM_INFINITY
                 if _hard == _resource.RLIM_INFINITY or _hard >= _target
                 else _hard)
    if _soft != _resource.RLIM_INFINITY and _soft < _target:
        _resource.setrlimit(_resource.RLIMIT_STACK, (_new_soft, _hard))
        print(f"[startup] stack limit raised: {_soft // 1024} KB → {_new_soft if _new_soft == _resource.RLIM_INFINITY else str(_new_soft // 1024) + ' KB'}", flush=True)
    else:
        print(f"[startup] stack limit already sufficient: {_soft}", flush=True)
except Exception as _e:
    print(f"[startup] could not raise stack limit: {_e}", flush=True)

# C. Pre-compile pandas + numpy .pyc in a subprocess with unlimited stack.
#    After this, our process loads .pyc files directly; compile() is never
#    called for these packages, regardless of this process's stack limit.
_sp = _sysconfig.get_path("purelib")  # e.g. .../site-packages
if _sp:
    print(f"[startup] pre-compiling numpy + pandas in {_sp} …", flush=True)
    _rc = _subprocess.run(
        ["bash", "-c",
         f"ulimit -s unlimited && "
         f'"{_sys.executable}" -W ignore -m compileall -q -l '
         f'"{_sp}/numpy" "{_sp}/pandas" 2>/dev/null'],
        timeout=180,
        capture_output=True,
    )
    print(f"[startup] pre-compile done  rc={_rc.returncode}", flush=True)
    if _rc.stderr:
        print(f"[startup] pre-compile stderr: {_rc.stderr.decode(errors='replace')[:400]}", flush=True)
else:
    print("[startup] WARNING: could not find site-packages; skipping pre-compile", flush=True)

# D + E. Environment variables
_os.environ.setdefault("CUDA_MODULE_LOADING",    "LAZY")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_os.environ.setdefault("RAYON_NUM_THREADS",      "1")

print(f"[startup] Python {_sys.version}", flush=True)
print(f"[startup] pid={_os.getpid()}", flush=True)
print("[startup] importing torch …", flush=True)
del _faulthandler, _resource, _subprocess, _sysconfig
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import contextlib
import csv
import dataclasses
import gc
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
print(f"[startup] torch {torch.__version__}  CUDA={torch.version.cuda}", flush=True)
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers import DataCollatorForSeq2Seq
from transformers.modeling_outputs import BaseModelOutput
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

# ── repo-root on sys.path (for `python scripts/train_inca.py`) ─────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.inca.config  import INCAConfig
from models.inca.layer_manager import INCALayerManager
from models.inca.plateau import INCAPlateauDetector, SaturationEvent
from models.inca.cka     import CKAMonitor
from models.inca.replay  import INCAReplayBuffer

import data as data_module   # data/__init__.py  (load_periods, tokenize, …)
from data.tokenizer import build_tokenized_periods, make_dataloader, make_replay_dataloader, compute_percentile_length
from training.memory_tracker import MemoryTracker


def _patch_gcl() -> None:
    """Fix GradientCheckpointingLayer.__call__ to not crash on Python 3.12.

    In newer transformers, T5Block modules are wrapped with GradientCheckpointingLayer
    at model-load time (not only when gradient_checkpointing_enable() is called).
    Its __call__ does:
        return super().__call__(*args, **kwargs)
    On Python 3.12, super() in a magic method override that sits above a C-extension
    base class (nn.Module.__call__ is _wrapped_call_impl in C) corrupts the MRO
    lookup and ends up calling __getattr__ on self._modules with a dict as the key,
    giving 'TypeError: unhashable type: dict'.
    gradient_checkpointing_disable() only sets a flag inside the wrapper; it does NOT
    remove the wrapper. We patch __call__ once to use an explicit nn.Module.__call__
    dispatch that bypasses the super() resolution entirely.
    """
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer as _GCL
        _base = nn.Module.__call__
        _GCL.__call__ = lambda self, *a, **kw: _base(self, *a, **kw)
    except (ImportError, AttributeError):
        pass


_patch_gcl()


def _patch_module_train() -> None:
    """Fix nn.Module.train to not use _modules.items() which crashes Python 3.12.

    nn.Module.train() calls children() → named_children() → self._modules.items().
    On Python 3.12, iterating _modules.items() on certain module types (e.g. modules
    that were wrapped by GradientCheckpointingLayer at load time) returns wrong objects
    — e.g. a bare NewGELUActivation instead of a (name, module) tuple — causing:
        TypeError: 'NewGELUActivation' object is not iterable
    at the 'for name, module in self._modules.items()' line.

    Fix: replace nn.Module.train with an implementation that iterates _modules KEYS
    only (via list(dict) which uses __iter__, not .items()) and then fetches values
    via dict.get().  This avoids the broken items() iterator entirely.
    """
    def _safe_train(self, mode: bool = True):
        self.__dict__['training'] = mode
        mods = self.__dict__.get('_modules', {})
        for k in list(mods):              # iterate KEYS — safe on Python 3.12
            child = mods.get(k)
            if child is not None and isinstance(child, nn.Module):
                child.train(mode)         # recursive; uses this patched version
        return self

    nn.Module.train = _safe_train


_patch_module_train()


# ──────────────────────────────────────────────────────────────────────────────
# Simple run logger (replaces Phase0 RunLogger)
# ──────────────────────────────────────────────────────────────────────────────

class _RunLogger:
    """Write timestamped log lines to stdout and to a JSON-Lines file."""

    def __init__(self, out_dir: Path, cfg_snapshot: dict) -> None:
        self.out_dir = out_dir
        self._log_path = out_dir / "run_log.jsonl"
        self._append({"event": "config", "cfg": cfg_snapshot})

    def log(self, msg: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"[{ts}] {msg}", flush=True)
        self._append({"event": "log", "msg": msg, "ts": ts})

    def grow(
        self,
        period: str,
        block_from: int,
        block_to: int,
        opt_step: int,
        score: float,
        rir: float,
        cka: float,
        grad_norm_ema: float,
        trigger: str = "BLOCK_FULL",
    ) -> None:
        """Log a freeze-and-grow event.  Provides the EXP_T data for the paper figure."""
        ts = datetime.now().isoformat(timespec="seconds")
        self._append({
            "event":         "grow",
            "period":        period,
            "block_from":    block_from,
            "block_to":      block_to,
            "opt_step":      opt_step,   # EXP_T — step at which grow fires
            "score":         round(score, 6),
            "rir":           round(rir, 6),
            "cka":           round(cka, 6),
            "grad_norm_ema": round(grad_norm_ema, 6),
            "trigger":       trigger,
            "ts":            ts,
        })
        print(
            f"[{ts}] ↑ GROW  {period}  block {block_from}→{block_to}"
            f"  opt_step={opt_step}  rir={rir:.3f}  cka={cka:.3f}",
            flush=True,
        )

    def _append(self, record: dict) -> None:
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Loss-curve CSV writer
# ──────────────────────────────────────────────────────────────────────────────

class _LossLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["period", "block", "epoch", "opt_step", "loss", "timestamp"]
            )

    def log(
        self,
        period: str,
        block: int,
        epoch: int,
        step: int,
        loss: float,
    ) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [period, block, epoch, step, f"{loss:.6f}",
                 datetime.now().isoformat(timespec="seconds")]
            )


class _SignalLog:
    """Write per-k_eval saturation-signal trajectories to CSV.

    This is the raw data behind three paper figures:
      • CKA / RIR / grad_norm trajectory plot (validates detector independence)
      • EXP_T distribution histogram (when grow fires across runs)
      • E-TIMING comparison (score vs step under different timing conditions)

    Columns
    -------
    period, block, epoch, opt_step : position in training
    rir      : Relative Improvement Rate since period start
    score    : token-F1 evaluation score at this step
    cka      : CKA between current and reference representations
    gnorm_ema: exponential-moving-average gradient L2 norm
    avg_loss : mean training loss over the last k_eval steps
    event    : NONE / PERIOD_LEARNED / BLOCK_FULL / EXHAUSTED
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "period", "block", "epoch", "opt_step",
                "rir", "score", "cka", "gnorm_ema", "avg_loss", "event",
            ])

    def log(
        self,
        period: str,
        block: int,
        epoch: int,
        opt_step: int,
        rir: float,
        score: float,
        cka: float,
        gnorm_ema: float,
        avg_loss: float,
        event: str,
    ) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                period, block, epoch, opt_step,
                f"{rir:.6f}", f"{score:.6f}", f"{cka:.6f}",
                f"{gnorm_ema:.6f}", f"{avg_loss:.6f}", event,
            ])


# ──────────────────────────────────────────────────────────────────────────────
# Batch utilities
# ──────────────────────────────────────────────────────────────────────────────

def _batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _empty_cache(device: str) -> None:
    """Release PyTorch's cached (but unused) memory back to the device allocator.

    After a grow event the old optimiser's Adam states (~9 GB for FLAN-T5-large)
    are freed to Python GC but stay in the MPS/CUDA allocator cache.  Calling
    this before post-grow eval or replay scoring prevents OOM on those passes.
    """
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def _forward_loss(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run INCA manager encoder → T5 decoder → cross-entropy loss."""
    enc_hidden = manager(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    enc_out = BaseModelOutput(last_hidden_state=enc_hidden)
    out = model(
        encoder_outputs=enc_out,
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return out.loss


# ──────────────────────────────────────────────────────────────────────────────
# Token-level F1 helper (SQuAD-style) — used for saturation scoring
# ──────────────────────────────────────────────────────────────────────────────

import re as _re
import string as _string
from collections import Counter as _Counter
from typing import NamedTuple


class EvalResult(NamedTuple):
    """Dual eval metrics returned by _eval_accuracy.

    token_f1    — BoW token F1 over the full generated text (smooth,
                  used as the primary saturation signal / RIR numerator).
    exact_match — exact string match on the extracted final answer
                  (1.0 / 0.0; more interpretable for paper tables and for
                  diagnosing whether the model actually gets answers right).
    """
    token_f1: float
    exact_match: float


def _token_f1(pred: str, gold: str) -> float:
    """Bag-of-words token F1, identical to the SQuAD evaluation script.

    Lower-cases and strips punctuation before splitting on whitespace.
    Returns a float in [0, 1].  Full credit (1.0) when prediction equals
    reference; partial credit for partial token overlap.

    Used instead of exact-match for the INCA saturation signal so that the
    score is non-zero even when the model produces a paraphrase rather than
    the verbatim target — giving the RIR tracker a meaningful gradient to
    act on.
    """
    def _toks(s: str):
        return s.lower().translate(
            str.maketrans("", "", _string.punctuation)
        ).split()

    p_toks = _toks(pred)
    g_toks = _toks(gold)
    if not p_toks or not g_toks:
        return 0.0
    common = sum((_Counter(p_toks) & _Counter(g_toks)).values())
    if common == 0:
        return 0.0
    precision = common / len(p_toks)
    recall    = common / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def _extract_final_answer(text: str) -> str:
    """Extract the final answer token from generated or reference text.

    Priority order:
    1. "#### X"  — GSM8K / MetaMathQA end-marker
    2. "The answer is X" — MetaMath chain-of-thought closing phrase
    3. Last non-whitespace token — works for single-word targets (SciQ,
       medmcqa option text) when neither pattern is present
    """
    _punct = str.maketrans("", "", _string.punctuation)

    m = _re.search(r'####\s*([^\n]+)', text)
    if m:
        return m.group(1).strip().lower().translate(_punct)

    m = _re.search(r'[Tt]he answer is\s+([^\.\n]+)', text)
    if m:
        return m.group(1).strip().lower().translate(_punct)

    words = text.strip().split()
    return words[-1].lower().translate(_punct) if words else ""


def _answer_exact_match(pred: str, gold: str) -> float:
    """Exact match on extracted final answers (1.0 or 0.0).

    Extracts the final answer from both strings before comparing.
    Returns 0.0 if extraction fails for either side (e.g. code targets
    where no answer marker exists).
    """
    p = _extract_final_answer(pred)
    g = _extract_final_answer(gold)
    if not p or not g:
        return 0.0
    return 1.0 if p == g else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation — greedy decode + token-F1 over raw (un-tokenized) Dataset
# ──────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _eval_ctx(model: "T5ForConditionalGeneration"):
    """Set model to eval mode and restore training mode on exit.

    _patch_gcl() and _patch_module_train() applied at import time make
    model.eval() / model.train() safe on Python 3.12.  This context manager
    only needs to handle gradient-checkpointing bookkeeping.

    Note: _greedy_decode always passes use_cache=False explicitly to avoid
    the T5Attention flash-attention SIGSEGV that occurs with KV-cache growth.
    """
    _gc_on = getattr(model, "is_gradient_checkpointing", False)
    if _gc_on:
        model.gradient_checkpointing_disable()
    model.eval()
    try:
        yield
    finally:
        model.train()
        if _gc_on:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()


@torch.no_grad()
def _greedy_decode(
    model:     "T5ForConditionalGeneration",
    enc_hidden: torch.Tensor,
    attn_mask:  torch.Tensor,
    max_new_tokens: int,
    device: str,
) -> torch.Tensor:
    """Greedy T5 decode — no KV cache, full sequence re-fed each step.

    We intentionally do NOT use use_cache=True / past_key_values.  The
    incremental KV-cache path in T5Attention triggers a CUDA kernel
    SIGSEGV (invalid memory access in flash-attention) once the cache
    grows large enough — observable reliably after ~100 training steps
    even though the first two evals (steps 50, 100) complete successfully.

    Without KV cache:
      - Each step feeds the full generated sequence to the decoder.
      - out[0] = lm_logits (batch, current_len, vocab); we take [:, -1, :].
      - past_key_values is None and absent from the output tuple.
      - O(n²) computation, but negligible vs. training time for n ≤ 512.
    """
    batch     = enc_hidden.shape[0]
    dec_start = model.config.decoder_start_token_id
    eos_id    = model.config.eos_token_id or 1

    generated = torch.full((batch, 1), dec_start, dtype=torch.long, device=device)
    done      = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        out = model(
            encoder_outputs=(enc_hidden,),   # (last_hidden_state,) — skips encoder
            attention_mask=attn_mask,        # encoder attention mask for cross-attn
            decoder_input_ids=generated,     # full sequence so far; grows by 1/step
            use_cache=False,                 # no KV cache — avoids flash-attn SIGSEGV
            return_dict=False,               # plain tuple, no ModelOutput created
        )
        # return_dict=False, use_cache=False, no labels:
        #   out[0] = lm_logits  (batch, current_len, vocab)
        logits   = out[0][:, -1, :]              # last-position logits (batch, vocab)
        next_tok = logits.argmax(-1, keepdim=True)  # (batch, 1)
        generated = torch.cat([generated, next_tok], dim=1)
        done     |= next_tok.squeeze(-1) == eos_id
        if done.all():
            break

    return generated


@torch.no_grad()
def _eval_accuracy(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    eval_ds: Dataset,
    tokenizer,
    device: str,
    batch_size: int = 32,
    max_input_length: int = 256,
    max_new_tokens: int = 256,
    n_samples: int = 500,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    amp_dev: str = "cpu",
) -> EvalResult:
    """Sample *n_samples* rows from *eval_ds* and return EvalResult(token_f1, exact_match).

    *eval_ds* must have columns: input_text (str), target_text (str).

    token_f1    — BoW F1 over full generated text; smooth signal for RIR.
    exact_match — exact match on extracted final answer (handles
                  "#### X" and "The answer is X" patterns); 0.0 for
                  code targets where no answer marker exists.
    """
    if len(eval_ds) == 0:
        return EvalResult(0.0, 0.0)

    indices = list(range(len(eval_ds)))
    if len(indices) > n_samples:
        indices = random.sample(indices, n_samples)
    subset = eval_ds.select(indices)

    total_f1 = 0.0
    total_em = 0.0
    total    = 0

    with _eval_ctx(model):
        manager.eval()
        for start in range(0, len(subset), batch_size):
            chunk  = subset.select(range(start, min(start + batch_size, len(subset))))
            inputs = tokenizer(
                list(chunk["input_text"]),
                truncation=True,
                max_length=max_input_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = _batch_to_device(inputs, device)

            with torch.autocast(device_type=amp_dev, dtype=amp_dtype, enabled=amp_enabled):
                enc_hidden = manager(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                gen_ids = _greedy_decode(
                    model, enc_hidden, inputs["attention_mask"], max_new_tokens, device
                )
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            del gen_ids, enc_hidden
            gc.collect()
            _empty_cache(device)
            golds = chunk["target_text"]

            for pred, gold in zip(preds, golds):
                total    += 1
                total_f1 += _token_f1(pred, str(gold))
                total_em += _answer_exact_match(pred, str(gold))

    manager.train()
    n = max(1, total)
    return EvalResult(token_f1=total_f1 / n, exact_match=total_em / n)


# ──────────────────────────────────────────────────────────────────────────────
# CKA reference helper
# ──────────────────────────────────────────────────────────────────────────────

def _cache_cka_reference(
    monitor: CKAMonitor,
    manager: INCALayerManager,
    raw_ds: Dataset,
    tokenizer,
    device: str,
    max_seq_len: int = 256,
    n_samples: int = 200,
) -> None:
    """Cache CKA reference set from raw Dataset rows.

    CKAMonitor._items_to_texts now reads ``input_text`` / ``text`` / ``question``
    natively, so we can hand it raw rows without wrapping.
    """
    indices = list(range(len(raw_ds)))
    if len(indices) > n_samples:
        indices = random.sample(indices, n_samples)
    subset = raw_ds.select(indices)
    items = [dict(row) for row in subset]
    monitor.cache_reference(manager, items, tokenizer, device, max_seq_len=max_seq_len)


# ──────────────────────────────────────────────────────────────────────────────
# T1.3 Replay-drift check
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize_replay_items(
    raw_items: List[dict],
    tokenizer,
    max_input_length: int = 256,
    max_target_length: int = 256,
) -> List[dict]:
    """Tokenize raw replay dicts (input_text/target_text) into variable-length lists.

    Returns per-item dicts with Python lists (no padding, no tensors) so that
    DataCollatorForSeq2Seq can pad them together with stream items at collation time.
    """
    if not raw_items:
        return []
    enc = tokenizer(
        [it["input_text"] for it in raw_items],
        truncation=True, max_length=max_input_length,
        padding=False,
    )
    dec = tokenizer(
        text_target=[it["target_text"] for it in raw_items],
        truncation=True, max_length=max_target_length,
        padding=False,
    )
    result = []
    for i in range(len(raw_items)):
        result.append({
            "input_ids":      enc["input_ids"][i],       # Python list (variable length)
            "attention_mask": enc["attention_mask"][i],  # Python list
            "labels":         dec["input_ids"][i],       # Python list; -100 applied by collator
        })
    return result


@torch.no_grad()
def _per_item_losses(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    items: List[dict],
    tokenizer,
    device: str,
    max_input_length: int,
    max_target_length: int,
    batch_size: int,
) -> List[float]:
    """Compute per-item cross-entropy loss (no reduction) for replay scoring.

    Returns one scalar per item, in the same order as ``items``.  Used to
    refresh INCAReplayBuffer entries so Phase B's hard/easy/mid study
    schedule has real loss values to sort on.
    """
    if not items:
        return []
    model.eval()
    manager.eval()
    pad_id = tokenizer.pad_token_id
    losses: List[float] = []
    ce_none = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    try:
        for start in range(0, len(items), batch_size):
            chunk = items[start: start + batch_size]
            enc = tokenizer(
                [it["input_text"] for it in chunk],
                truncation=True, max_length=max_input_length,
                padding=True, return_tensors="pt",
            ).to(device)
            dec = tokenizer(
                text_target=[it["target_text"] for it in chunk],
                truncation=True, max_length=max_target_length,
                padding=True, return_tensors="pt",
            )
            labels = dec["input_ids"].to(device)
            labels_masked = labels.clone()
            labels_masked[labels_masked == pad_id] = -100

            enc_hidden = manager(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            enc_out = BaseModelOutput(last_hidden_state=enc_hidden)
            out = model(
                encoder_outputs=enc_out,
                attention_mask=enc["attention_mask"],
                labels=labels_masked,
            )
            logits = out.logits           # (B, T, V)
            B, T, V = logits.shape
            flat_loss = ce_none(logits.reshape(B * T, V), labels_masked.reshape(B * T))
            flat_loss = flat_loss.view(B, T)
            valid = (labels_masked != -100).float()
            per_item = (flat_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            losses.extend(per_item.detach().cpu().tolist())
            # Periodically release MPS allocator cache to prevent accumulation over
            # up to 2 000 replay items (500 batches at batch_size=4).
            del out, logits, flat_loss, valid, per_item, enc_out, enc_hidden
            if (start // batch_size) % 50 == 49:   # every 200 items
                gc.collect()
                _empty_cache(device)
    finally:
        model.train()
        manager.train()
    return losses


def _check_replay_drift(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    replay_buf: INCAReplayBuffer,
    prev_acc: float,
    tokenizer,
    device: str,
    tol: float,
    batch_size: int = 32,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    amp_dev: str = "cpu",
) -> bool:
    """Return True if replay accuracy has dropped by more than *tol*."""
    all_items = replay_buf.all_items()
    if not all_items:
        return False

    replay_ds = Dataset.from_list(all_items)
    cur_acc = _eval_accuracy(
        model, manager, replay_ds, tokenizer, device,
        batch_size=batch_size,
        n_samples=len(all_items),
        amp_enabled=amp_enabled, amp_dtype=amp_dtype, amp_dev=amp_dev,
    )
    return (prev_acc - cur_acc.token_f1) > tol


# ──────────────────────────────────────────────────────────────────────────────
# Grow helper — freeze + grow + new optimiser + new scheduler
# ──────────────────────────────────────────────────────────────────────────────

def _grow_block(
    manager: INCALayerManager,
    cfg: INCAConfig,
    device: str,
    warmup_steps: int,
    total_opt_steps: int,
) -> Tuple[torch.optim.Optimizer, object]:
    """Freeze current block, grow a new one, return (new_optimizer, new_scheduler)."""
    manager.freeze_and_grow()
    params = manager.trainable_params()

    if getattr(cfg, "use_adafactor", False):
        optimizer = Adafactor(
            params, lr=cfg.lr, relative_step=False,
            scale_parameter=False, warmup_init=False,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, max(warmup_steps + 1, total_opt_steps),
    )
    return optimizer, scheduler


# ──────────────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────────────

def train(cfg: INCAConfig, device: str,
          resume_dir: Optional[str] = None) -> str:
    """Full INCA-v2 training loop.

    Parameters
    ----------
    cfg        : validated INCAConfig dataclass
    device     : "cuda" | "mps" | "cpu"
    resume_dir : if set, reuse this exact directory and resume from the latest
                 period checkpoint found inside it.  Periods already checkpointed
                 are skipped; the model/manager state is restored from the last
                 completed period before the new period loop begins.

    Returns
    -------
    str — absolute path to the run output directory (for the orchestrator registry)
    """
    if resume_dir:
        out_dir = Path(resume_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg.out_dir) / f"inca_v2_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

    logger      = _RunLogger(out_dir, dataclasses.asdict(cfg))
    loss_log    = _LossLog(out_dir / "loss_curve.csv")
    signal_log  = _SignalLog(out_dir / "signals.csv")   # per-k_eval trajectory

    # Write an identity marker so the orchestrator can discover this out_dir
    # even if the trainer is launched outside of run_paper_b.py.
    with open(out_dir / "run_id.json", "w") as _f:
        json.dump({"out_dir": str(out_dir), "started_at": datetime.now().isoformat()}, _f)

    # ── reproducibility ────────────────────────────────────────────────
    seed = getattr(cfg, "seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    # ── mixed-precision autocast ───────────────────────────────────────────
    # CUDA: bf16 (Ampere+) or fp16 per config.
    # MPS:  fp32 ONLY — torch.autocast("mps", fp16) is experimental and
    #       produces NaN losses in T5 attention / softmax at step ~300.
    #       Apple MPS runs fp32 fast enough on M-series; no autocast needed.
    # CPU:  disabled.
    _precision = getattr(cfg, "precision", "bf16")
    if device.startswith("cuda"):
        _amp_enabled = _precision in ("bf16", "fp16")
        _amp_dtype   = torch.bfloat16 if _precision == "bf16" else torch.float16
        _amp_dev     = "cuda"
    elif device == "mps":
        _amp_enabled = False          # fp16 autocast on MPS → NaN; stay fp32
        _amp_dtype   = torch.float32
        _amp_dev     = "cpu"          # unused when disabled
    else:
        _amp_enabled = False
        _amp_dtype   = torch.float32
        _amp_dev     = "cpu"

    # ── DataLoader worker count ────────────────────────────────────────
    # Always 0. On CUDA (Linux), the DataLoader is iterated once per epoch
    # from the same loader object.  At each epoch boundary PyTorch calls
    # iter(loader) which forks new workers via the default "fork" start
    # method.  Fork-after-CUDA-init corrupts CUDA file descriptors in the
    # child processes; when they exit they close /dev/nvidia*, which
    # SIGSEGV's the parent mid-epoch-2.  The dataset is already in RAM and
    # GPU training is the bottleneck (>20 batch/s), so workers add nothing.
    _dl_workers = 0

    # ── load dataset ───────────────────────────────────────────────────
    dataset_name = getattr(cfg, "dataset", "cc_news")
    n_per_period = getattr(cfg, "n_per_period", 20_000)
    max_periods  = getattr(cfg, "max_periods",  None)

    logger.log(f"Loading dataset: {dataset_name}  n_per_period={n_per_period}")
    raw_periods: Dict[str, Dataset] = data_module.load_periods(
        dataset_name,
        n_per_period=n_per_period,
        seed=seed,
    )

    period_ids = list(raw_periods.keys())
    if max_periods and len(period_ids) > max_periods:
        period_ids = period_ids[:max_periods]
        raw_periods = {k: raw_periods[k] for k in period_ids}

    logger.log(f"Periods ({len(period_ids)}): {period_ids}")

    # ── Phase 1 resume: scan for completed period checkpoints ─────────────
    # Done here (after period_ids are known, before the model is built) so
    # we can log what will be skipped before expensive model loading starts.
    _resume_periods_done: set = set()
    _resume_ckpt_path: Optional[Path] = None
    if resume_dir:
        _ckpt_files = sorted(
            out_dir.glob("inca_period_*.pt"),
            key=lambda p: period_ids.index(p.stem.replace("inca_period_", ""))
            if p.stem.replace("inca_period_", "") in period_ids else -1,
        )
        for cf in _ckpt_files:
            pid = cf.stem.replace("inca_period_", "")
            if pid in period_ids:
                _resume_periods_done.add(pid)
                _resume_ckpt_path = cf
        if _resume_periods_done:
            logger.log(
                f"[Resume] Found {len(_resume_periods_done)} completed period(s): "
                f"{sorted(_resume_periods_done, key=period_ids.index)}  "
                f"— will load weights from {_resume_ckpt_path.name}"
            )
        else:
            logger.log("[Resume] No period checkpoints found — starting from scratch.")

    # ── tokenizer + base model ─────────────────────────────────────────
    logger.log(f"Loading model: {cfg.model_name}")
    tokenizer  = AutoTokenizer.from_pretrained(cfg.model_name)
    base_model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)

    # DataCollatorForSeq2Seq: pads each batch to its OWN longest sequence,
    # replaces decoder padding tokens with -100, returns "pt" tensors.
    # This is the standard HF approach for seq2seq — replaces static max_length padding.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=base_model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,   # pad to multiple of 8 for CUDA tensor core alignment
        return_tensors="pt",
    )

    if getattr(cfg, "gradient_checkpointing", False):
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
        base_model.config.use_cache = False   # T5 use_cache incompatible with GC

    base_model.to(device)

    # ── tokenize all periods up-front ──────────────────────────────────
    # ── dynamic max_target_length (0 = auto-compute from data) ───────────
    _cfg_max_tgt = getattr(cfg, "max_target_length", 0)
    if _cfg_max_tgt == 0:
        logger.log("Scanning target lengths for dynamic max_target_length …")
        _max_tgt = compute_percentile_length(
            raw_periods, tokenizer, percentile=95.0, hard_cap=256
        )
        logger.log(f"  → max_target_length = {_max_tgt}  (P95 ≤256, rounded to ×64)")
    else:
        _max_tgt = _cfg_max_tgt

    logger.log("Tokenizing periods …")
    tok_periods: Dict[str, Dataset] = build_tokenized_periods(
        raw_periods,
        tokenizer=tokenizer,
        max_input_length=cfg.max_input_length,
        max_target_length=_max_tgt,
        cache_dir="cache/tokenized",
    )

    # ── INCA manager ───────────────────────────────────────────────────
    manager = INCALayerManager(base_model, cfg).to(device)

    # ── Phase 2 resume: restore model/manager from latest checkpoint ───
    if _resume_ckpt_path is not None:
        logger.log(f"[Resume] Loading checkpoint: {_resume_ckpt_path}")
        _ckpt = torch.load(_resume_ckpt_path, map_location=device)
        base_model.load_state_dict(_ckpt["base_model_state"])
        manager.load_manager_state(_ckpt["manager_state"])
        block_idx      = _ckpt.get("block_idx", 0)
        global_opt_step_offset = _ckpt.get("global_opt_step", 0)
        logger.log(
            f"[Resume] Restored block_idx={block_idx}  "
            f"global_opt_step={global_opt_step_offset}"
        )
    else:
        block_idx = 0
        global_opt_step_offset = 0

    # ── LR schedule parameters (estimated over all periods) ───────────
    accum           = max(1, getattr(cfg, "grad_accum_steps", 1))
    batches_per_ep  = max(1, n_per_period // (cfg.batch_size * accum))
    total_opt_steps = batches_per_ep * cfg.epochs_per_period * len(period_ids)
    # Prefer explicit warmup_steps from config; fall back to warmup_ratio.
    # Previously only warmup_ratio was used, so cfg.warmup_steps was silently ignored.
    _cfg_warmup_steps = getattr(cfg, "warmup_steps", 0)
    warmup_steps = (
        _cfg_warmup_steps if _cfg_warmup_steps > 0
        else max(1, int(total_opt_steps * getattr(cfg, "warmup_ratio", 0.06)))
    )
    logger.log(
        f"  LR schedule: total_opt_steps={total_opt_steps}  "
        f"warmup_steps={warmup_steps}  max_lr={cfg.lr:.2e}"
    )

    # ── initial optimiser ──────────────────────────────────────────────
    params = manager.trainable_params()
    if getattr(cfg, "use_adafactor", False):
        optimizer = Adafactor(
            params, lr=cfg.lr, relative_step=False,
            scale_parameter=False, warmup_init=False,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            params, lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, max(warmup_steps + 1, total_opt_steps),
    )

    # ── replay buffer ──────────────────────────────────────────────────
    replay_buf = INCAReplayBuffer(
        max_size_per_period=cfg.buffer_max_size,
        n_revise=cfg.n_revise,
        p_hard=cfg.p_hard,
        p_easy=cfg.p_easy,
        p_mid=cfg.p_mid,
    )

    # ── saturation + CKA monitors ──────────────────────────────────────
    detector    = INCAPlateauDetector(cfg)
    cka_monitor = CKAMonitor(ref_size=cfg.cka_ref_size)

    global_opt_step = global_opt_step_offset   # 0 on fresh start; restored on resume
    # block_idx already set above (0 or restored from checkpoint)
    prev_replay_acc: float = 1.0

    # ── memory / efficiency tracker ────────────────────────────────────────
    mem_tracker = MemoryTracker(device=device, method="inca")

    # ══════════════════════════════════════════════════════════════════
    # Period loop
    # ══════════════════════════════════════════════════════════════════
    for period_idx, period_id in enumerate(period_ids):
        # ── resume: skip already-completed periods ─────────────────────────
        if period_id in _resume_periods_done:
            logger.log(f"[Resume] Skipping {period_id} (checkpoint found).")
            continue

        raw_ds = raw_periods[period_id]
        tok_ds = tok_periods[period_id]
        # No set_format("torch") — sequences are variable-length (no pre-padding).
        # DataCollatorForSeq2Seq handles padding + tensor conversion per batch.

        # ── train / eval split ─────────────────────────────────────────
        eval_frac = getattr(cfg, "ppl_eval_frac", 0.05)
        n_eval    = max(64, int(len(raw_ds) * eval_frac))

        raw_split = raw_ds.train_test_split(test_size=n_eval, seed=seed)
        eval_raw  = raw_split["test"]   # raw rows for greedy-decode eval

        tok_split = tok_ds.train_test_split(test_size=n_eval, seed=seed)
        train_tok = tok_split["train"]
        # No set_format — DataCollatorForSeq2Seq handles this

        logger.log(
            f"\n{'='*60}\n"
            f"Period {period_idx+1}/{len(period_ids)}: {period_id}  "
            f"(train={len(train_tok)}, eval={len(eval_raw)})\n"
            f"Block chain: {manager.n_blocks} block(s)"
        )
        # ── memory tracker: period start ───────────────────────────────────
        mem_tracker.period_start(period_id, base_model)

        # ── CKA reference at period start ──────────────────────────────
        _cache_cka_reference(
            cka_monitor, manager, raw_ds, tokenizer, device,
            max_seq_len=cfg.max_input_length,
            n_samples=cfg.cka_ref_size,
        )

        # ── pre-period baseline eval — REQUIRED for saturation detector ──
        # RIRTracker.rir = (score_now - baseline) / max(baseline, chance).
        # Without calling detector.reset_period(baseline), _baseline = None
        # and rir always returns 0.0, so PERIOD_LEARNED/BLOCK_FULL can never
        # fire from the RIR signal.  Evaluate here before any training so the
        # baseline reflects the model's ability at the START of the period.
        logger.log(f"  Computing pre-period baseline …")
        gc.collect()
        _empty_cache(device)
        _pre_result = _eval_accuracy(
            base_model, manager, eval_raw, tokenizer, device,
            batch_size=getattr(cfg, "eval_batch_size", 32),
            max_input_length=cfg.max_input_length,
            max_new_tokens=getattr(cfg, "max_new_tokens", 256),
            n_samples=getattr(cfg, "sat_eval_samples", 200),
            amp_enabled=_amp_enabled, amp_dtype=_amp_dtype, amp_dev=_amp_dev,
        )
        pre_period_score = _pre_result.token_f1   # float; used as RIR denominator
        detector.reset_period(pre_period_score)
        logger.log(
            f"  Pre-period baseline: f1={pre_period_score:.4f}  "
            f"em={_pre_result.exact_match:.4f}  (RIR denominator = f1)"
        )

        # ── snapshot replay accuracy before this period ───────────────
        had_replay_before = len(replay_buf.periods) > 0
        replay_acc_before: float = prev_replay_acc

        # ── build DataLoader (replay-mixed if buffer non-empty) ────────
        # Sampling uses the current period index as the "epoch" proxy: items
        # added in earlier periods will already have refreshed loss values
        # from previous update_losses() calls, so Phase B's hard/easy/mid
        # schedule has real signal to act on.
        replay_n = getattr(cfg, "replay_n_per_period", 2_000)
        raw_replay = replay_buf.sample(n=replay_n, epoch=period_idx) if had_replay_before else []
        replay_items = _tokenize_replay_items(
            raw_replay, tokenizer,
            max_input_length=cfg.max_input_length,
            max_target_length=_max_tgt,   # use computed P95 length, not raw cfg sentinel (0)
        )

        if replay_items:
            train_loader = make_replay_dataloader(
                stream_dataset=train_tok,
                replay_items=replay_items,
                batch_size=cfg.batch_size,
                replay_ratio=getattr(cfg, "replay_ratio", 0.25),
                seed=seed,
                num_workers=0,       # IterableDataset: always 0 (worker fork breaks state)
                collate_fn=data_collator,
            )
        else:
            train_loader = make_dataloader(
                train_tok,
                batch_size=cfg.batch_size,
                num_workers=_dl_workers,
                collate_fn=data_collator,
            )

        # ── epoch loop ────────────────────────────────────────────────
        period_done      = False
        timeout_counter  = 0
        first_epoch_done = False
        last_grad_norm: float = 0.0

        for epoch in range(cfg.epochs_per_period):
            if period_done:
                break

            manager.train()
            base_model.train()
            accumulate_loss = torch.tensor(0.0, device=device)
            micro_losses: List[float] = []

            total_batches = math.ceil(len(train_tok) / cfg.batch_size)
            pbar = tqdm(
                enumerate(train_loader),
                total=total_batches,
                desc=f"  {period_id} ep{epoch+1}/{cfg.epochs_per_period}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for micro_step, batch in pbar:
                if period_done:
                    break

                batch = _batch_to_device(batch, device)
                with torch.autocast(device_type=_amp_dev, dtype=_amp_dtype, enabled=_amp_enabled):
                    loss = _forward_loss(base_model, manager, batch)

                # NaN/Inf guard — if loss is non-finite, skip the entire
                # accumulation window and reset gradients.  A NaN propagated
                # through backward() corrupts Adam's m/v state permanently.
                if not torch.isfinite(loss):
                    logger.log(
                        f"  [WARNING] non-finite loss={loss.item()} "
                        f"at micro_step={micro_step} opt_step={global_opt_step} "
                        f"— skipping accumulation window"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    accumulate_loss = torch.tensor(0.0, device=device)
                    continue

                (loss / accum).backward()
                accumulate_loss = accumulate_loss + loss.detach()

                if (micro_step + 1) % accum == 0:
                    last_grad_norm = nn.utils.clip_grad_norm_(
                        manager.trainable_params(),
                        max_norm=getattr(cfg, "max_grad_norm", 1.0),
                    ).item()

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_opt_step += 1
                    step_loss = (accumulate_loss / accum).item()
                    accumulate_loss = torch.tensor(0.0, device=device)
                    micro_losses.append(step_loss)
                    pbar.set_postfix(
                        loss=f"{step_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        step=global_opt_step,
                    )

                    log_every = getattr(cfg, "log_every_n_steps", 50)
                    if global_opt_step % log_every == 0:
                        loss_log.log(
                            period_id, block_idx, epoch,
                            global_opt_step, step_loss,
                        )

                    # ── k_eval saturation check ────────────────────────
                    if global_opt_step % cfg.k_eval == 0:
                        _cur_result = _eval_accuracy(
                            base_model, manager, eval_raw, tokenizer, device,
                            batch_size=getattr(cfg, "eval_batch_size", 32),
                            max_input_length=cfg.max_input_length,
                            max_new_tokens=getattr(cfg, "max_new_tokens", 256),
                            n_samples=getattr(cfg, "sat_eval_samples", 200),
                            amp_enabled=_amp_enabled, amp_dtype=_amp_dtype, amp_dev=_amp_dev,
                        )
                        cur_score = _cur_result.token_f1   # float; drives RIR / saturation
                        cka_val  = cka_monitor.compute(manager, device)
                        avg_loss = (
                            sum(micro_losses[-cfg.k_eval:])
                            / max(1, min(len(micro_losses), cfg.k_eval))
                        )

                        detector.update(avg_loss, cur_score, last_grad_norm, cka_value=cka_val)
                        event = detector.check(epoch)

                        # ── signal trajectory (paper figures) ─────────────
                        _det_state = detector.state_dict()
                        signal_log.log(
                            period=period_id,
                            block=block_idx,
                            epoch=epoch,
                            opt_step=global_opt_step,
                            rir=_det_state["rir"],
                            score=cur_score,
                            cka=cka_val,
                            gnorm_ema=_det_state["grad_ema"],
                            avg_loss=avg_loss,
                            event=event.name,
                        )

                        _es_pat = getattr(cfg, "early_stop_patience", 5)
                        logger.log(
                            f"  [e{epoch} s{global_opt_step}] "
                            f"f1={cur_score:.4f} em={_cur_result.exact_match:.4f} "
                            f"cka={cka_val:.3f} "
                            f"rir={_det_state['rir']:.3f} "
                            f"gnorm={last_grad_norm:.4f} loss={avg_loss:.4f} "
                            f"es={_det_state['early_stop_no_improve']}/{_es_pat} "
                            f"→ {event.name}"
                        )

                        if event == SaturationEvent.PERIOD_LEARNED:
                            logger.log("  ✓ PERIOD_LEARNED — advancing to next period.")
                            period_done = True
                            break

                        elif event == SaturationEvent.BLOCK_FULL:
                            # Log grow event with EXP_T for paper Fig: timing distribution
                            logger.grow(
                                period=period_id,
                                block_from=block_idx,
                                block_to=block_idx + 1,
                                opt_step=global_opt_step,
                                score=cur_score,
                                rir=_det_state["rir"],
                                cka=cka_val,
                                grad_norm_ema=_det_state["grad_ema"],
                                trigger="BLOCK_FULL",
                            )
                            optimizer, scheduler = _grow_block(
                                manager, cfg, device, warmup_steps, total_opt_steps,
                            )
                            detector.reset_block()
                            cka_monitor.reset()
                            # Spec §5.1: buffer is per-block. Frozen block weights ARE
                            # the memory — raw examples no longer needed. Clear completely
                            # so the new block starts with a fresh slate (cross-block
                            # replay is meaningless per spec).
                            replay_buf.clear_all()
                            block_idx += 1
                            _empty_cache(device)  # free old Adam states from MPS/CUDA cache
                            period_done = True
                            break

                        else:
                            # Only count toward timeout when:
                            #   1. grokking guard has passed (epoch >= min_epochs_before_grow)
                            #   2. loss has genuinely plateaued
                            # Counting while loss is still improving would terminate training
                            # mid-learning — the timeout is a safety valve for "loss is flat
                            # but no signal fired", not a wall-clock limit on training time.
                            if epoch >= cfg.min_epochs_before_grow and detector.loss_plateau:
                                timeout_counter += 1
                            max_evals = getattr(cfg, "patience_timeout", cfg.patience * 3)
                            if timeout_counter >= max_evals:
                                fallback = detector.check_timeout()
                                logger.log(
                                    f"  Timeout after {timeout_counter} evals "
                                    f"→ {fallback.name}"
                                )
                                # Spec §5.2 T1.2: high RIR → PERIOD_LEARNED (block not
                                # full, advance to next period on same block; replay of
                                # this period's items will protect against within-block
                                # forgetting).  Low RIR → EXHAUSTED → freeze-and-grow.
                                if fallback == SaturationEvent.PERIOD_LEARNED:
                                    logger.log("  ✓ Timeout PERIOD_LEARNED — advancing period, same block.")
                                    period_done = True
                                    break
                                else:  # EXHAUSTED → BLOCK_FULL path
                                    logger.log("  Timeout EXHAUSTED — freeze-and-grow.")
                                    logger.grow(
                                        period=period_id,
                                        block_from=block_idx,
                                        block_to=block_idx + 1,
                                        opt_step=global_opt_step,
                                        score=cur_score,
                                        rir=_det_state["rir"],
                                        cka=cka_val,
                                        grad_norm_ema=_det_state["grad_ema"],
                                        trigger="EXHAUSTED",
                                    )
                                    optimizer, scheduler = _grow_block(
                                        manager, cfg, device,
                                        warmup_steps, total_opt_steps,
                                    )
                                    detector.reset_block()
                                    cka_monitor.reset()
                                    replay_buf.clear_all()  # spec §5.1: frozen weights ARE the memory
                                    block_idx += 1
                                    _empty_cache(device)
                                    period_done = True
                                    break

            # ── after first epoch: populate replay buffer ──────────────
            if not first_epoch_done:
                cap = min(cfg.buffer_max_size, len(raw_ds))
                replay_buf.add_period(
                    period_id,
                    [dict(row) for row in raw_ds.select(range(cap))],
                )
                first_epoch_done = True

        # ── T1.3 drift check after period ────────────────────────────
        if had_replay_before:
            drift = _check_replay_drift(
                base_model, manager, replay_buf,
                prev_acc=replay_acc_before,
                tokenizer=tokenizer,
                device=device,
                tol=cfg.period_drift_tol,
                batch_size=getattr(cfg, "eval_batch_size", 32),
                amp_enabled=_amp_enabled, amp_dtype=_amp_dtype, amp_dev=_amp_dev,
            )
            if drift:
                logger.log("  [T1.3] Replay drift > tol — early BLOCK_FULL.")
                try:
                    optimizer, scheduler = _grow_block(
                        manager, cfg, device, warmup_steps, total_opt_steps,
                    )
                    detector.reset_block()
                    cka_monitor.reset()
                    # Spec §5.1: T1.3 drift forces freeze — same rule, clear all.
                    replay_buf.clear_all()
                    block_idx += 1
                except RuntimeError as exc:
                    logger.log(f"  [T1.3] grow skipped: {exc}")

        # ── post-period eval ───────────────────────────────────────────
        # Release old Adam-state cache BEFORE generation eval: after a grow,
        # the replaced optimiser's ~9 GB of states sit in the MPS/CUDA cache
        # and will OOM the 256-token generation pass if not freed first.
        _empty_cache(device)
        _post_result = _eval_accuracy(
            base_model, manager, eval_raw, tokenizer, device,
            batch_size=getattr(cfg, "eval_batch_size", 32),
            max_input_length=cfg.max_input_length,
            max_new_tokens=getattr(cfg, "max_new_tokens", 256),
            amp_enabled=_amp_enabled, amp_dtype=_amp_dtype, amp_dev=_amp_dev,
        )
        post_score = _post_result.token_f1
        logger.log(
            f"  Post-period  token-F1={post_score:.4f}  "
            f"exact-match={_post_result.exact_match:.4f}"
        )
        prev_replay_acc = post_score
        # ── memory tracker: period end ─────────────────────────────────────
        mem_tracker.period_end(
            period_id, base_model, acc_delta=post_score - pre_period_score
        )
        logger.log(mem_tracker.summary())

        # ── refresh replay-buffer losses for Phase B study schedule ───
        # Without this the buffer's loss values stay 0.0 → Phase B
        # hard/easy/mid sampling degenerates to uniform.
        if period_id in replay_buf.periods:
            stored_items = [e.item for e in replay_buf._store[period_id]]
            gc.collect()          # force Python to drop any lingering eval-pass tensors
            _empty_cache(device)  # release eval-pass activations before per-item scoring
            per_item_losses = _per_item_losses(
                base_model, manager, stored_items, tokenizer, device,
                cfg.max_input_length,
                _max_tgt,   # use computed P95 length, not raw cfg sentinel (0)
                cfg.batch_size,
            )
            replay_buf.update_losses(period_id, stored_items, per_item_losses)

        # ── period checkpoint ──────────────────────────────────────────
        ckpt_path = out_dir / f"inca_period_{period_id}.pt"
        torch.save({
            "period":           period_id,
            "block_idx":        block_idx,
            "global_opt_step":  global_opt_step,
            "manager_state":    manager.manager_state(),
            "base_model_state": base_model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "cfg":              dataclasses.asdict(cfg),
        }, ckpt_path)
        logger.log(f"  Checkpoint → {ckpt_path.name}")

    # ── final checkpoint ───────────────────────────────────────────────
    final_ckpt = out_dir / "inca_v2_final.pt"
    torch.save({
        "period":           period_ids[-1] if period_ids else "none",
        "block_idx":        block_idx,
        "global_opt_step":  global_opt_step,
        "manager_state":    manager.manager_state(),
        "base_model_state": base_model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "cfg":              dataclasses.asdict(cfg),
    }, final_ckpt)
    logger.log(f"\nTraining complete.  Final checkpoint → {final_ckpt}")
    mem_tracker.save(out_dir / "memory_log.json")
    return str(out_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI (called via scripts/train_inca.py)
# ──────────────────────────────────────────────────────────────────────────────

_EXPAND_AT_OVERRIDES: dict[str, dict] = {
    "early":      {"rir_threshold": 99.0, "rir_negligible": 99.0,
                   "patience": 1, "min_epochs_before_grow": 1},
    "saturation": {},
    "late":       {"rir_threshold": 99.0, "patience": 999,
                   "min_epochs_before_grow": 99},
    "never":      {"n_max_blocks": 1},
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train INCA-v2")
    p.add_argument("--config",    required=True, help="Path to YAML config")
    p.add_argument("--dataset",   default=None,  help="Override cfg.dataset")
    p.add_argument("--selector",  default=None,  help="Override cfg.selector")
    p.add_argument("--seed",      type=int, default=None, help="Override cfg.seed")
    p.add_argument("--device",    default=None,  help="cpu | mps | cuda")
    p.add_argument("--expand_at", default=None,
                   choices=list(_EXPAND_AT_OVERRIDES),
                   help="E-TIMING ablation mode: early | saturation | late | never")
    p.add_argument("--resume_dir", default=None,
                   help="Resume from this run directory (reuses out_dir, skips done periods)")
    p.add_argument("--dry-run",   action="store_true",
                   help="Validate config + build model — don't train")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    import yaml
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}

    if args.dataset:   cfg_dict["dataset"]   = args.dataset
    if args.selector:  cfg_dict["selector"]  = args.selector
    if args.seed:      cfg_dict["seed"]      = args.seed
    if args.expand_at: cfg_dict.update(_EXPAND_AT_OVERRIDES[args.expand_at])

    cfg = INCAConfig(**{k: v for k, v in cfg_dict.items()
                        if k in INCAConfig.__dataclass_fields__})

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"INCA-v2  |  model={cfg.model_name}  |  selector={cfg.selector}  "
          f"|  dataset={getattr(cfg, 'dataset', 'cc_news')}  |  device={device}")

    if args.dry_run:
        print("--dry-run: config valid.  Exiting.")
        return

    out_dir = train(cfg, device, resume_dir=args.resume_dir)
    print(f"[train_inca] Run complete → {out_dir}")


if __name__ == "__main__":
    main()

"""Shared continual-learning baseline infrastructure  (training/cl_runner.py)

Replaces all Phase0.common.* imports for baselines B1–B7.

Provides
--------
Period               — dataclass(label, index, train_items)
make_loader          — wraps data.tokenizer.make_dataloader for seq2seq items
model_dtype          — returns torch dtype for a device string
make_epoch_bar       — tqdm epoch progress bar
make_batch_bar       — tqdm batch progress bar
seq2seq_train_loop   — one-period training loop for seq2seq models
standard_train_loop  — one-period training loop for causal-LM models
BaselineRunner       — orchestrates data loading + calls baseline hooks

Usage (inside any baseline)
---------------------------
from training.cl_runner import (
    Period, make_loader, model_dtype,
    make_epoch_bar, make_batch_bar,
    seq2seq_train_loop, standard_train_loop,
    BaselineRunner,
)
from models.inca.config import INCAConfig
"""

from __future__ import annotations

import csv
import dataclasses
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer

# ── HuggingFace datasets import guard (must come before sys.path manipulation) ─
# Baseline scripts do sys.path.insert(0, REPO_ROOT) before importing this module.
# That makes `import datasets` resolve to the local datasets/ data folder instead
# of the installed HuggingFace package.  Strip repo-root and CWD for this import.
_REPO_ROOT_STR = str(Path(__file__).resolve().parent.parent)
_sp_guard = sys.path[:]
sys.path = [p for p in sys.path if p not in ("", ".", _REPO_ROOT_STR)]
from datasets import Dataset
sys.path[:] = _sp_guard
del _sp_guard, _REPO_ROOT_STR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import data as data_module
from data.tokenizer import make_dataloader as _tok_make_dataloader
from models.inca.config import INCAConfig

try:
    from tqdm.auto import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


# ── Utilities ─────────────────────────────────────────────────────────────────

def model_dtype(device: str) -> torch.dtype:
    """Return float16 on CUDA, bfloat16 on MPS (if available), else float32."""
    if "cuda" in device:
        return torch.float16
    if device == "mps" and torch.backends.mps.is_available():
        return torch.bfloat16
    return torch.float32


# ── Period dataclass ──────────────────────────────────────────────────────────

@dataclass
class Period:
    """A single temporal training period."""
    label:       str
    index:       int
    train_items: List[Dict[str, Any]]   # list of {"input_text": str, "target_text": str, ...}

    def __len__(self) -> int:
        return len(self.train_items)


# ── Dataset wrappers ──────────────────────────────────────────────────────────

class Seq2SeqDataset(TorchDataset):
    """Tokenises seq2seq items on-the-fly for a dataloader."""

    def __init__(self, items: List[Dict[str, Any]], tokenizer,
                 max_input_len: int = 256, max_answer_len: int = 256):
        self.items         = items
        self.tokenizer     = tokenizer
        self.max_input_len = max_input_len
        self.max_ans_len   = max_answer_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item  = self.items[idx]
        enc   = self.tokenizer(
            item["input_text"],
            truncation=True, max_length=self.max_input_len,
            padding="max_length", return_tensors="pt",
        )
        dec   = self.tokenizer(
            item["target_text"],
            truncation=True, max_length=self.max_ans_len,
            padding="max_length", return_tensors="pt",
        )
        labels = dec["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


class TextDataset(TorchDataset):
    """Tokenises causal-LM text items on-the-fly."""

    def __init__(self, items: List[Dict[str, Any]], tokenizer, max_seq_len: int = 512):
        self.items       = items
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        text = item.get("input_text", item.get("text", ""))
        enc  = self.tokenizer(
            text, truncation=True, max_length=self.max_seq_len,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def make_loader(
    items: List[Dict[str, Any]],
    tokenizer,
    batch_size: int = 32,
    max_seq_len: int = 256,
    shuffle: bool = True,
    model_type: str = "seq2seq",
    max_answer_len: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from a list of period items."""
    if model_type == "seq2seq":
        ds = Seq2SeqDataset(items, tokenizer,
                            max_input_len=max_seq_len, max_answer_len=max_answer_len)
    else:
        ds = TextDataset(items, tokenizer, max_seq_len=max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, num_workers=num_workers)


# ── Progress bars ─────────────────────────────────────────────────────────────

def make_epoch_bar(n_epochs: int, period_label: str, period_idx: int, n_periods: int):
    desc = f"P{period_idx+1}/{n_periods} {period_label}"
    if _TQDM:
        return tqdm(range(n_epochs), desc=desc, leave=True)
    return range(n_epochs)


def make_batch_bar(dataloader, epoch: int, n_epochs: int):
    if _TQDM:
        return tqdm(dataloader, desc=f"  ep{epoch}", leave=False)
    return dataloader


# ── Training loops ────────────────────────────────────────────────────────────

def seq2seq_train_loop(
    model,
    optimizer,
    dataloader: DataLoader,
    device: str,
    cfg: INCAConfig,
    period_label: str,
    period_idx: int,
    n_periods: int,
    extra_loss_fn: Optional[Callable[[], torch.Tensor]] = None,
    scheduler=None,
    loss_logger=None,
    text_logger=None,
) -> float:
    """Standard seq2seq training loop (used by B1, B3, B6)."""
    n_epochs    = cfg.epochs_per_period
    accum_steps = max(1, getattr(cfg, "grad_accum_steps", 1))
    log_every   = max(1, getattr(cfg, "log_every_n_steps", 50))
    max_grad    = getattr(cfg, "max_grad_norm", 1.0)

    epoch_bar = make_epoch_bar(n_epochs, period_label, period_idx, n_periods)
    last_loss = 0.0
    opt_step  = 0

    for epoch in epoch_bar:
        model.train()
        total, n   = 0.0, 0
        accum_loss = 0.0
        batch_bar  = make_batch_bar(dataloader, epoch, n_epochs)

        for micro_step, batch in enumerate(batch_bar, 1):
            ids    = batch["input_ids"].to(device)
            mask   = batch.get("attention_mask")
            if mask is not None: mask = mask.to(device)
            labels = batch["labels"].to(device)
            out    = model(input_ids=ids, attention_mask=mask, labels=labels)

            if not torch.isfinite(out.loss):
                continue

            loss = out.loss
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn()
            (loss / accum_steps).backward()
            accum_loss += out.loss.item()

            if (micro_step % accum_steps == 0) or (micro_step == len(dataloader)):
                nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                opt_step  += 1
                step_loss  = accum_loss / accum_steps
                accum_loss = 0.0
                total     += step_loss
                n         += 1
                if device == "mps" and opt_step % log_every == 0:
                    torch.mps.empty_cache()
                if loss_logger is not None and opt_step % log_every == 0:
                    loss_logger(period_label, epoch, opt_step, step_loss)
                if _TQDM and hasattr(batch_bar, "set_postfix"):
                    batch_bar.set_postfix(
                        loss=f"{step_loss:.4f}", avg=f"{total/n:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

        last_loss = total / max(n, 1)
        if _TQDM and hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(avg_loss=f"{last_loss:.4f}")
        if loss_logger is not None:
            loss_logger(period_label, epoch, opt_step, last_loss)

    return last_loss


def standard_train_loop(
    model,
    optimizer,
    dataloader: DataLoader,
    device: str,
    cfg: INCAConfig,
    period_label: str,
    period_idx: int,
    n_periods: int,
    extra_loss_fn: Optional[Callable[[], torch.Tensor]] = None,
    scheduler=None,
    loss_logger=None,
    text_logger=None,
) -> float:
    """Causal-LM training loop."""
    import torch.nn.functional as F

    n_epochs    = cfg.epochs_per_period
    accum_steps = max(1, getattr(cfg, "grad_accum_steps", 1))
    log_every   = max(1, getattr(cfg, "log_every_n_steps", 50))
    max_grad    = getattr(cfg, "max_grad_norm", 1.0)

    epoch_bar = make_epoch_bar(n_epochs, period_label, period_idx, n_periods)
    last_loss = 0.0
    opt_step  = 0

    for epoch in epoch_bar:
        model.train()
        total, n   = 0.0, 0
        accum_loss = 0.0
        batch_bar  = make_batch_bar(dataloader, epoch, n_epochs)

        for micro_step, batch in enumerate(batch_bar, 1):
            ids  = batch["input_ids"].to(device)
            mask = batch.get("attention_mask")
            if mask is not None: mask = mask.to(device)
            labels = ids.clone()
            if mask is not None: labels[mask == 0] = -100
            out = model(input_ids=ids, attention_mask=mask, labels=labels)

            if not torch.isfinite(out.loss):
                continue

            loss = out.loss
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn()
            (loss / accum_steps).backward()
            accum_loss += out.loss.item()

            if (micro_step % accum_steps == 0) or (micro_step == len(dataloader)):
                nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                opt_step  += 1
                step_loss  = accum_loss / accum_steps
                accum_loss = 0.0
                total     += step_loss
                n         += 1
                if device == "mps" and opt_step % log_every == 0:
                    torch.mps.empty_cache()
                if loss_logger is not None and opt_step % log_every == 0:
                    loss_logger(period_label, epoch, opt_step, step_loss)
                if _TQDM and hasattr(batch_bar, "set_postfix"):
                    batch_bar.set_postfix(
                        loss=f"{step_loss:.4f}", avg=f"{total/n:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

        last_loss = total / max(n, 1)
        if _TQDM and hasattr(epoch_bar, "set_postfix"):
            epoch_bar.set_postfix(avg_loss=f"{last_loss:.4f}")
        if loss_logger is not None:
            loss_logger(period_label, epoch, opt_step, last_loss)

    return last_loss


# ── Loggers ───────────────────────────────────────────────────────────────────

class _RunLogger:
    def __init__(self, out_dir: Path, cfg_snapshot: dict) -> None:
        self._path = out_dir / "run_log.jsonl"
        self._write({"event": "config", "cfg": cfg_snapshot})

    def log(self, msg: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"[{ts}] {msg}", flush=True)
        self._write({"event": "log", "msg": msg, "ts": ts})

    def _write(self, record: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class _LossLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["period", "epoch", "opt_step", "loss", "timestamp"])

    def __call__(self, period: str, epoch: int, step: int, loss: float) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [period, epoch, step, f"{loss:.6f}",
                 datetime.now().isoformat(timespec="seconds")]
            )


# ── Probe loading + EM evaluation (for BWT/FWT regret matrix) ────────────────

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_probe_split(cfg: INCAConfig, period_id: str) -> List[Dict[str, Any]]:
    """Load frozen v2 evaluation probes for *period_id* from probes_v2/ JSONL."""
    dataset_name = getattr(cfg, "dataset", "cc_news")
    path = (
        _REPO_ROOT / "datasets" / dataset_name / "processed"
        / "probes_v2" / f"{period_id}.jsonl"
    )
    if not path.exists():
        return []
    probes: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                probes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return probes


@torch.no_grad()
def _eval_cloze_accuracy(
    model,
    probes: List[Dict[str, Any]],
    tokenizer,
    device: str,
    batch_size: int = 32,
    max_input_len: int = 256,
    max_new_tokens: int = 32,
) -> float:
    """Exact-match accuracy on entity_cloze + date_cloze probes from probes_v2 JSONL.

    Uses the pre-formatted `input` field directly (already period-prefixed).
    Checks the generated text against `answer` (and `aliases` when present).
    Returns accuracy in [0, 1]; returns 0.0 if no cloze probes are present.
    """
    cloze = [p for p in probes
             if p.get("probe_type") in ("entity_cloze", "date_cloze")]
    if not cloze:
        return 0.0

    def _normalise(s: str) -> str:
        import re, string
        s = s.lower().translate(str.maketrans("", "", string.punctuation))
        return re.sub(r"\s+", " ", s).strip()

    model.eval()
    correct = total = 0

    for start in range(0, len(cloze), batch_size):
        chunk = cloze[start: start + batch_size]
        enc = tokenizer(
            [p["input"] for p in chunk],
            truncation=True, max_length=max_input_len,
            padding=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        try:
            gen = model.generate(**enc, max_new_tokens=max_new_tokens)
        except Exception:
            continue
        preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for pred, probe in zip(preds, chunk):
            total += 1
            gold    = _normalise(probe.get("answer", ""))
            pred_n  = _normalise(pred)
            aliases = [_normalise(a) for a in (probe.get("aliases") or [])]
            if pred_n == gold or pred_n in aliases or (gold and gold in pred_n):
                correct += 1

    return correct / max(total, 1)


# ── BaselineRunner ────────────────────────────────────────────────────────────

class BaselineRunner:
    """Orchestrates data loading and calls each baseline's standard hooks.

    Expected interface on *baseline* object:
        baseline.name                  str
        baseline.build_model(tok, dev) → nn.Module
        baseline.scoring_model()       → nn.Module
        baseline.on_period_start(p)
        baseline.train_period(p, scheduler, loss_logger, text_logger) → float
        baseline.on_period_end(p)
    """

    def __init__(self, cfg: INCAConfig, baseline, device: Optional[str] = None):
        self.cfg      = cfg
        self.baseline = baseline
        self.device   = device or self._auto_device()

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():      return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def run(self) -> None:
        cfg      = self.cfg
        baseline = self.baseline
        device   = self.device

        dataset_name = getattr(cfg, "dataset", "cc_news")
        n_per_period = getattr(cfg, "n_per_period", 20_000)
        max_periods  = getattr(cfg, "max_periods",  None)
        seed         = getattr(cfg, "seed", 42)
        max_seq_len  = getattr(cfg, "max_input_length", 256)
        max_ans_len  = getattr(cfg, "max_target_length", max_seq_len)

        random.seed(seed)
        torch.manual_seed(seed)

        # ── output dir ────────────────────────────────────────────────────────
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(getattr(cfg, "out_dir", "results")) / f"{baseline.name}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger   = _RunLogger(out_dir, dataclasses.asdict(cfg))
        loss_log = _LossLog(out_dir / "loss_curve.csv")

        logger.log(f"Baseline: {baseline.name} | dataset={dataset_name} | device={device}")

        # ── load data ─────────────────────────────────────────────────────────
        logger.log(f"Loading {dataset_name} …")
        raw_periods: Dict[str, Dataset] = data_module.load_periods(
            dataset_name, n_per_period=n_per_period, seed=seed,
        )
        period_ids = list(raw_periods.keys())
        if max_periods:
            period_ids = period_ids[:max_periods]

        # ── tokeniser & model ─────────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        baseline.build_model(tokenizer, device)

        # ── scheduler (cosine with warmup) ────────────────────────────────────
        from transformers.optimization import get_cosine_schedule_with_warmup

        n_total_steps = (
            max(1, n_per_period // cfg.batch_size)
            * cfg.epochs_per_period
            * len(period_ids)
        )
        warmup_steps = max(1, int(n_total_steps * getattr(cfg, "warmup_ratio", 0.06)))

        _opt      = getattr(baseline, "_optimizer", None)
        scheduler = None
        if _opt is not None:
            scheduler = get_cosine_schedule_with_warmup(
                _opt, warmup_steps, max(warmup_steps + 1, n_total_steps),
            )

        # ── BWT/FWT regret matrix ─────────────────────────────────────────────
        # R[t][j] = accuracy of model after training period t on period j's probes.
        # Diagonal = in-distribution.  Lower triangle = BWT.  Upper = FWT.
        # We use StreamAccuracyMatrix from evaluation/metrics.py and write a CSV.
        from evaluation.metrics import StreamAccuracyMatrix, bwt as _bwt, acc as _acc, fwt as _fwt
        mat = StreamAccuracyMatrix(matrix=[], labels=period_ids)
        probes_by_period: Dict[str, List[Dict[str, Any]]] = {}

        # ── main loop ─────────────────────────────────────────────────────────
        for idx, pid in enumerate(period_ids):
            ds = raw_periods[pid]
            items: List[Dict[str, Any]] = [
                {"input_text": row["input_text"], "target_text": row["target_text"]}
                for row in ds
            ]
            period = Period(label=pid, index=idx, train_items=items)
            logger.log(f"\nPeriod {idx+1}/{len(period_ids)}: {pid}  ({len(items)} items)")

            # Load frozen probes for this period
            probes_by_period[pid] = _load_probe_split(cfg, pid)
            n_cloze = sum(
                1 for p in probes_by_period[pid]
                if p.get("probe_type") in ("entity_cloze", "date_cloze")
            )
            logger.log(f"  Probes: {len(probes_by_period[pid])} total, {n_cloze} cloze")

            # FWT: evaluate on this period's probes BEFORE training it.
            # Gives R[idx-1, idx] — how much the model already knows.
            if idx > 0:
                fwt_acc = _eval_cloze_accuracy(
                    baseline.scoring_model(), probes_by_period[pid], tokenizer, device,
                )
                mat.set(idx - 1, idx, fwt_acc)
                logger.log(f"  FWT pre-train  R[{idx-1},{idx}]={fwt_acc:.4f}")

            # ── FIX: call on_period_start BEFORE rebinding scheduler ──────────
            # B5/B6/B7 create/replace self._optimizer inside on_period_start().
            # The old code bound the scheduler first, then called on_period_start(),
            # leaving B6/B7's first period without a scheduler and later periods
            # with a scheduler attached to a stale optimizer.
            baseline.on_period_start(period)

            # Rebind scheduler to current optimizer AFTER on_period_start()
            _opt_now = getattr(baseline, "_optimizer", None)
            if _opt_now is not None and _opt_now is not _opt:
                _opt = _opt_now
                scheduler = get_cosine_schedule_with_warmup(
                    _opt,
                    max(1, int(n_total_steps * getattr(cfg, "warmup_ratio", 0.06))),
                    max(warmup_steps + 1, n_total_steps),
                )

            avg_loss = baseline.train_period(
                period,
                scheduler=scheduler,
                loss_logger=loss_log,
                text_logger=logger.log,
            )
            baseline.on_period_end(period)
            logger.log(f"  Period {pid} done | avg_loss={avg_loss:.4f}")

            # After training: evaluate on probes from all periods 0..idx.
            # Fills the diagonal (in-distribution) and lower triangle (BWT).
            score_model = baseline.scoring_model()
            for j, jpid in enumerate(period_ids[: idx + 1]):
                if jpid not in probes_by_period:
                    continue
                row_acc = _eval_cloze_accuracy(
                    score_model, probes_by_period[jpid], tokenizer, device,
                )
                mat.set(idx, j, row_acc)
                tag = "diag" if j == idx else "BWT "
                logger.log(f"  [{tag}] R[{idx},{j}] ({pid}→{jpid})={row_acc:.4f}")

            # Save checkpoint
            ckpt_path = out_dir / f"{baseline.name}_period_{pid}.pt"
            torch.save(score_model.state_dict(), ckpt_path)
            logger.log(f"  Checkpoint → {ckpt_path.name}")

        # ── Final BWT / FWT / ACC ─────────────────────────────────────────────
        bwt_score = _bwt(mat)
        acc_score = _acc(mat)
        fwt_score = _fwt(mat)

        sep = "=" * 52
        logger.log(f"\n{sep}")
        logger.log(f"  BWT = {bwt_score:+.4f}   (↑ less forgetting;  target: beat B6)")
        logger.log(f"  ACC = {acc_score:.4f}    (probe accuracy at end of stream)")
        logger.log(f"  FWT = {fwt_score:+.4f}   (↑ better forward transfer)")
        logger.log(f"{sep}")

        # Save regret matrix CSV
        import csv as _csv
        mat_path = out_dir / "regret_matrix.csv"
        with open(mat_path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["after_period"] + [f"probes_{p}" for p in period_ids])
            for t_idx, t_pid in enumerate(period_ids):
                w.writerow(
                    [t_pid] + [f"{mat.get(t_idx, j):.4f}" for j in range(len(period_ids))]
                )
        logger.log(f"  Regret matrix → {mat_path.name}")

        # Save summary JSON (feed into baselines_report.md)
        summary = {
            "baseline": baseline.name,
            "dataset": dataset_name,
            "n_periods": len(period_ids),
            "period_ids": period_ids,
            "BWT": round(bwt_score, 4),
            "ACC": round(acc_score, 4),
            "FWT": round(fwt_score, 4),
            "regret_matrix": mat.matrix,
        }
        summary_path = out_dir / "metrics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.log(f"  Metrics summary → {summary_path.name}")
        logger.log(f"\nBaseline {baseline.name} complete → {out_dir}")

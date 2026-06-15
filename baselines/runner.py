"""Continual-learning baseline runner  (baselines/runner.py)

Single source of truth for all B1-B7 baselines. Self-contained:
no separate data package or training package needed.

Run any baseline from repo root:
    python baselines/b1_finetune.py --config configs/base.yaml --device mps
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

# ── Add repo root to sys.path once here — baselines use `from runner import ...`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.inca.config import INCAConfig          # re-exported for baselines
from evaluation.metrics import (
    StreamAccuracyMatrix,
    bwt as _bwt,
    acc as _acc,
    fwt as _fwt,
)

try:
    from tqdm.auto import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

_DATASETS_ROOT = _ROOT / "datasets"
_CC_STREAM_V2  = _DATASETS_ROOT / "cc_news" / "processed" / "stream_v2"
_CC_PROBES_V2  = _DATASETS_ROOT / "cc_news" / "processed" / "probes_v2"
_CC_GOOD_PERIODS = ["2017_H1", "2017_H2", "2018_H1", "2018_H2"]


# ── dtype ─────────────────────────────────────────────────────────────────────

def model_dtype(device: str) -> torch.dtype:
    return torch.float16 if "cuda" in device else torch.float32


# ── CC-News v2 inline loader — no HuggingFace Dataset, no data package ────────

def _load_cc_news_v2(
    n_per_period: int = 20_000,
    seed: int = 42,
    max_periods: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if not _CC_STREAM_V2.exists():
        raise FileNotFoundError(
            f"CC-News v2 stream not found at {_CC_STREAM_V2}.\n"
            "Run:  python preprocessing/run.py cc_news --force"
        )
    available = sorted(
        p.stem for p in _CC_STREAM_V2.glob("*.jsonl") if p.stat().st_size > 100
    )
    periods = [p for p in _CC_GOOD_PERIODS if p in available] or available
    if max_periods:
        periods = periods[:max_periods]

    rng    = random.Random(seed)
    result: Dict[str, List[Dict[str, Any]]] = {}
    for period in periods:
        rows: List[Dict[str, Any]] = []
        with open(_CC_STREAM_V2 / f"{period}.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                inp = (obj.get("input") or "").strip()
                tgt = (obj.get("target") or "").strip()
                if inp and tgt:
                    rows.append({"input_text": inp, "target_text": tgt, "period": period})
        rng.shuffle(rows)
        result[period] = rows[:n_per_period]
        print(f"  CC-News v2  {period}: {len(result[period]):,} training examples")
    return result


def _load_probes(period_id: str) -> List[Dict[str, Any]]:
    path = _CC_PROBES_V2 / f"{period_id}.jsonl"
    if not path.exists():
        return []
    probes: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                probes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return probes


# ── Dataset + DataLoader ──────────────────────────────────────────────────────

@dataclass
class Period:
    label:       str
    index:       int
    train_items: List[Dict[str, Any]]

    def __len__(self) -> int:
        return len(self.train_items)


class Seq2SeqDataset(TorchDataset):
    def __init__(self, encoded: List[Dict[str, List[int]]]):
        """Accepts pre-tokenized items (lists of ints, no padding yet)."""
        self.data = encoded

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.data[idx]


def _pre_tokenize(
    items: List[Dict[str, Any]],
    tokenizer,
    max_input_len: int,
    max_answer_len: int,
    chunk: int = 1024,
) -> List[Dict[str, List[int]]]:
    """Tokenize all items in one pass (batched). Returns list of dicts with raw int lists.

    Padding is intentionally NOT applied here — the collator handles it per-batch
    so short sequences don't waste computation on padding tokens.
    """
    pad_id = tokenizer.pad_token_id or 0
    out: List[Dict[str, List[int]]] = []
    for i in range(0, len(items), chunk):
        batch  = items[i : i + chunk]
        enc    = tokenizer(
            [x["input_text"]  for x in batch],
            truncation=True, max_length=max_input_len, padding=False,
        )
        dec    = tokenizer(
            [x["target_text"] for x in batch],
            truncation=True, max_length=max_answer_len, padding=False,
        )
        for j in range(len(batch)):
            # Replace pad tokens in labels with -100 (loss ignores them)
            labels = [
                t if t != pad_id else -100
                for t in dec["input_ids"][j]
            ]
            out.append({
                "input_ids":      enc["input_ids"][j],
                "attention_mask": enc["attention_mask"][j],
                "labels":         labels,
            })
    return out


def _make_collate(pad_id: int):
    """Dynamic padding collate: pad each batch to its own longest sequence."""
    def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_in  = max(len(x["input_ids"]) for x in batch)
        max_lb  = max(len(x["labels"])    for x in batch)
        B       = len(batch)
        in_ids  = torch.full((B, max_in), pad_id,  dtype=torch.long)
        attn    = torch.zeros(B, max_in,            dtype=torch.long)
        lbl     = torch.full((B, max_lb), -100,     dtype=torch.long)
        for i, x in enumerate(batch):
            li = len(x["input_ids"]); ll = len(x["labels"])
            in_ids[i, :li] = torch.tensor(x["input_ids"],  dtype=torch.long)
            attn[i,   :li] = torch.tensor(x["attention_mask"], dtype=torch.long)
            lbl[i,    :ll] = torch.tensor(x["labels"],     dtype=torch.long)
        return {"input_ids": in_ids, "attention_mask": attn, "labels": lbl}
    return collate


def make_loader(
    items: List[Dict[str, Any]],
    tokenizer,
    batch_size:     int  = 32,
    max_seq_len:    int  = 256,
    shuffle:        bool = True,
    max_answer_len: int  = 256,
    drop_last:      bool = False,
    num_workers:    int  = 0,
) -> DataLoader:
    """Pre-tokenize all items once, then wrap in a DataLoader with dynamic padding."""
    encoded = _pre_tokenize(items, tokenizer, max_seq_len, max_answer_len)
    ds      = Seq2SeqDataset(encoded)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers,
        collate_fn=_make_collate(tokenizer.pad_token_id or 0),
        pin_memory=False,
    )


# ── Progress bars ─────────────────────────────────────────────────────────────

def make_epoch_bar(n_epochs, period_label, period_idx, n_periods):
    desc = f"[{period_idx+1}/{n_periods}] {period_label}"
    return tqdm(range(n_epochs), desc=desc, unit="epoch", leave=False) if _TQDM else range(n_epochs)


def make_batch_bar(dataloader, epoch, n_epochs):
    return tqdm(dataloader, desc=f"  epoch {epoch+1}/{n_epochs}", unit="batch", leave=False) if _TQDM else dataloader


# ── Shared seq2seq training loop ──────────────────────────────────────────────

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
    accum     = max(1, getattr(cfg, "grad_accum_steps", 1))
    log_every = max(1, getattr(cfg, "log_every_n_steps", 50))
    max_grad  = getattr(cfg, "max_grad_norm", 1.0)
    epoch_bar = make_epoch_bar(cfg.epochs_per_period, period_label, period_idx, n_periods)
    last_loss = 0.0
    opt_step  = 0

    for epoch in epoch_bar:
        model.train()
        total = n = 0
        accum_loss = 0.0
        batch_bar  = make_batch_bar(dataloader, epoch, cfg.epochs_per_period)
        for ms, batch in enumerate(batch_bar, 1):
            ids    = batch["input_ids"].to(device)
            mask   = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(device)
            labels = batch["labels"].to(device)
            out    = model(input_ids=ids, attention_mask=mask, labels=labels)
            if not torch.isfinite(out.loss):
                continue
            loss = out.loss + (extra_loss_fn() if extra_loss_fn else 0.0)
            (loss / accum).backward()
            accum_loss += out.loss.item()
            if ms % accum == 0 or ms == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                opt_step += 1
                sl = accum_loss / accum
                accum_loss = 0.0
                total += sl
                n     += 1
                if loss_logger and opt_step % log_every == 0:
                    loss_logger(period_label, epoch, opt_step, sl)
        last_loss = total / max(n, 1)
        if loss_logger:
            loss_logger(period_label, epoch, opt_step, last_loss)
    return last_loss


# ── Cloze probe evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def eval_cloze_accuracy(
    model, probes: List[Dict[str, Any]], tokenizer, device: str,
    batch_size: int = 32, max_input_len: int = 256, max_new_tokens: int = 32,
) -> float:
    import string
    cloze = [p for p in probes if p.get("probe_type") in ("entity_cloze", "date_cloze")]
    if not cloze:
        return 0.0

    def _norm(s: str) -> str:
        return " ".join(s.lower().translate(str.maketrans("", "", string.punctuation)).split())

    model.eval()
    correct = total = 0
    for i in range(0, len(cloze), batch_size):
        batch  = cloze[i: i + batch_size]
        inputs = tokenizer(
            [p["input"] for p in batch], return_tensors="pt",
            padding=True, truncation=True, max_length=max_input_len,
        ).to(device)
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
        preds   = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        for pred, probe in zip(preds, batch):
            answer   = probe.get("target") or probe.get("answer") or ""
            aliases  = probe.get("aliases") or []
            accepted = {_norm(answer)} | {_norm(a) for a in aliases}
            if _norm(pred) in accepted:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


# ── Logger helpers ────────────────────────────────────────────────────────────

class _RunLogger:
    def __init__(self, out_dir: Path, cfg_snapshot: dict) -> None:
        self._path = out_dir / "run_log.txt"
        self._path.write_text("")
        self.log(f"config: {json.dumps(cfg_snapshot, indent=2, default=str)}")

    def log(self, msg: str) -> None:
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class _LossLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["period", "epoch", "step", "loss"])

    def __call__(self, period: str, epoch: int, step: int, loss: float) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([period, epoch, step, f"{loss:.6f}"])


# ── BaselineRunner ────────────────────────────────────────────────────────────

class BaselineRunner:
    """Drives any B1-B7 baseline through the CC-News v2 temporal stream."""

    def __init__(self, cfg: INCAConfig, baseline, device: Optional[str] = None):
        self.cfg      = cfg
        self.baseline = baseline
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def run(self) -> None:
        cfg      = self.cfg
        baseline = self.baseline
        device   = self.device

        n_per_period = getattr(cfg, "n_per_period", 20_000)
        max_periods  = getattr(cfg, "max_periods",  None)
        seed         = getattr(cfg, "seed", 42)

        random.seed(seed)
        torch.manual_seed(seed)

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(getattr(cfg, "out_dir", "results")) / f"{baseline.name}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        logger   = _RunLogger(out_dir, dataclasses.asdict(cfg))
        loss_log = _LossLog(out_dir / "loss_curve.csv")

        logger.log(f"Baseline: {baseline.name} | device={device}")

        # Load data
        logger.log("Loading CC-News v2 …")
        raw        = _load_cc_news_v2(n_per_period=n_per_period, seed=seed, max_periods=max_periods)
        period_ids = list(raw.keys())

        # Tokeniser + model
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        baseline.build_model(tokenizer, device)

        # Gradient checkpointing — recomputes activations on backward pass instead of
        # storing them, trading ~20% speed for ~40% peak memory reduction.
        if getattr(cfg, "gradient_checkpointing", False):
            m = getattr(baseline, "_model", None) or getattr(baseline, "_backbone", None)
            if m is not None and hasattr(m, "gradient_checkpointing_enable"):
                m.gradient_checkpointing_enable()
                logger.log("  gradient_checkpointing: ON")

        # Scheduler
        from transformers.optimization import get_cosine_schedule_with_warmup
        n_total_steps = max(1, n_per_period // cfg.batch_size) * cfg.epochs_per_period * len(period_ids)
        warmup_steps  = max(1, int(n_total_steps * getattr(cfg, "warmup_ratio", 0.06)))
        _opt      = getattr(baseline, "_optimizer", None)
        scheduler = None
        if _opt is not None:
            scheduler = get_cosine_schedule_with_warmup(_opt, warmup_steps, max(warmup_steps + 1, n_total_steps))

        # BWT/FWT matrix
        mat          = StreamAccuracyMatrix(matrix=[], labels=period_ids)
        probes_cache: Dict[str, List[Dict[str, Any]]] = {}

        for idx, pid in enumerate(period_ids):
            items  = raw[pid]
            period = Period(label=pid, index=idx, train_items=items)
            logger.log(f"\nPeriod {idx+1}/{len(period_ids)}: {pid}  ({len(items):,} items)")

            probes_cache[pid] = _load_probes(pid)
            n_cloze = sum(1 for p in probes_cache[pid] if p.get("probe_type") in ("entity_cloze", "date_cloze"))
            logger.log(f"  Probes: {len(probes_cache[pid])} total, {n_cloze} cloze")

            if idx > 0:
                fwt_acc = eval_cloze_accuracy(baseline.scoring_model(), probes_cache[pid], tokenizer, device)
                mat.set(idx - 1, idx, fwt_acc)
                logger.log(f"  FWT  R[{idx-1},{idx}]={fwt_acc:.4f}")

            # on_period_start BEFORE rebinding scheduler
            baseline.on_period_start(period)
            _opt_now = getattr(baseline, "_optimizer", None)
            if _opt_now is not None and _opt_now is not _opt:
                _opt = _opt_now
                scheduler = get_cosine_schedule_with_warmup(
                    _opt,
                    max(1, int(n_total_steps * getattr(cfg, "warmup_ratio", 0.06))),
                    max(warmup_steps + 1, n_total_steps),
                )

            avg_loss = baseline.train_period(period, scheduler=scheduler,
                                             loss_logger=loss_log, text_logger=logger.log)
            baseline.on_period_end(period)
            logger.log(f"  Period {pid} done | avg_loss={avg_loss:.4f}")

            # Flush MPS/CUDA cache between periods to reclaim fragmented memory
            if device == "mps":
                torch.mps.empty_cache()
            elif "cuda" in device:
                torch.cuda.empty_cache()

            score_model = baseline.scoring_model()
            for j, jpid in enumerate(period_ids[: idx + 1]):
                if jpid not in probes_cache:
                    continue
                row_acc = eval_cloze_accuracy(score_model, probes_cache[jpid], tokenizer, device)
                mat.set(idx, j, row_acc)
                logger.log(f"  [{'diag' if j == idx else 'BWT '}] R[{idx},{j}] ({pid}→{jpid})={row_acc:.4f}")

            ckpt_path = out_dir / f"{baseline.name}_period_{pid}.pt"
            torch.save(score_model.state_dict(), ckpt_path)
            logger.log(f"  Checkpoint → {ckpt_path.name}")

        # Final metrics
        bwt_score = _bwt(mat)
        acc_score = _acc(mat)
        fwt_score = _fwt(mat)
        logger.log(f"\n{'='*52}")
        logger.log(f"  BWT={bwt_score:+.4f}  ACC={acc_score:.4f}  FWT={fwt_score:+.4f}")
        logger.log(f"{'='*52}")

        import csv as _csv
        mat_path = out_dir / "regret_matrix.csv"
        with open(mat_path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["after_period"] + [f"probes_{p}" for p in period_ids])
            for t_idx, t_pid in enumerate(period_ids):
                w.writerow([t_pid] + [f"{mat.get(t_idx, j):.4f}" for j in range(len(period_ids))])

        summary = {"baseline": baseline.name, "n_periods": len(period_ids),
                   "period_ids": period_ids, "BWT": round(bwt_score, 4),
                   "ACC": round(acc_score, 4), "FWT": round(fwt_score, 4),
                   "regret_matrix": mat.matrix}
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        logger.log(f"Done → {out_dir}")

"""Phase 0 shared evaluation harness (v2: seq2seq + causal, CAPSEL metrics).

Fully self-contained. The harness wraps four things:

1. :class:`TextDataset` — batch-tokenised causal-LM documents (Track B).
2. :class:`Seq2SeqDataset` — batch-tokenised QA pairs for encoder-decoder
   models (Track A). Input = "question: {q} context: {evidence}", label =
   "{answer}".
3. :func:`load_periods` — turns :mod:`common.datasets` output into a list of
   :class:`Period` records with deterministic train/eval splits. In seq2seq
   mode the probes are used as training items (80/20 split); in causal mode
   the stream docs are used.
4. :class:`RunLogger` — owns the per-baseline output directory and writes
   ``metrics.json`` / ``training.log`` / ``config.snapshot.json`` /
   ``probes_period*.csv``.

All metrics imported from :mod:`common.metrics` (CAPSEL XIII definitions).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .datasets import PeriodData, load_dataset
from .metrics import (
    StreamAccuracyMatrix,
    acc as stream_acc,
    bwt as stream_bwt,
    combined_score,
    fwt as stream_fwt,
    perplexity,
    ppl_to_score,
    probe_accuracy,
    rir,
    seq2seq_combined_score,
    seq2seq_perplexity,
    seq2seq_probe_accuracy,
)


# ── Causal-LM dataset ────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Pre-tokenises ``{"text": ...}`` items once at construction time.

    Items shorter than 10 characters are dropped.
    """

    def __init__(self, items: List[Dict[str, Any]], tokenizer, max_len: int = 128,
                 _batch: int = 2048):
        texts = [it["text"] for it in items if len(it.get("text", "")) >= 10]
        self._data: List[Dict[str, torch.Tensor]] = []
        for start in range(0, len(texts), _batch):
            chunk = texts[start: start + _batch]
            enc = tokenizer(
                chunk,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            for i in range(len(chunk)):
                self._data.append({
                    "input_ids":      enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                })

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]


# ── Seq2seq dataset ──────────────────────────────────────────────────────────

def _probe_to_input_target(item: Dict[str, Any]) -> Optional[tuple]:
    """Format a probe dict into (input_text, target_text) for FLAN-T5.

    Returns None if the probe is missing question or answer.
    """
    question = (item.get("question", "") or "").strip()
    evidence = (item.get("evidence", "") or "").strip()

    # Resolve gold answer
    gold = (item.get("answer", "") or "").strip()
    if not gold:
        key     = str(item.get("answer_key", ""))
        choices = item.get("choices", {})
        if isinstance(choices, dict):
            gold = (choices.get(key, "") or "").strip()
        elif isinstance(choices, list) and key.isdigit():
            idx  = int(key)
            gold = choices[idx].strip() if idx < len(choices) else ""

    if not question or not gold:
        return None

    if evidence:
        input_text = f"question: {question} context: {evidence[:400]}"
    else:
        input_text = f"question: {question}"

    return input_text, gold


class Seq2SeqDataset(Dataset):
    """Pre-tokenises QA probe items for encoder-decoder (T5-family) training.

    Each item is formatted as:
      encoder input : "question: {q} context: {evidence}"
      decoder labels: "{answer}"

    Padding positions in labels are replaced with -100 so they are ignored
    by the cross-entropy loss.
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        tokenizer,
        max_input_len: int = 256,
        max_answer_len: int = 32,
        _batch: int = 512,
    ):
        pairs = [_probe_to_input_target(it) for it in items]
        pairs = [p for p in pairs if p is not None]

        self._data: List[Dict[str, torch.Tensor]] = []
        for start in range(0, len(pairs), _batch):
            chunk = pairs[start: start + _batch]
            inputs, targets = zip(*chunk)

            enc = tokenizer(
                list(inputs),
                truncation=True,
                max_length=max_input_len,
                padding="max_length",
                return_tensors="pt",
            )
            dec = tokenizer(
                list(targets),
                truncation=True,
                max_length=max_answer_len,
                padding="max_length",
                return_tensors="pt",
            )
            # Replace pad token in labels with -100 (ignored by CE loss)
            label_ids = dec["input_ids"].clone()
            label_ids[label_ids == tokenizer.pad_token_id] = -100

            for i in range(len(chunk)):
                self._data.append({
                    "input_ids":      enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "labels":         label_ids[i],
                })

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]


# ── Loader factory ───────────────────────────────────────────────────────────

def make_loader(
    items: List[Dict[str, Any]],
    tokenizer,
    batch_size: int,
    max_seq_len: int,
    shuffle: bool = True,
    model_type: str = "seq2seq",
    max_answer_len: int = 32,
) -> DataLoader:
    """Build a DataLoader for either seq2seq or causal-LM training.

    - ``model_type == "seq2seq"`` → :class:`Seq2SeqDataset` (probe QA pairs)
    - ``model_type == "causal"``  → :class:`TextDataset` (raw article text)
    """
    if model_type == "seq2seq":
        ds = Seq2SeqDataset(items, tokenizer,
                            max_input_len=max_seq_len,
                            max_answer_len=max_answer_len)
    else:
        ds = TextDataset(items, tokenizer, max_len=max_seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


# ── Period handle ────────────────────────────────────────────────────────────

@dataclass
class Period:
    """A period materialised for the trainer (train / eval split applied)."""
    label: str
    index: int
    train_items: List[Dict[str, Any]]   # QA probes (seq2seq) or text docs (causal)
    eval_items: List[Dict[str, Any]]    # QA probes (seq2seq) or text docs (causal)
    probes: List[Dict[str, Any]]        # all probes for BWT re-evaluation


def load_periods(
    data_root: str,
    max_periods: int,
    ppl_eval_frac: float = 0.20,
    seed: int = 42,
    max_docs_per_period: Optional[int] = None,
    model_type: str = "seq2seq",
    max_train_probes: int = 0,
) -> List[Period]:
    """Load every period and split train/eval.

    - seq2seq: probes are split 80/20 (train/eval). Stream docs kept as
      background context but not used directly for training.
    - causal:  stream docs are split 80/20.
    In both modes, ``probes`` field holds all probes for BWT re-evaluation.
    """
    rng = random.Random(seed)
    raw = load_dataset(
        data_root=data_root,
        max_periods=max_periods,
        max_docs_per_period=max_docs_per_period,
    )
    out: List[Period] = []
    for pidx, pd in enumerate(raw):
        all_probes = list(pd.probes)
        rng.shuffle(all_probes)

        if model_type == "seq2seq":
            # Use QA probes as training items
            items = all_probes
            if not items:
                # Fall back to stream docs formatted as text if no probes
                items = [{"question": it.get("text", "")[:200], "answer": ""}
                         for it in pd.docs]
            # Cap training probes if requested (shuffle already applied above)
            if max_train_probes and len(items) > max_train_probes:
                items = items[:max_train_probes]
        else:
            # Use raw article text
            items = list(pd.docs)
            rng.shuffle(items)

        split = max(1, int(len(items) * (1 - ppl_eval_frac)))
        out.append(Period(
            label=pd.label,
            index=pidx,
            train_items=items[:split],
            eval_items=items[split:],
            probes=all_probes,
        ))
    return out


# ── Logger ───────────────────────────────────────────────────────────────────

class RunLogger:
    """Owns a baseline's output directory. Writes five artefacts:

    * ``training.log``        — human-readable line-per-event log
    * ``metrics.json``        — list-of-records, one per period
    * ``config.snapshot.json``— exact config at start of run
    * ``probes_period<N>.csv``— per-probe predictions (EM/F1 or MCQ)
    * ``summary.json``        — final BWT / ACC / FWT summary
    """

    def __init__(self, out_dir: Path, baseline_id: str, cfg_snapshot: Dict[str, Any]):
        self.out_dir = Path(out_dir)

        # ── Wipe any previous run in this folder so files never accumulate ──
        if self.out_dir.exists():
            import shutil
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_id  = baseline_id
        self.log_path     = self.out_dir / "training.log"
        self.metrics_path = self.out_dir / "metrics.json"
        self.config_path  = self.out_dir / "config.snapshot.json"
        self._records: List[Dict[str, Any]] = []
        self.log_path.write_text("")          # create empty log file
        self.config_path.write_text(json.dumps(cfg_snapshot, indent=2))

    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def save_record(self, record: Dict[str, Any]) -> None:
        self._records.append(record)
        self.metrics_path.write_text(json.dumps(self._records, indent=2))

    def log_loss(self, period: str, epoch: int, step: int, loss: float) -> None:
        """Append one row to ``loss_curve.csv`` (created on first call)."""
        import csv
        path = self.out_dir / "loss_curve.csv"
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["period", "epoch", "opt_step", "loss", "timestamp"])
            w.writerow([period, epoch, step, f"{loss:.6f}",
                        datetime.now().isoformat(timespec="seconds")])

    def finalize(self, summary: Dict[str, Any]) -> None:
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.log(f"Run finalised. Records: {len(self._records)}. Output: {self.out_dir}")


# ── Param counting ───────────────────────────────────────────────────────────

def param_counts(model) -> Dict[str, int]:
    """Return total + trainable parameter counts for a model."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}


# ── BWT matrix helper ────────────────────────────────────────────────────────

def evaluate_past_periods(
    model,
    tokenizer,
    device: str,
    periods_so_far: List[Period],
    cfg,
) -> Dict[str, Dict[str, float]]:
    """Re-evaluate the model on every past period; returns BWT row.

    Routes automatically based on ``cfg.model_type``:
    - seq2seq → EM + F1 via generate(); combined = seq2seq_combined_score
    - causal  → PPL + MCQ probe_acc; combined = combined_score

    Keys are period labels. Each value is a metric dict.
    """
    row: Dict[str, Dict[str, float]] = {}

    for period in periods_so_far:
        if cfg.model_type == "seq2seq":
            em, f1, _ = seq2seq_probe_accuracy(
                model, tokenizer, device,
                period.probes,
                max_n=cfg.probe_max,
                max_new_tokens=cfg.max_new_tokens,
                max_input_len=cfg.max_seq_len,
            )
            ppl = seq2seq_perplexity(
                model, period.probes, tokenizer, device,
                max_n=cfg.ppl_eval_samples,
                max_input_len=cfg.max_seq_len,
                max_answer_len=cfg.max_answer_len,
            )
            comb = seq2seq_combined_score(em, f1, cfg.em_weight, cfg.f1_weight)
            row[period.label] = {"em": em, "f1": f1, "ppl": ppl, "combined": comb}
        else:
            ppl = perplexity(
                model, period.eval_items, tokenizer, device,
                max_seq_len=cfg.max_seq_len, max_n=cfg.ppl_eval_samples,
            )
            probe_acc, _ = probe_accuracy(
                model, tokenizer, device,
                period.probes, max_n=cfg.probe_max,
            )
            comb = combined_score(ppl, probe_acc,
                                  ppl_weight=cfg.ppl_weight,
                                  probe_weight=cfg.probe_weight,
                                  ppl_decay=cfg.ppl_decay)
            row[period.label] = {
                "ppl":      ppl,
                "probe_acc": probe_acc,
                "combined": comb,
            }

    return row

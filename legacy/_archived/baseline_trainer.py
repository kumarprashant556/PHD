"""Baseline seq2seq trainer — vanilla FLAN-T5 fine-tuning without INCA growth.

Trains a standard T5ForConditionalGeneration on the same period datasets used
by the INCA trainer, providing the CL-baseline comparison for the CAPSEL paper.

Two training modes
------------------
joint
    All periods concatenated into one pass — upper bound on performance, but
    no continual-learning challenge.
sequential
    One period at a time (no replay, no growth) — lower bound; shows forgetting.

Usage (from repo root)
----------------------
    python scripts/train_baseline.py --config configs/inca.yaml --mode sequential
    python scripts/train_baseline.py --config configs/inca.yaml --mode joint
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.inca.config import INCAConfig, BaseConfig   # re-use config fields
import data as data_module
from data.tokenizer import build_tokenized_periods, make_dataloader


# ──────────────────────────────────────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────────────────────────────────────

class _RunLogger:
    def __init__(self, out_dir: Path, cfg_snapshot: dict) -> None:
        self._path = out_dir / "run_log.jsonl"
        self._append({"event": "config", "cfg": cfg_snapshot})

    def log(self, msg: str) -> None:
        ts = datetime.now().isoformat(timespec="seconds")
        print(f"[{ts}] {msg}", flush=True)
        self._append({"event": "log", "msg": msg, "ts": ts})

    def _append(self, record: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class _LossLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["period", "epoch", "opt_step", "loss", "timestamp"]
            )

    def log(self, period: str, epoch: int, step: int, loss: float) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [period, epoch, step, f"{loss:.6f}",
                 datetime.now().isoformat(timespec="seconds")]
            )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _batch_to_device(
    batch: Dict[str, torch.Tensor], device: str
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


@torch.no_grad()
def _eval_accuracy(
    model: T5ForConditionalGeneration,
    eval_ds: Dataset,
    tokenizer,
    device: str,
    batch_size: int = 32,
    max_input_length: int = 256,
    max_new_tokens: int = 64,
    n_samples: int = 500,
) -> float:
    """Greedy-decode exact-match accuracy on *eval_ds* (columns: input_text, target_text)."""
    if len(eval_ds) == 0:
        return 0.0

    indices = list(range(len(eval_ds)))
    if len(indices) > n_samples:
        indices = random.sample(indices, n_samples)
    subset = eval_ds.select(indices)

    model.eval()
    correct = total = 0

    for start in range(0, len(subset), batch_size):
        chunk  = subset.select(range(start, min(start + batch_size, len(subset))))
        inputs = tokenizer(
            chunk["input_text"],
            truncation=True, max_length=max_input_length,
            padding=True, return_tensors="pt",
        )
        inputs = _batch_to_device(inputs, device)

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for pred, gold in zip(preds, chunk["target_text"]):
            total += 1
            if pred.strip().lower() == gold.strip().lower():
                correct += 1

    return correct / max(1, total)


def _one_epoch(
    model: T5ForConditionalGeneration,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    accum: int,
    logger: _RunLogger,
    loss_log: _LossLog,
    period_label: str,
    epoch: int,
    n_epochs: int,
    global_step_ref: List[int],   # mutable single-element list
    log_every: int = 50,
) -> float:
    """Run one epoch; return average loss."""
    from tqdm.auto import tqdm

    model.train()
    total_loss = 0.0
    n_steps    = 0
    acc_loss   = torch.tensor(0.0, device=device)

    batch_bar = tqdm(
        loader,
        desc=f"  ep {epoch+1}/{n_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for micro_step, batch in enumerate(batch_bar):
        batch = _batch_to_device(batch, device)
        out   = model(**batch)
        (out.loss / accum).backward()
        acc_loss = acc_loss + out.loss.detach()

        if (micro_step + 1) % accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step_ref[0] += 1
            step_loss = (acc_loss / accum).item()
            acc_loss  = torch.tensor(0.0, device=device)
            total_loss += step_loss
            n_steps    += 1

            batch_bar.set_postfix(
                loss=f"{step_loss:.4f}",
                avg=f"{total_loss/n_steps:.4f}",
                step=global_step_ref[0],
            )

            if global_step_ref[0] % log_every == 0:
                loss_log.log(period_label, epoch, global_step_ref[0], step_loss)

    return total_loss / max(1, n_steps)


# ──────────────────────────────────────────────────────────────────────────────
# Training functions
# ──────────────────────────────────────────────────────────────────────────────

def train_sequential(cfg: INCAConfig, device: str) -> None:
    """Train one period at a time — shows catastrophic forgetting baseline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(cfg.out_dir) / f"baseline_seq_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger   = _RunLogger(out_dir, dataclasses.asdict(cfg))
    loss_log = _LossLog(out_dir / "loss_curve.csv")

    seed = getattr(cfg, "seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    # ── data ──────────────────────────────────────────────────────────
    dataset_name = getattr(cfg, "dataset", "cc_news")
    n_per_period = getattr(cfg, "n_per_period", 20_000)
    max_periods  = getattr(cfg, "max_periods",  None)

    raw_periods: Dict[str, Dataset] = data_module.load_periods(
        dataset_name, n_per_period=n_per_period, seed=seed,
    )
    period_ids = list(raw_periods.keys())
    if max_periods:
        period_ids = period_ids[:max_periods]
        raw_periods = {k: raw_periods[k] for k in period_ids}

    tokenizer  = AutoTokenizer.from_pretrained(cfg.model_name)

    tok_periods = build_tokenized_periods(
        raw_periods,
        tokenizer_name=cfg.model_name,
        max_input_length=cfg.max_input_length,
        max_target_length=getattr(cfg, "max_target_length", cfg.max_input_length),
        cache_dir="cache/tokenized",
    )
    model      = T5ForConditionalGeneration.from_pretrained(cfg.model_name).to(device)
    if getattr(cfg, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    logger.log(f"Baseline-sequential | {len(period_ids)} periods | device={device}")

    accum           = max(1, getattr(cfg, "grad_accum_steps", 1))
    batches_per_ep  = max(1, n_per_period // (cfg.batch_size * accum))
    total_opt_steps = batches_per_ep * cfg.epochs_per_period * len(period_ids)
    warmup_steps    = max(1, int(total_opt_steps * getattr(cfg, "warmup_ratio", 0.06)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, max(warmup_steps + 1, total_opt_steps),
    )

    global_step = [0]   # mutable ref

    for period_idx, period_id in enumerate(period_ids):
        raw_ds = raw_periods[period_id]
        tok_ds = tok_periods[period_id]
        tok_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        eval_frac = getattr(cfg, "ppl_eval_frac", 0.05)
        n_eval    = max(64, int(len(raw_ds) * eval_frac))
        raw_split = raw_ds.train_test_split(test_size=n_eval, seed=seed)
        tok_split = tok_ds.train_test_split(test_size=n_eval, seed=seed)
        eval_raw  = raw_split["test"]
        train_tok = tok_split["train"]
        train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        loader = make_dataloader(train_tok, batch_size=cfg.batch_size, num_workers=0)

        logger.log(
            f"\nPeriod {period_idx+1}/{len(period_ids)}: {period_id}  "
            f"(train={len(train_tok)}, eval={len(eval_raw)})"
        )

        from tqdm.auto import tqdm as _tqdm
        epoch_bar = _tqdm(
            range(cfg.epochs_per_period),
            desc=f"  Period {period_idx+1}/{len(period_ids)} [{period_id}]",
            leave=True,
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            avg_loss = _one_epoch(
                model, loader, optimizer, scheduler, device, accum,
                logger, loss_log, period_id, epoch, cfg.epochs_per_period,
                global_step,
            )
            epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")
            logger.log(f"  Epoch {epoch+1}/{cfg.epochs_per_period} loss={avg_loss:.4f}")

        acc = _eval_accuracy(
            model, eval_raw, tokenizer, device,
            batch_size=cfg.batch_size, max_input_length=cfg.max_input_length,
        )
        logger.log(f"  Post-period accuracy: {acc:.4f}")

        torch.save(
            model.state_dict(),
            out_dir / f"baseline_period_{period_id}.pt",
        )

    logger.log(f"\nSequential baseline complete → {out_dir}")


def train_joint(cfg: INCAConfig, device: str) -> None:
    """Train on all periods jointly (upper-bound baseline)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(cfg.out_dir) / f"baseline_joint_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger   = _RunLogger(out_dir, dataclasses.asdict(cfg))
    loss_log = _LossLog(out_dir / "loss_curve.csv")

    seed = getattr(cfg, "seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    dataset_name = getattr(cfg, "dataset", "cc_news")
    n_per_period = getattr(cfg, "n_per_period", 20_000)
    max_periods  = getattr(cfg, "max_periods",  None)

    tokenizer  = AutoTokenizer.from_pretrained(cfg.model_name)
    raw_periods = data_module.load_periods(
        dataset_name, n_per_period=n_per_period, seed=seed,
    )
    period_ids = list(raw_periods.keys())
    if max_periods:
        period_ids = period_ids[:max_periods]
        raw_periods = {k: raw_periods[k] for k in period_ids}

    tok_periods = build_tokenized_periods(
        raw_periods, tokenizer_name=cfg.model_name,
        max_input_length=cfg.max_input_length,
        max_target_length=getattr(cfg, "max_target_length", cfg.max_input_length),
    )

    # Concatenate all periods
    all_tok = concatenate_datasets(list(tok_periods.values()))
    all_raw = concatenate_datasets(list(raw_periods.values()))

    eval_frac = getattr(cfg, "ppl_eval_frac", 0.05)
    n_eval    = max(128, int(len(all_raw) * eval_frac))
    raw_split = all_raw.train_test_split(test_size=n_eval, seed=seed)
    tok_split = all_tok.train_test_split(test_size=n_eval, seed=seed)
    eval_raw  = raw_split["test"]
    train_tok = tok_split["train"]
    train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = T5ForConditionalGeneration.from_pretrained(cfg.model_name).to(device)
    loader = make_dataloader(train_tok, batch_size=cfg.batch_size, shuffle=True)

    accum           = max(1, getattr(cfg, "grad_accum_steps", 1))
    batches_per_ep  = max(1, len(train_tok) // (cfg.batch_size * accum))
    total_opt_steps = batches_per_ep * cfg.epochs_per_period
    warmup_steps    = max(1, int(total_opt_steps * getattr(cfg, "warmup_ratio", 0.06)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, max(warmup_steps + 1, total_opt_steps),
    )

    logger.log(
        f"Baseline-joint | {len(period_ids)} periods merged | "
        f"train={len(train_tok)} | eval={len(eval_raw)} | device={device}"
    )

    global_step = [0]

    from tqdm.auto import tqdm as _tqdm
    epoch_bar = _tqdm(
        range(cfg.epochs_per_period),
        desc="  Joint training",
        leave=True,
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        avg_loss = _one_epoch(
            model, loader, optimizer, scheduler, device, accum,
            logger, loss_log, "all", epoch, cfg.epochs_per_period, global_step,
        )
        acc = _eval_accuracy(
            model, eval_raw, tokenizer, device,
            batch_size=cfg.batch_size, max_input_length=cfg.max_input_length,
        )
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")
        logger.log(f"Epoch {epoch+1}/{cfg.epochs_per_period}: loss={avg_loss:.4f}  acc={acc:.4f}")

    final_ckpt = out_dir / "baseline_joint_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    logger.log(f"Joint baseline complete → {final_ckpt}")


def train(cfg: INCAConfig, device: str, mode: str = "sequential") -> None:
    """Dispatch to the correct baseline mode."""
    if mode == "joint":
        train_joint(cfg, device)
    else:
        train_sequential(cfg, device)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Train FLAN-T5 baseline")
    p.add_argument("--config",  required=True)
    p.add_argument("--mode",    default="sequential",
                   choices=["sequential", "joint"],
                   help="Training mode: sequential (default) or joint")
    p.add_argument("--dataset", default=None, help="Override cfg.dataset")
    p.add_argument("--seed",    type=int, default=None)
    p.add_argument("--device",  default=None)
    args = p.parse_args()

    import yaml
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}

    if args.dataset: cfg_dict["dataset"] = args.dataset
    if args.seed:    cfg_dict["seed"]    = args.seed

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

    print(f"Baseline  |  mode={args.mode}  |  model={cfg.model_name}  "
          f"|  dataset={getattr(cfg, 'dataset', 'cc_news')}  |  device={device}")

    train(cfg, device, mode=args.mode)


if __name__ == "__main__":
    main()

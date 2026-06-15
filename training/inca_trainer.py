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

import argparse
import csv
import dataclasses
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
from datasets import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
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
from data.tokenizer import build_tokenized_periods, make_dataloader, make_replay_dataloader


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


# ──────────────────────────────────────────────────────────────────────────────
# Batch utilities
# ──────────────────────────────────────────────────────────────────────────────

def _batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


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
# Evaluation — greedy decode + exact-match over raw (un-tokenized) Dataset
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_accuracy(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    eval_ds: Dataset,
    tokenizer,
    device: str,
    batch_size: int = 32,
    max_input_length: int = 256,
    max_new_tokens: int = 64,
    n_samples: int = 500,
) -> float:
    """Sample *n_samples* rows from *eval_ds* and compute exact-match accuracy.

    *eval_ds* must have columns: input_text (str), target_text (str).
    """
    if len(eval_ds) == 0:
        return 0.0

    indices = list(range(len(eval_ds)))
    if len(indices) > n_samples:
        indices = random.sample(indices, n_samples)
    subset = eval_ds.select(indices)

    model.eval()
    manager.eval()
    correct = 0
    total   = 0

    for start in range(0, len(subset), batch_size):
        chunk  = subset.select(range(start, min(start + batch_size, len(subset))))
        inputs = tokenizer(
            chunk["input_text"],
            truncation=True,
            max_length=max_input_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = _batch_to_device(inputs, device)

        enc_hidden = manager(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        enc_out = BaseModelOutput(last_hidden_state=enc_hidden)

        gen_ids = model.generate(
            encoder_outputs=enc_out,
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        golds = chunk["target_text"]

        for pred, gold in zip(preds, golds):
            total += 1
            if pred.strip().lower() == gold.strip().lower():
                correct += 1

    return correct / max(1, total)


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
    """Adapt raw Dataset rows for CKAMonitor (which expects dicts with 'question')."""
    indices = list(range(len(raw_ds)))
    if len(indices) > n_samples:
        indices = random.sample(indices, n_samples)
    subset = raw_ds.select(indices)
    items = [{"question": row["input_text"]} for row in subset]
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
    """Tokenize raw replay dicts (input_text/target_text) into tensor dicts."""
    if not raw_items:
        return []
    pad_id = tokenizer.pad_token_id
    enc = tokenizer(
        [it["input_text"] for it in raw_items],
        truncation=True, max_length=max_input_length,
        padding=True, return_tensors="pt",
    )
    dec = tokenizer(
        text_target=[it["target_text"] for it in raw_items],
        truncation=True, max_length=max_target_length,
        padding=True, return_tensors="pt",
    )
    labels = dec["input_ids"].clone()
    labels[labels == pad_id] = -100
    # Return as list of per-item dicts so _ReplayMixDataset can shuffle them
    result = []
    for i in range(len(raw_items)):
        result.append({
            "input_ids":      enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels":         labels[i],
        })
    return result


def _check_replay_drift(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    replay_buf: INCAReplayBuffer,
    prev_acc: float,
    tokenizer,
    device: str,
    tol: float,
    batch_size: int = 32,
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
    )
    return (prev_acc - cur_acc) > tol


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

def train(cfg: INCAConfig, device: str) -> None:
    """Full INCA-v2 training loop.

    Parameters
    ----------
    cfg    : validated INCAConfig dataclass
    device : "cuda" | "mps" | "cpu"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) / f"inca_v2_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger   = _RunLogger(out_dir, dataclasses.asdict(cfg))
    loss_log = _LossLog(out_dir / "loss_curve.csv")

    # ── reproducibility ────────────────────────────────────────────────
    seed = getattr(cfg, "seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

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

    # ── tokenizer + base model ─────────────────────────────────────────
    logger.log(f"Loading model: {cfg.model_name}")
    tokenizer  = AutoTokenizer.from_pretrained(cfg.model_name)
    base_model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)

    if getattr(cfg, "gradient_checkpointing", False):
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()

    base_model.to(device)

    # ── tokenize all periods up-front ──────────────────────────────────
    logger.log("Tokenizing periods …")
    tok_periods: Dict[str, Dataset] = build_tokenized_periods(
        raw_periods,
        tokenizer=tokenizer,
        max_input_length=cfg.max_input_length,
        max_target_length=getattr(cfg, "max_target_length", cfg.max_input_length),
    )

    # ── INCA manager ───────────────────────────────────────────────────
    manager = INCALayerManager(base_model, cfg).to(device)

    # ── LR schedule parameters (estimated over all periods) ───────────
    accum           = max(1, getattr(cfg, "grad_accum_steps", 1))
    batches_per_ep  = max(1, n_per_period // (cfg.batch_size * accum))
    total_opt_steps = batches_per_ep * cfg.epochs_per_period * len(period_ids)
    warmup_steps    = max(1, int(total_opt_steps * getattr(cfg, "warmup_ratio", 0.06)))

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

    global_opt_step = 0
    block_idx       = 0
    prev_replay_acc: float = 1.0

    # ══════════════════════════════════════════════════════════════════
    # Period loop
    # ══════════════════════════════════════════════════════════════════
    for period_idx, period_id in enumerate(period_ids):
        raw_ds = raw_periods[period_id]
        tok_ds = tok_periods[period_id]
        tok_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # ── train / eval split ─────────────────────────────────────────
        eval_frac = getattr(cfg, "ppl_eval_frac", 0.05)
        n_eval    = max(64, int(len(raw_ds) * eval_frac))

        raw_split = raw_ds.train_test_split(test_size=n_eval, seed=seed)
        eval_raw  = raw_split["test"]   # raw rows for greedy-decode eval

        tok_split = tok_ds.train_test_split(test_size=n_eval, seed=seed)
        train_tok = tok_split["train"]
        train_tok.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        logger.log(
            f"\n{'='*60}\n"
            f"Period {period_idx+1}/{len(period_ids)}: {period_id}  "
            f"(train={len(train_tok)}, eval={len(eval_raw)})\n"
            f"Block chain: {manager.n_blocks} block(s)"
        )

        # ── CKA reference at period start ──────────────────────────────
        _cache_cka_reference(
            cka_monitor, manager, raw_ds, tokenizer, device,
            max_seq_len=cfg.max_input_length,
            n_samples=cfg.cka_ref_size,
        )

        # ── snapshot replay accuracy before this period ───────────────
        had_replay_before = len(replay_buf.periods) > 0
        replay_acc_before: float = prev_replay_acc

        # ── build DataLoader (replay-mixed if buffer non-empty) ────────
        replay_n = getattr(cfg, "replay_n_per_period", 2_000)
        raw_replay = replay_buf.sample(n=replay_n, epoch=0) if had_replay_before else []
        replay_items = _tokenize_replay_items(
            raw_replay, tokenizer,
            max_input_length=cfg.max_input_length,
            max_target_length=getattr(cfg, "max_target_length", cfg.max_input_length),
        )

        if replay_items:
            train_loader = make_replay_dataloader(
                stream_dataset=train_tok,
                replay_items=replay_items,
                batch_size=cfg.batch_size,
                replay_ratio=getattr(cfg, "replay_ratio", 0.25),
                seed=seed,
            )
        else:
            train_loader = make_dataloader(train_tok, batch_size=cfg.batch_size)

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

            for micro_step, batch in enumerate(train_loader):
                if period_done:
                    break

                batch = _batch_to_device(batch, device)
                loss  = _forward_loss(base_model, manager, batch)
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

                    log_every = getattr(cfg, "log_every_n_steps", 50)
                    if global_opt_step % log_every == 0:
                        loss_log.log(
                            period_id, block_idx, epoch,
                            global_opt_step, step_loss,
                        )

                    # ── k_eval saturation check ────────────────────────
                    if global_opt_step % cfg.k_eval == 0:
                        cur_score = _eval_accuracy(
                            base_model, manager, eval_raw, tokenizer, device,
                            batch_size=cfg.batch_size,
                            max_input_length=cfg.max_input_length,
                        )
                        cka_val  = cka_monitor.compute(manager, device)
                        avg_loss = (
                            sum(micro_losses[-cfg.k_eval:])
                            / max(1, min(len(micro_losses), cfg.k_eval))
                        )

                        detector.update(avg_loss, cur_score, last_grad_norm, cka_value=cka_val)
                        event = detector.check(epoch)

                        logger.log(
                            f"  [e{epoch} s{global_opt_step}] "
                            f"score={cur_score:.4f} cka={cka_val:.3f} "
                            f"gnorm={last_grad_norm:.4f} loss={avg_loss:.4f} "
                            f"→ {event.name}"
                        )

                        if event == SaturationEvent.PERIOD_LEARNED:
                            logger.log("  ✓ PERIOD_LEARNED — advancing to next period.")
                            period_done = True
                            break

                        elif event == SaturationEvent.BLOCK_FULL:
                            logger.log("  ✓ BLOCK_FULL — freezing and growing new block.")
                            optimizer, scheduler = _grow_block(
                                manager, cfg, device, warmup_steps, total_opt_steps,
                            )
                            detector.reset_block()
                            cka_monitor.reset()
                            block_idx += 1
                            period_done = True   # restart period after grow
                            break

                        else:
                            timeout_counter += 1
                            max_evals = cfg.patience * 3
                            if timeout_counter >= max_evals:
                                fallback = detector.check_timeout()
                                logger.log(
                                    f"  Timeout after {timeout_counter} evals "
                                    f"→ {fallback.name}"
                                )
                                if fallback == SaturationEvent.PERIOD_LEARNED:
                                    period_done = True
                                    break
                                else:
                                    logger.log("  EXHAUSTED — freeze-and-grow.")
                                    optimizer, scheduler = _grow_block(
                                        manager, cfg, device,
                                        warmup_steps, total_opt_steps,
                                    )
                                    detector.reset_block()
                                    cka_monitor.reset()
                                    block_idx += 1
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
                batch_size=cfg.batch_size,
            )
            if drift:
                logger.log("  [T1.3] Replay drift > tol — early BLOCK_FULL.")
                try:
                    optimizer, scheduler = _grow_block(
                        manager, cfg, device, warmup_steps, total_opt_steps,
                    )
                    detector.reset_block()
                    cka_monitor.reset()
                    block_idx += 1
                except RuntimeError as exc:
                    logger.log(f"  [T1.3] grow skipped: {exc}")

        # ── post-period eval ───────────────────────────────────────────
        post_score = _eval_accuracy(
            base_model, manager, eval_raw, tokenizer, device,
            batch_size=cfg.batch_size,
            max_input_length=cfg.max_input_length,
        )
        logger.log(f"  Post-period accuracy: {post_score:.4f}")
        prev_replay_acc = post_score

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


# ──────────────────────────────────────────────────────────────────────────────
# CLI (called via scripts/train_inca.py)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train INCA-v2")
    p.add_argument("--config",   required=True, help="Path to YAML config")
    p.add_argument("--dataset",  default=None,  help="Override cfg.dataset")
    p.add_argument("--selector", default=None,  help="Override cfg.selector")
    p.add_argument("--seed",     type=int, default=None, help="Override cfg.seed")
    p.add_argument("--device",   default=None,  help="cpu | mps | cuda")
    p.add_argument("--dry-run",  action="store_true",
                   help="Validate config + build model — don't train")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    import yaml
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f) or {}

    if args.dataset:  cfg_dict["dataset"]  = args.dataset
    if args.selector: cfg_dict["selector"] = args.selector
    if args.seed:     cfg_dict["seed"]     = args.seed

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

    train(cfg, device)


if __name__ == "__main__":
    main()

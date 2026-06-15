"""INCA-v2 training loop  (Phase 1).

Entry point for the growing-block INCA trainer.  Orchestrates:

  * INCALayerManager          — block-chain growth / freeze
  * INCAPlateauDetector       — multi-signal consensus saturation (T1.1)
  * CKAMonitor                — representational-stability signal (T1.5)
  * INCAReplayBuffer          — study-schedule replay (T1.4)
  * T1.2 early-stop relabelling — timeout → PERIOD_LEARNED or BLOCK_FULL
  * T1.3 replay drift check   — early BLOCK_FULL if past-period accuracy drops

Usage
-----
    python -m Phase1.src.train_inca_v3 \
        --config Phase1/configs/inca_v2_smoke.yaml \
        [--device mps|cpu|cuda] [--seed 42]

The output directory is <cfg.out_dir>/inca_v2_<timestamp>/.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, AutoTokenizer
from transformers.optimization import Adafactor, get_cosine_schedule_with_warmup

# ── sibling imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import dataclasses

from Phase0.common.harness import RunLogger, load_periods   # both live in harness

from .inca_config import INCAConfig
from .inca_layer_manager import INCALayerManager
from .inca_plateau import INCAPlateauDetector, SaturationEvent
from .inca_cka import CKAMonitor
from .inca_replay import INCAReplayBuffer


# ────────────────────────────────────────────────────────────────────────────────
# Helper: build T5 seq2seq batch
# ────────────────────────────────────────────────────────────────────────────────

def _make_batch(
    items: List[dict],
    tokenizer,
    device: str,
    max_src: int = 256,
    max_tgt: int = 32,
) -> Dict[str, torch.Tensor]:
    """Tokenize a list of probe dicts into a model-ready batch dict."""
    src_texts = [
        f"question: {it.get('question','').strip()} "
        f"context: {(it.get('evidence') or '')[:400].strip()}"
        for it in items
    ]
    tgt_texts = [str(it.get("answer", "")).strip() for it in items]

    enc = tokenizer(
        src_texts, truncation=True, max_length=max_src,
        padding=True, return_tensors="pt",
    )
    dec = tokenizer(
        tgt_texts, truncation=True, max_length=max_tgt,
        padding=True, return_tensors="pt",
    )
    labels = dec["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids":      enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "labels":         labels.to(device),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Helper: compute accuracy on a list of items (greedy decode)
# ────────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_accuracy(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    items: List[dict],
    tokenizer,
    device: str,
    batch_size: int = 32,
    max_src: int = 256,
    max_new: int = 32,
) -> float:
    """Greedy-decode and compute exact-match accuracy over *items*."""
    if not items:
        return 0.0
    model.eval()
    manager.eval()
    correct = 0
    for start in range(0, len(items), batch_size):
        chunk = items[start: start + batch_size]
        batch = _make_batch(chunk, tokenizer, device, max_src=max_src, max_tgt=max_new)

        # Override encoder_outputs with INCA manager
        encoder_hidden = manager(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        from transformers.modeling_outputs import BaseModelOutput
        enc_out = BaseModelOutput(last_hidden_state=encoder_hidden)

        gen_ids = model.generate(
            encoder_outputs=enc_out,
            attention_mask=batch["attention_mask"],
            max_new_tokens=max_new,
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        golds = [str(it.get("answer", "")).strip() for it in chunk]
        correct += sum(p.strip().lower() == g.strip().lower() for p, g in zip(preds, golds))

    return correct / len(items)


# ────────────────────────────────────────────────────────────────────────────────
# Helper: single forward pass returning scalar loss
# ────────────────────────────────────────────────────────────────────────────────

def _forward_loss(
    model: T5ForConditionalGeneration,
    manager: INCALayerManager,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run INCA manager encoder → T5 decoder → cross-entropy loss."""
    encoder_hidden = manager(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    from transformers.modeling_outputs import BaseModelOutput
    enc_out = BaseModelOutput(last_hidden_state=encoder_hidden)

    out = model(
        encoder_outputs=enc_out,
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    return out.loss


# ────────────────────────────────────────────────────────────────────────────────
# Loss-curve CSV writer
# ────────────────────────────────────────────────────────────────────────────────

class _LossLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._write_header()

    def _write_header(self) -> None:
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["period", "block", "epoch", "opt_step", "loss", "timestamp"]
            )

    def log(self, period: str, block: int, epoch: int, step: int, loss: float) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [period, block, epoch, step, f"{loss:.6f}",
                 datetime.now().isoformat(timespec="seconds")]
            )


# ────────────────────────────────────────────────────────────────────────────────
# T1.3 drift check
# ────────────────────────────────────────────────────────────────────────────────

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
    """Return True if past-period replay accuracy has dropped by > tol.

    T1.3: after advancing to the next period we re-eval the replay buffer.
    If accuracy on past material falls by more than ``tol`` we interpret it
    as forward-interference → trigger early BLOCK_FULL.
    """
    old_items = replay_buf.all_items()
    if not old_items:
        return False
    cur_acc = _eval_accuracy(
        model, manager, old_items, tokenizer, device, batch_size=batch_size
    )
    drift = prev_acc - cur_acc
    return drift > tol


# ────────────────────────────────────────────────────────────────────────────────
# Main training loop
# ────────────────────────────────────────────────────────────────────────────────

def train(cfg: INCAConfig, device: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) / f"inca_v2_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(
        out_dir,
        baseline_id="inca_v2",
        cfg_snapshot=dataclasses.asdict(cfg),
    )
    loss_log = _LossLog(out_dir / "loss_curve.csv")

    # ── seed ──────────────────────────────────────────────────────────
    seed = getattr(cfg, "seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    # ── data ──────────────────────────────────────────────────────────
    max_probes = getattr(cfg, "max_train_probes", 0)
    # data_root mirrors the pattern in Phase0/common/runner.py
    _repo_root = Path(__file__).resolve().parent.parent.parent
    data_root  = _repo_root / "Phase0" / "data" / "processed" / cfg.dataset
    periods = load_periods(
        data_root=str(data_root),
        max_periods=cfg.max_periods,
        ppl_eval_frac=cfg.ppl_eval_frac,
        seed=seed,
        max_docs_per_period=cfg.max_docs_per_period or None,
        model_type=cfg.model_type,
        max_train_probes=max_probes,
    )
    logger.log(f"Loaded {len(periods)} periods from {data_root}")

    # ── base model ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    base_model = T5ForConditionalGeneration.from_pretrained(cfg.model_name)
    if getattr(cfg, "gradient_checkpointing", False):
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()
    base_model.to(device)

    # ── INCA manager ──────────────────────────────────────────────────
    manager = INCALayerManager(base_model, cfg).to(device)

    # ── optimiser ─────────────────────────────────────────────────────
    def _make_optimizer():
        params = manager.trainable_params()
        if getattr(cfg, "use_adafactor", False):
            return Adafactor(
                params, lr=cfg.lr, relative_step=False,
                scale_parameter=False, warmup_init=False,
                weight_decay=cfg.weight_decay,
            )
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    optimizer = _make_optimizer()

    # ── LR scheduler (created over all periods × epochs) ──────────────
    accum = max(1, getattr(cfg, "grad_accum_steps", 1))
    batches_per_period = max(1, max_probes // max(1, cfg.batch_size) // accum) \
        if max_probes else 200
    total_opt_steps = batches_per_period * cfg.epochs_per_period * len(periods)
    warmup_steps = max(1, int(total_opt_steps * getattr(cfg, "warmup_ratio", 0.06)))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    # ── replay buffer ─────────────────────────────────────────────────
    replay_buf = INCAReplayBuffer(
        max_size_per_period=cfg.buffer_max_size,
        n_revise=cfg.n_revise,
        p_hard=cfg.p_hard,
        p_easy=cfg.p_easy,
        p_mid=cfg.p_mid,
    )

    # ── saturation detector + CKA monitor ─────────────────────────────
    detector = INCAPlateauDetector(cfg)
    cka_monitor = CKAMonitor(ref_size=cfg.cka_ref_size)

    global_opt_step = 0
    block_idx = 0           # for logging
    prev_replay_acc: float = 1.0   # used for T1.3 drift check

    # ══════════════════════════════════════════════════════════════════
    # Period loop
    # ══════════════════════════════════════════════════════════════════
    for period_idx, period in enumerate(periods):
        period_name = period.label
        train_items = list(period.train_items)
        eval_items  = list(period.eval_items)

        logger.log(f"\n{'='*60}")
        logger.log(f"Period {period_idx+1}/{len(periods)}: {period_name}  "
                    f"(train={len(train_items)}, eval={len(eval_items)})")
        logger.log(f"Block chain: {manager.n_blocks} block(s)")

        # ── cache CKA reference ───────────────────────────────────────
        cka_monitor.cache_reference(
            encoder=manager,
            items=train_items,
            tokenizer=tokenizer,
            device=device,
            max_seq_len=256,
        )

        # ── pre-period eval (sets RIR baseline) ───────────────────────
        pre_score = _eval_accuracy(
            base_model, manager, eval_items, tokenizer, device,
            batch_size=cfg.batch_size,
        )
        detector.reset_period(pre_score)
        logger.log(f"  Pre-period score: {pre_score:.4f}")

        # ── T1.3: replay drift baseline before period training ─────────
        # had_replay_before: True only if past-period items were already in
        # the buffer when this period started.  Period 1 always False — the
        # current period's items are added after epoch 1, so checking drift
        # against a 1.0 sentinel would always false-fire.
        had_replay_before = bool(replay_buf.all_items())
        replay_acc_before = _eval_accuracy(
            base_model, manager, replay_buf.all_items(), tokenizer, device,
            batch_size=cfg.batch_size,
        ) if had_replay_before else 1.0

        timeout_counter = 0          # counts eval steps with no decision
        period_done = False
        last_grad_n: float = 0.0     # updated each opt step before zero_grad

        # ── add this period's items to replay buffer after first pass ──
        # (items added after the first epoch so at least one forward pass
        # has occurred and losses are meaningful)
        first_epoch_done = False

        # ══════════════════════════════════════════════════════════════
        # Epoch loop
        # ══════════════════════════════════════════════════════════════
        for epoch in range(1, cfg.epochs_per_period + 1):
            if period_done:
                break

            random.shuffle(train_items)

            # Mix in replay items (replay_ratio fraction of each micro-batch)
            n_replay = int(cfg.batch_size * cfg.replay_ratio)
            n_fresh  = cfg.batch_size - n_replay

            micro_losses: List[float] = []
            accumulate_loss = torch.tensor(0.0, device=device)
            micro_step = 0  # steps within the current accumulation window

            # ── batch loop ────────────────────────────────────────────
            for start in range(0, len(train_items), max(1, n_fresh)):
                if period_done:
                    break

                fresh = train_items[start: start + max(1, n_fresh)]
                replay_items = replay_buf.sample(n_replay, epoch) if n_replay > 0 else []
                batch_items = fresh + replay_items
                if not batch_items:
                    continue

                batch = _make_batch(batch_items, tokenizer, device)
                base_model.train()
                manager.train()

                loss = _forward_loss(base_model, manager, batch) / accum
                loss.backward()
                accumulate_loss = accumulate_loss + loss.detach()
                micro_step += 1

                if micro_step % accum == 0:
                    nn.utils.clip_grad_norm_(manager.trainable_params(), 1.0)
                    # Capture grad norm BEFORE zero_grad so it is non-zero
                    last_grad_n = manager.grad_norm()
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_opt_step += 1
                    step_loss = accumulate_loss.item()
                    accumulate_loss = torch.tensor(0.0, device=device)
                    micro_losses.append(step_loss)

                    log_every = getattr(cfg, "log_every_n_steps", 50)
                    if global_opt_step % log_every == 0:
                        loss_log.log(period_name, block_idx, epoch,
                                     global_opt_step, step_loss)

                    # ── k_eval saturation check ────────────────────────
                    if global_opt_step % cfg.k_eval == 0:
                        cur_score = _eval_accuracy(
                            base_model, manager, eval_items, tokenizer, device,
                            batch_size=cfg.batch_size,
                        )
                        cka_val = cka_monitor.compute(manager, device)
                        grad_n  = last_grad_n   # captured before zero_grad
                        avg_loss = sum(micro_losses[-cfg.k_eval:]) / max(1, len(micro_losses[-cfg.k_eval:]))

                        detector.update(avg_loss, cur_score, grad_n, cka_val)
                        event = detector.check(epoch)

                        logger.log(
                            f"  [e{epoch} s{global_opt_step}] "
                            f"score={cur_score:.4f} cka={cka_val:.3f} "
                            f"gnorm={grad_n:.4f} loss={avg_loss:.4f} → {event.name}"
                        )

                        if event == SaturationEvent.PERIOD_LEARNED:
                            logger.log("  ✓ PERIOD_LEARNED — advancing period.")
                            period_done = True
                            break

                        elif event == SaturationEvent.BLOCK_FULL:
                            logger.log("  ✓ BLOCK_FULL — freezing and growing.")
                            optimizer = _grow_block(
                                manager, optimizer, scheduler,
                                cfg, warmup_steps, total_opt_steps, device,
                            )
                            detector.reset_block()
                            cka_monitor.reset()
                            block_idx += 1
                            period_done = True   # restart period after grow
                            break

                        else:
                            timeout_counter += 1
                            max_evals = cfg.patience * 3  # generous timeout
                            if timeout_counter >= max_evals:
                                fallback = detector.check_timeout()
                                logger.log(
                                    f"  Timeout after {timeout_counter} evals "
                                    f"→ {fallback.name}"
                                )
                                if fallback == SaturationEvent.PERIOD_LEARNED:
                                    period_done = True
                                    break
                                else:  # EXHAUSTED → block-full path
                                    logger.log("  EXHAUSTED → freeze-and-grow.")
                                    optimizer = _grow_block(
                                        manager, optimizer, scheduler,
                                        cfg, warmup_steps, total_opt_steps, device,
                                    )
                                    detector.reset_block()
                                    cka_monitor.reset()
                                    block_idx += 1
                                    period_done = True
                                    break

            # ── end of epoch: update replay losses ─────────────────────
            if not first_epoch_done:
                replay_buf.add_period(period_name, train_items)
                first_epoch_done = True

        # ── T1.3 replay drift check after period advance ───────────────
        # Only check drift against periods that existed BEFORE this one started.
        if had_replay_before:
            drift_triggered = _check_replay_drift(
                base_model, manager, replay_buf,
                prev_acc=replay_acc_before,
                tokenizer=tokenizer,
                device=device,
                tol=cfg.period_drift_tol,
                batch_size=cfg.batch_size,
            )
            if drift_triggered:
                logger.log(
                    "  [T1.3] Replay drift > tol — early BLOCK_FULL triggered."
                )
                try:
                    optimizer = _grow_block(
                        manager, optimizer, scheduler,
                        cfg, warmup_steps, total_opt_steps, device,
                    )
                    detector.reset_block()
                    cka_monitor.reset()
                    block_idx += 1
                except RuntimeError as exc:
                    logger.log(f"  [T1.3] grow skipped: {exc}")

        # ── post-period eval ───────────────────────────────────────────
        post_score = _eval_accuracy(
            base_model, manager, eval_items, tokenizer, device,
            batch_size=cfg.batch_size,
        )
        logger.log(f"  Post-period score: {post_score:.4f}")
        prev_replay_acc = post_score

    # ── final checkpoint ──────────────────────────────────────────────
    ckpt_path = out_dir / "inca_v2_final.pt"
    torch.save({
        "manager_state": manager.manager_state(),
        "base_model_state": base_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_opt_step": global_opt_step,
        "cfg": vars(cfg),
    }, ckpt_path)
    logger.log(f"\nTraining complete.  Checkpoint → {ckpt_path}")


# ────────────────────────────────────────────────────────────────────────────────
# Grow helper (encapsulates optimizer + scheduler rebuild)
# ────────────────────────────────────────────────────────────────────────────────

def _grow_block(
    manager: INCALayerManager,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: INCAConfig,
    warmup_steps: int,
    total_opt_steps: int,
    device: str,
) -> torch.optim.Optimizer:
    """Freeze current block, grow a new one.

    Returns a *new* optimizer bound to the new trainable param set.
    The caller must replace its reference with the returned object.
    The scheduler is left untouched — it does not hold param references.
    """
    manager.freeze_and_grow()

    params = manager.trainable_params()
    if getattr(cfg, "use_adafactor", False):
        new_opt = Adafactor(
            params, lr=cfg.lr, relative_step=False,
            scale_parameter=False, warmup_init=False,
            weight_decay=cfg.weight_decay,
        )
    else:
        new_opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    return new_opt


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train INCA-v2")
    p.add_argument("--config",  required=True, help="Path to YAML config file")
    p.add_argument("--device",  default=None,  help="cpu | mps | cuda (auto-detect if omitted)")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def _load_config(path: str) -> INCAConfig:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    return INCAConfig(**{k: v for k, v in data.items() if hasattr(INCAConfig, k)})


def main() -> None:
    args = _parse_args()

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    cfg = _load_config(args.config)
    cfg.seed = args.seed  # type: ignore[attr-defined]

    print(f"INCA-v2 | device={device} | model={cfg.model_name} | "
          f"config={args.config}")
    train(cfg, device)


if __name__ == "__main__":
    main()

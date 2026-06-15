"""Shared tqdm training loop and CAPSEL display utilities for Phase 0.

Public API
----------
standard_train_loop   — causal-LM training loop with nested tqdm bars (B1,B3,B6).
seq2seq_train_loop    — encoder-decoder training loop (FLAN-T5 family).
make_epoch_bar        — outer epoch tqdm (for baselines with custom inner loops).
make_batch_bar        — inner batch tqdm for one epoch.
display_period_header — bold banner before each period.
display_pre_metrics   — pre-training metrics line (EM/F1 or PPL/probe).
display_post_metrics  — post-training metrics with delta colouring.
display_accuracy_matrix — lower-triangular probe-acc matrix with colour coding.
display_capsel_running  — live ACC / BWT / FWT box after each period.
display_final_summary   — full CAPSEL summary + per-period table at the end.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from tqdm import tqdm as _tqdm
    _TQDM = True
except ImportError:
    _TQDM = False
    def _tqdm(it, **kw):          # type: ignore[misc]
        return it


# ── ANSI colour helpers ───────────────────────────────────────────────────────
import sys
_USE_COLOR = sys.stdout.isatty()

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _bold(t: str)   -> str: return _c(t, "1")
def _green(t: str)  -> str: return _c(t, "32")
def _red(t: str)    -> str: return _c(t, "31")
def _cyan(t: str)   -> str: return _c(t, "36")
def _yellow(t: str) -> str: return _c(t, "33")
def _dim(t: str)    -> str: return _c(t, "2")

_W = 72  # default display width


# ── Shared training loop (B1, B3, B5, B6) ────────────────────────────────────

def standard_train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    device: str,
    cfg,
    period_label: str,
    period_idx: int,
    n_periods: int,
    extra_loss_fn: Optional[Callable[[], torch.Tensor]] = None,
    scheduler=None,
    loss_logger=None,
    text_logger=None,
) -> float:
    """Causal-LM training loop with nested tqdm bars.

    Supports gradient accumulation (``cfg.grad_accum_steps``), an optional
    LR scheduler, and an optional ``loss_logger`` callback for CSV logging.
    ``text_logger(msg)`` is called at each epoch end to write a human-readable
    summary line to training.log.
    ``extra_loss_fn()`` is called each step and added to CE loss (e.g. EWC).
    Returns the final epoch's average CE loss.
    """
    if len(dataloader) == 0:
        return 0.0

    n_epochs    = cfg.epochs_per_period
    accum_steps = max(1, getattr(cfg, "grad_accum_steps", 1))
    log_every   = max(1, getattr(cfg, "log_every_n_steps", 50))

    outer = _tqdm(
        range(1, n_epochs + 1),
        desc=_bold(f"  Period {period_idx+1}/{n_periods} [{period_label}]"),
        unit="epoch", position=0, leave=True, dynamic_ncols=True,
        colour="green" if _TQDM else None,
    )
    last_loss = 0.0
    opt_step  = 0

    for epoch in outer:
        model.train()
        total, n   = 0.0, 0
        accum_loss = 0.0
        inner = _tqdm(
            dataloader,
            desc=f"    Epoch {epoch}/{n_epochs}",
            unit="batch", position=1, leave=False, dynamic_ncols=True,
        )
        for micro_step, batch in enumerate(inner, 1):
            ids  = batch["input_ids"].to(device)
            mask = batch.get("attention_mask")
            if mask is not None:
                mask = mask.to(device)
            labels = ids.clone()
            if mask is not None:
                labels[mask == 0] = -100
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            ce_loss = out.loss
            if not torch.isfinite(ce_loss):
                inner.set_postfix(loss="NaN!", avg=f"{total/max(n,1):.4f}")
                continue
            extra      = extra_loss_fn() if extra_loss_fn is not None else 0.0
            loss       = (ce_loss + extra) / accum_steps
            loss.backward()
            accum_loss += ce_loss.item()

            is_accum_step = (micro_step % accum_steps == 0) or (micro_step == len(dataloader))
            if is_accum_step:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                opt_step   += 1
                step_loss   = accum_loss / accum_steps
                accum_loss  = 0.0
                total      += step_loss
                n          += 1
                if device == "mps" and opt_step % log_every == 0:
                    torch.mps.empty_cache()
                if loss_logger is not None and opt_step % log_every == 0:
                    loss_logger(period_label, epoch, opt_step, step_loss)
                inner.set_postfix(
                    loss=f"{step_loss:.4f}",
                    avg=f"{total/n:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
        last_loss = total / max(n, 1)
        outer.set_postfix(avg_loss=f"{last_loss:.4f}")
        if loss_logger is not None:
            loss_logger(period_label, epoch, opt_step, last_loss)
        if text_logger is not None:
            lr_now = optimizer.param_groups[0]["lr"]
            text_logger(
                f"  Epoch {epoch}/{n_epochs} done | period={period_label} | "
                f"avg_loss={last_loss:.4f} | opt_steps={opt_step} | lr={lr_now:.2e}"
            )

    return last_loss


# ── Seq2seq training loop (B1-B7 in seq2seq mode) ────────────────────────────

def seq2seq_train_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    device: str,
    cfg,
    period_label: str,
    period_idx: int,
    n_periods: int,
    extra_loss_fn=None,
    scheduler=None,
    loss_logger=None,
    text_logger=None,
) -> float:
    """Encoder-decoder (T5-family) training loop with nested tqdm bars.

    Supports gradient accumulation (``cfg.grad_accum_steps``), an optional
    LR scheduler stepped once per *optimiser* step, and an optional
    ``loss_logger(period, epoch, opt_step, loss)`` callback for CSV logging.
    ``text_logger(msg)`` is called at each epoch end to write a human-readable
    summary line to training.log.

    ``extra_loss_fn()`` is called each step and added to CE loss (e.g. EWC).
    Returns the final epoch's average CE loss.
    """
    if len(dataloader) == 0:
        return 0.0

    n_epochs    = cfg.epochs_per_period
    accum_steps = max(1, getattr(cfg, "grad_accum_steps", 1))
    log_every   = max(1, getattr(cfg, "log_every_n_steps", 50))

    outer = _tqdm(
        range(1, n_epochs + 1),
        desc=_bold(f"  Period {period_idx+1}/{n_periods} [{period_label}]"),
        unit="epoch", position=0, leave=True, dynamic_ncols=True,
        colour="cyan" if _TQDM else None,
    )
    last_loss  = 0.0
    opt_step   = 0   # counts actual optimiser.step() calls

    for epoch in outer:
        model.train()
        total, n        = 0.0, 0
        accum_loss      = 0.0   # accumulates scaled loss across micro-batches
        inner = _tqdm(
            dataloader,
            desc=f"    Epoch {epoch}/{n_epochs}",
            unit="batch", position=1, leave=False, dynamic_ncols=True,
        )
        for micro_step, batch in enumerate(inner, 1):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            ce_loss = out.loss
            if not torch.isfinite(ce_loss):
                inner.set_postfix(loss="NaN!", avg=f"{total/max(n,1):.4f}")
                continue

            extra = extra_loss_fn() if extra_loss_fn is not None else 0.0
            loss  = (ce_loss + extra) / accum_steps
            loss.backward()
            accum_loss += ce_loss.item()

            # Optimiser step every accum_steps micro-batches or at end of epoch
            is_accum_step = (micro_step % accum_steps == 0) or (micro_step == len(dataloader))
            if is_accum_step:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                opt_step   += 1
                step_loss   = accum_loss / accum_steps
                accum_loss  = 0.0
                total      += step_loss
                n          += 1
                # Flush MPS command buffer periodically
                if device == "mps" and opt_step % log_every == 0:
                    torch.mps.empty_cache()
                # Loss-curve callback
                if loss_logger is not None and opt_step % log_every == 0:
                    loss_logger(period_label, epoch, opt_step, step_loss)
                inner.set_postfix(
                    loss=f"{step_loss:.4f}",
                    avg=f"{total/n:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
        last_loss = total / max(n, 1)
        outer.set_postfix(avg_loss=f"{last_loss:.4f}")
        # CSV loss log (epoch-end summary row)
        if loss_logger is not None:
            loss_logger(period_label, epoch, opt_step, last_loss)
        # Human-readable epoch summary to training.log
        if text_logger is not None:
            lr_now = optimizer.param_groups[0]["lr"]
            text_logger(
                f"  Epoch {epoch}/{n_epochs} done | period={period_label} | "
                f"avg_loss={last_loss:.4f} | opt_steps={opt_step} | lr={lr_now:.2e}"
            )

    return last_loss


def make_epoch_bar(n_epochs: int, period_label: str, period_idx: int, n_periods: int):
    """Outer epoch tqdm for baselines with custom inner loops."""
    return _tqdm(
        range(1, n_epochs + 1),
        desc=_bold(f"  Period {period_idx+1}/{n_periods} [{period_label}]"),
        unit="epoch", position=0, leave=True, dynamic_ncols=True,
        colour="green" if _TQDM else None,
    )


def make_batch_bar(dataloader, epoch: int, n_epochs: int):
    """Inner batch tqdm for one epoch."""
    return _tqdm(
        dataloader,
        desc=f"    Epoch {epoch}/{n_epochs}",
        unit="batch", position=1, leave=False, dynamic_ncols=True,
    )


# ── Period banner ─────────────────────────────────────────────────────────────

def display_period_header(period_idx: int, n_periods: int, period_label: str,
                          baseline_name: str, n_train: int, n_eval: int,
                          n_probes: int) -> None:
    print()
    print(_bold("━" * _W))
    print(_bold(f"  PERIOD {period_idx+1}/{n_periods}  ·  {period_label}  ·  {baseline_name}"))
    print(_dim(f"  train: {n_train:,} docs   eval: {n_eval:,} docs   probes: {n_probes:,}"))
    print(_bold("━" * _W))


def display_pre_metrics(
    ppl: float,
    probe: float,
    combined: float,
    em: float = 0.0,
    f1: float = 0.0,
    model_type: str = "seq2seq",
) -> None:
    if model_type == "seq2seq":
        print(
            f"  {_dim('PRE ')}  "
            f"EM={_yellow(f'{em:.3f}')}  "
            f"F1={_yellow(f'{f1:.3f}')}  "
            f"PPL={_yellow(f'{ppl:.2f}')}  "
            f"Combined={_yellow(f'{combined:.3f}')}"
        )
    else:
        print(
            f"  {_dim('PRE ')}  "
            f"PPL={_yellow(f'{ppl:.3f}')}  "
            f"Probe={_yellow(f'{probe:.3f}')}  "
            f"Combined={_yellow(f'{combined:.3f}')}"
        )


def display_post_metrics(
    ppl: float,
    probe: float,
    combined: float,
    rir_val: float,
    loss: float,
    pre_combined: float,
    em: float = 0.0,
    f1: float = 0.0,
    model_type: str = "seq2seq",
) -> None:
    delta   = combined - pre_combined
    delta_s = _green(f"+{delta:.3f}") if delta >= 0 else _red(f"{delta:.3f}")
    rir_s   = _green(f"+{rir_val:.3f}") if rir_val >= 0 else _red(f"{rir_val:.3f}")
    if model_type == "seq2seq":
        print(
            f"  {_bold('POST')}  "
            f"EM={_cyan(f'{em:.3f}')}  "
            f"F1={_cyan(f'{f1:.3f}')}  "
            f"PPL={_cyan(f'{ppl:.2f}')}  "
            f"Combined={_cyan(f'{combined:.3f}')} ({delta_s})  "
            f"RIR={rir_s}  Loss={loss:.4f}"
        )
    else:
        print(
            f"  {_bold('POST')}  "
            f"PPL={_cyan(f'{ppl:.3f}')}  "
            f"Probe={_cyan(f'{probe:.3f}')}  "
            f"Combined={_cyan(f'{combined:.3f}')} ({delta_s})  "
            f"RIR={rir_s}  Loss={loss:.4f}"
        )


# ── Accuracy matrix ───────────────────────────────────────────────────────────

def display_accuracy_matrix(acc_mat, period_idx: int) -> None:
    """Print the lower-triangular probe-acc matrix accumulated so far.

    Colour coding:
      green  = diagonal (period just trained)
      cyan   = last row (current model's view of all past periods → BWT)
      plain  = earlier off-diagonal entries
    """
    n = period_idx + 1
    labels = acc_mat.labels[:n]
    col_w = max(max(len(lb) for lb in labels), 6) + 1

    print()
    print(_bold(f"  Accuracy Matrix (probe_acc) — after Period {n}/{acc_mat.labels.__len__()}:"))

    # Header row (column labels)
    header = "  " + " " * (col_w + 2)
    for lb in labels:
        header += lb.rjust(col_w)
    print(_bold(header))
    print("  " + "─" * (col_w + 2 + col_w * n))

    for t in range(n):
        # Row label — pre-format to fixed width, then bold
        label_cell = _bold(labels[t].ljust(col_w + 2))
        row = "  " + label_cell
        for p in range(n):
            if p > t:
                row += " " * col_w          # upper triangle blank
            else:
                val = acc_mat.get(t, p)
                cell = f"{val:.3f}".rjust(col_w)
                if p == t:
                    row += _green(cell)     # diagonal
                elif t == n - 1:
                    row += _cyan(cell)      # last row
                else:
                    row += cell
        print(row)

    print(_dim(
        "  " + " " * (col_w + 2)
        + "green=diagonal(trained)   cyan=last-row(BWT view)"
    ))


# ── Running CAPSEL metrics box ────────────────────────────────────────────────

def display_capsel_running(acc_val: float, bwt_val: float, fwt_val: float,
                           n_done: int, n_total: int) -> None:
    bwt_s = _green(f"{bwt_val:+.4f}") if bwt_val >= -0.01 else _red(f"{bwt_val:+.4f}")
    fwt_s = f"{fwt_val:+.4f}" if n_done >= 2 else _dim("n/a (need ≥2 periods)")

    title = f"  CAPSEL Running Metrics — after Period {n_done}/{n_total}"
    box_inner = _W - 4
    print()
    print("  ╔" + "═" * box_inner + "╗")
    print(f"  ║  {_bold(title.strip()):<{box_inner - 2}}  ║")
    print("  ╠" + "═" * box_inner + "╣")
    print(f"  ║  {'ACC  (avg probe acc, last row)':<34} {_bold(f'{acc_val:.4f}'):<20}  ║")
    print(f"  ║  {'BWT  (backward transfer)':<34} {bwt_s:<20}  ║")
    print(f"  ║  {'FWT  (forward transfer)':<34} {fwt_s:<20}  ║")
    print("  ╚" + "═" * box_inner + "╝")


# ── Final summary ─────────────────────────────────────────────────────────────

def display_final_summary(summary: Dict[str, Any],
                          records: List[Dict[str, Any]]) -> None:
    name    = summary.get("baseline", "?")
    dataset = summary.get("dataset", "?")
    model   = summary.get("model", "?")
    periods = summary.get("periods", [])
    acc_v   = summary.get("ACC", 0.0)
    bwt_v   = summary.get("BWT", 0.0)
    fwt_v   = summary.get("FWT", 0.0)
    final_c = summary.get("final_combined_last_period", 0.0)
    params  = summary.get("params_total", 0)
    trainp  = summary.get("params_trainable_final", 0)

    print()
    print(_bold("═" * _W))
    print(_bold(
        f"  FINAL CAPSEL SUMMARY  ·  {name}  ·  {dataset}"
        f"  ·  {len(periods)} periods  ·  {model}"
    ))
    print(_bold("═" * _W))

    bwt_s = _green(f"{bwt_v:+.4f}") if bwt_v >= -0.01 else _red(f"{bwt_v:+.4f}")

    rows = [
        ("ACC  (avg probe acc, final row)", _bold(f"{acc_v:.4f}"), ""),
        ("BWT  (backward transfer)",        bwt_s,                "← 0 = no forgetting"),
        ("FWT  (forward transfer)",         f"{fwt_v:+.4f}",      ""),
        ("Combined score (last period)",    f"{final_c:.4f}",      ""),
        ("Params total",                    f"{params:,}",         ""),
        ("Params trainable (final)",        f"{trainp:,}",         ""),
    ]
    print()
    print(f"  {'Metric':<38}  {'Value':<14}  {'Note'}")
    print(f"  {'─'*38}  {'─'*14}  {'─'*22}")
    for label, val, note in rows:
        print(f"  {label:<38}  {val:<14}  {note}")

    # Per-period breakdown
    if records:
        print()
        print(_bold("  Per-period breakdown:"))
        model_type = summary.get("model_type", "seq2seq")
        if model_type == "seq2seq":
            hdr = (f"  {'Period':<14}  {'Pre EM':>7}  {'Post EM':>7}  "
                   f"{'Pre F1':>7}  {'Post F1':>7}  {'RIR':>7}  {'Loss':>8}")
        else:
            hdr = (f"  {'Period':<14}  {'Pre PPL':>8}  {'Post PPL':>8}  "
                   f"{'Pre Probe':>10}  {'Post Probe':>10}  {'RIR':>7}  {'Loss':>8}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for r in records:
            rir_v = r.get("rir", 0.0)
            rir_s = _green(f"{rir_v:+.3f}") if rir_v >= 0 else _red(f"{rir_v:+.3f}")
            if model_type == "seq2seq":
                pre_em  = r.get("pre_em",  0.0)
                post_em = r.get("post_em", 0.0)
                pre_f1  = r.get("pre_f1",  0.0)
                post_f1 = r.get("post_f1", 0.0)
                post_f1_s = _green(f"{post_f1:.3f}") if post_f1 >= 0.5 else f"{post_f1:.3f}"
                print(
                    f"  {r['period']:<14}  "
                    f"{pre_em:>7.3f}  {post_em:>7.3f}  "
                    f"{pre_f1:>7.3f}  {post_f1_s:>7}  "
                    f"{rir_s:>7}  {r['final_loss']:>8.4f}"
                )
            else:
                pp   = r.get("post_probe_acc", 0.0)
                pp_s = _green(f"{pp:.3f}") if pp >= 0.5 else f"{pp:.3f}"
                print(
                    f"  {r['period']:<14}  "
                    f"{r['pre_ppl']:>8.3f}  {r['post_ppl']:>8.3f}  "
                    f"{r['pre_probe_acc']:>10.3f}  {pp_s:>10}  "
                    f"{rir_s:>7}  {r['final_loss']:>8.4f}"
                )

    print()
    print(_bold("═" * _W))
    print()

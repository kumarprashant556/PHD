"""Phase 0 shared trainer skeleton (v2: seq2seq + causal, CAPSEL metrics).

Every baseline subclass / wraps the :class:`Baseline` protocol and is driven
by the :class:`BaselineRunner` defined here. The runner:

* loads periods from ``Phase0/data/processed/<dataset>/`` via
  :func:`common.harness.load_periods`,
* computes pre-training metrics for each period
  (EM + F1 for seq2seq, PPL + probe_acc for causal),
* calls the baseline's three hooks
  (``on_period_start`` / ``train_period`` / ``on_period_end``),
* computes post-training metrics,
* re-evaluates every past period to build the ``StreamAccuracyMatrix``
  underlying BWT/ACC/FWT (using F1 for seq2seq, probe_acc for causal),
* displays a rich per-period panel (tqdm from baselines + CAPSEL metrics box),
* writes the shared artefacts and a final summary.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .config import Phase0Config, auto_device, ensure_results_dir, snapshot_config
from .harness import (
    Period,
    RunLogger,
    evaluate_past_periods,
    load_periods,
    param_counts,
)
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
from .progress import (
    display_period_header,
    display_pre_metrics,
    display_post_metrics,
    display_accuracy_matrix,
    display_capsel_running,
    display_final_summary,
)


# ── Baseline protocol ────────────────────────────────────────────────────────

class Baseline(Protocol):
    """Structural type every baseline must implement."""

    name: str
    extras: Dict[str, Any]

    def build_model(self, tokenizer, device: str) -> torch.nn.Module: ...
    def scoring_model(self) -> torch.nn.Module: ...
    def on_period_start(self, period: Period) -> None: ...
    def train_period(self, period: Period,
                     scheduler=None, loss_logger=None) -> float: ...
    def on_period_end(self, period: Period) -> None: ...


# ── Runner ───────────────────────────────────────────────────────────────────

class BaselineRunner:
    """Drives a baseline end-to-end and writes CAPSEL-flavoured metrics."""

    def __init__(self, cfg: Phase0Config, baseline: Baseline):
        self.cfg      = cfg
        self.baseline = baseline
        self.device   = cfg.device or auto_device()
        self._seed_everything(cfg.seed)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        data_root = (
            Path(__file__).resolve().parent.parent
            / "data" / "processed" / cfg.dataset
        )
        self.periods = load_periods(
            data_root=str(data_root),
            max_periods=cfg.max_periods,
            ppl_eval_frac=cfg.ppl_eval_frac,
            seed=cfg.seed,
            max_docs_per_period=cfg.max_docs_per_period or None,
            model_type=cfg.model_type,
            max_train_probes=getattr(cfg, "max_train_probes", 0),
        )

        out_dir      = ensure_results_dir(cfg, baseline.name)
        self.logger  = RunLogger(
            out_dir, baseline.name,
            snapshot_config(cfg, extras=baseline.extras),
        )
        self.model = baseline.build_model(self.tokenizer, self.device)

        # ── LR scheduler (linear warmup → cosine decay) ───────────────
        # Estimated total optimiser steps across all periods.
        accum = max(1, getattr(cfg, "grad_accum_steps", 1))
        total_opt_steps = max(1, sum(
            max(1, len(p.train_items) // max(1, cfg.batch_size) // accum)
            * cfg.epochs_per_period
            for p in self.periods
        ))
        warmup_steps = max(1, int(total_opt_steps * getattr(cfg, "warmup_ratio", 0.06)))
        self.logger.log(
            f"Scheduler: total_opt_steps={total_opt_steps}, "
            f"warmup_steps={warmup_steps}"
        )

        self._scheduler = None
        optimizer = getattr(baseline, "_optimizer", None)
        if optimizer is not None:
            try:
                from transformers import get_cosine_schedule_with_warmup
                self._scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_opt_steps,
                )
            except Exception as e:
                self.logger.log(f"Scheduler creation skipped: {e}")

        # ── Logger callbacks ──────────────────────────────────────────
        self._loss_logger = self.logger.log_loss   # CSV loss curve
        self._text_logger = self.logger.log        # human-readable training.log

    # ── utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _eval_period(self, period: Period) -> Dict[str, float]:
        """Evaluate the current scoring model on one period.

        Returns a flat dict with all metric keys.
        """
        cfg = self.cfg
        m   = self.baseline.scoring_model()

        if cfg.model_type == "seq2seq":
            em, f1, _ = seq2seq_probe_accuracy(
                m, self.tokenizer, self.device,
                period.probes,
                max_n=cfg.probe_max,
                max_new_tokens=cfg.max_new_tokens,
                max_input_len=cfg.max_seq_len,
            )
            ppl = seq2seq_perplexity(
                m, period.probes, self.tokenizer, self.device,
                max_n=cfg.ppl_eval_samples,
                max_input_len=cfg.max_seq_len,
                max_answer_len=cfg.max_answer_len,
            )
            comb = seq2seq_combined_score(em, f1, cfg.em_weight, cfg.f1_weight)
            return {"em": em, "f1": f1, "ppl": ppl, "combined": comb,
                    "probe_acc": f1}    # probe_acc alias = F1 for BWT matrix
        else:
            ppl = perplexity(
                m, period.eval_items, self.tokenizer, self.device,
                max_seq_len=cfg.max_seq_len, max_n=cfg.ppl_eval_samples,
            )
            probe_acc, _ = probe_accuracy(
                m, self.tokenizer, self.device,
                period.probes, max_n=cfg.probe_max,
            )
            comb = combined_score(ppl, probe_acc,
                                  ppl_weight=cfg.ppl_weight,
                                  probe_weight=cfg.probe_weight,
                                  ppl_decay=cfg.ppl_decay)
            return {"em": probe_acc, "f1": probe_acc, "ppl": ppl,
                    "combined": comb, "probe_acc": probe_acc}

    # ── main loop ──────────────────────────────────────────────────────

    def run(self) -> List[Dict[str, Any]]:
        cfg = self.cfg
        b   = self.baseline
        T   = len(self.periods)
        mt  = cfg.model_type

        self.logger.log("=" * 72)
        self.logger.log(
            f"Baseline '{b.name}' | device={self.device} | dataset={cfg.dataset} "
            f"| model={cfg.model_name} | model_type={mt} | periods={T} | seed={cfg.seed}"
        )
        self.logger.log("=" * 72)

        if not self.periods:
            self.logger.log(
                "No periods found. Did you run Phase0/data/download_"
                f"{cfg.dataset}.py first?"
            )
            return []

        init_counts = param_counts(b.scoring_model())
        self.logger.log(
            f"Initial params: total={init_counts['params_total']:,} "
            f"trainable={init_counts['params_trainable']:,}"
        )

        acc_mat = StreamAccuracyMatrix(matrix=[], labels=[p.label for p in self.periods])
        records: List[Dict[str, Any]] = []
        past:    List[Period]          = []

        for period in self.periods:

            # ── period banner ─────────────────────────────────────────
            display_period_header(
                period_idx=period.index,
                n_periods=T,
                period_label=period.label,
                baseline_name=b.name,
                n_train=len(period.train_items),
                n_eval=len(period.eval_items),
                n_probes=len(period.probes),
            )
            self.logger.log(
                f"\n── Period {period.index+1}/{T}: {period.label} "
                f"({len(period.train_items)} train, {len(period.eval_items)} eval, "
                f"{len(period.probes)} probes)"
            )

            # ── pre-training eval ─────────────────────────────────────
            pre = self._eval_period(period)
            display_pre_metrics(
                ppl=pre["ppl"], probe=pre["probe_acc"], combined=pre["combined"],
                em=pre["em"], f1=pre["f1"], model_type=mt,
            )
            self.logger.log(
                f"  [pre]  em={pre['em']:.3f}  f1={pre['f1']:.3f}  "
                f"ppl={pre['ppl']:.3f}  combined={pre['combined']:.3f}"
            )

            # FWT: zero-shot on this period before training
            if period.index > 0:
                acc_mat.set(period.index - 1, period.index, pre["f1"])

            # ── training ──────────────────────────────────────────────
            b.on_period_start(period)
            final_loss = b.train_period(
                period,
                scheduler=self._scheduler,
                loss_logger=self._loss_logger,
                text_logger=self._text_logger,
            )

            # ── post-training eval ────────────────────────────────────
            probes_csv = self.logger.out_dir / f"probes_period{period.index+1}.csv"
            if mt == "seq2seq":
                post_em, post_f1, _ = seq2seq_probe_accuracy(
                    b.scoring_model(), self.tokenizer, self.device,
                    period.probes,
                    max_n=cfg.probe_max,
                    max_new_tokens=cfg.max_new_tokens,
                    max_input_len=cfg.max_seq_len,
                    record_csv=probes_csv,
                )
                post_ppl = seq2seq_perplexity(
                    b.scoring_model(), period.probes, self.tokenizer, self.device,
                    max_n=cfg.ppl_eval_samples,
                    max_input_len=cfg.max_seq_len,
                    max_answer_len=cfg.max_answer_len,
                )
                post_combined = seq2seq_combined_score(
                    post_em, post_f1, cfg.em_weight, cfg.f1_weight,
                )
                post_probe = post_f1  # alias for BWT matrix
            else:
                post_ppl = perplexity(
                    b.scoring_model(), period.eval_items,
                    self.tokenizer, self.device,
                    max_seq_len=cfg.max_seq_len, max_n=cfg.ppl_eval_samples,
                )
                post_probe, _ = probe_accuracy(
                    b.scoring_model(), self.tokenizer, self.device,
                    period.probes, max_n=cfg.probe_max, record_csv=probes_csv,
                )
                post_em = post_probe
                post_f1 = post_probe
                post_combined = combined_score(
                    post_ppl, post_probe,
                    ppl_weight=cfg.ppl_weight,
                    probe_weight=cfg.probe_weight,
                    ppl_decay=cfg.ppl_decay,
                )

            period_rir = rir(post_combined, pre["combined"], chance=0.0)

            display_post_metrics(
                ppl=post_ppl, probe=post_probe, combined=post_combined,
                rir_val=period_rir, loss=final_loss, pre_combined=pre["combined"],
                em=post_em, f1=post_f1, model_type=mt,
            )
            self.logger.log(
                f"  [post] em={post_em:.3f}  f1={post_f1:.3f}  "
                f"ppl={post_ppl:.3f}  combined={post_combined:.3f}  "
                f"RIR={period_rir:+.3f}  loss={final_loss:.4f}"
            )

            b.on_period_end(period)
            past.append(period)

            # ── BWT matrix row ────────────────────────────────────────
            bwt_row = evaluate_past_periods(
                b.scoring_model(), self.tokenizer, self.device, past, cfg,
            )
            for past_p in past:
                # BWT matrix scalar = F1 (seq2seq) or probe_acc (causal)
                bwt_scalar = bwt_row[past_p.label].get("f1",
                             bwt_row[past_p.label].get("probe_acc", 0.0))
                acc_mat.set(period.index, past_p.index, bwt_scalar)

            self.logger.log(
                "  [BWT]  "
                + "  ".join(
                    f"{k}=f1:{v.get('f1', v.get('probe_acc', 0)):.3f}"
                    for k, v in bwt_row.items()
                )
            )

            # ── per-period record ─────────────────────────────────────
            counts_now = param_counts(b.scoring_model())
            records.append({
                "period":          period.label,
                "period_num":      period.index + 1,
                "pre_em":          pre["em"],
                "pre_f1":          pre["f1"],
                "pre_ppl":         pre["ppl"],
                "pre_probe_acc":   pre["probe_acc"],
                "pre_combined":    pre["combined"],
                "post_em":         post_em,
                "post_f1":         post_f1,
                "post_ppl":        post_ppl,
                "post_probe_acc":  post_probe,
                "post_combined":   post_combined,
                "rir":             period_rir,
                "final_loss":      final_loss,
                "bwt_row":         bwt_row,
                **counts_now,
            })
            self.logger.save_record(records[-1])

            # ── rich display ──────────────────────────────────────────
            display_accuracy_matrix(acc_mat, period.index)

            acc_v = stream_acc(acc_mat)
            bwt_v = stream_bwt(acc_mat)
            fwt_v = stream_fwt(acc_mat, chance=0.0)
            display_capsel_running(acc_v, bwt_v, fwt_v, period.index + 1, T)

            self.logger.log(
                f"  [CAPSEL] ACC={acc_v:.4f}  BWT={bwt_v:+.4f}  FWT={fwt_v:+.4f}"
            )

        # ── final summary ─────────────────────────────────────────────
        summary = {
            "baseline":                   b.name,
            "dataset":                    cfg.dataset,
            "model":                      cfg.model_name,
            "model_type":                 mt,
            "periods":                    [p.label for p in self.periods],
            "ACC":                        stream_acc(acc_mat),
            "BWT":                        stream_bwt(acc_mat),
            "FWT":                        stream_fwt(acc_mat, chance=0.0),
            "final_combined_last_period": records[-1]["post_combined"] if records else 0.0,
            "final_em_last_period":       records[-1]["post_em"]       if records else 0.0,
            "final_f1_last_period":       records[-1]["post_f1"]       if records else 0.0,
            "params_total":               records[-1]["params_total"]  if records else 0,
            "params_trainable_final":     records[-1]["params_trainable"] if records else 0,
        }
        self.logger.finalize(summary)
        display_final_summary(summary, records)
        return records

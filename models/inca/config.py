"""INCA-v2 configuration.

Extends a minimal BaseConfig with all INCA-specific hyperparameters.

Field-name convention (matches both trainers and inca.yaml exactly):
  lr                  — learning rate  (trainers use cfg.lr)
  epochs_per_period   — epochs per CL period (trainers use cfg.epochs_per_period)
  n_per_period        — max items per period  (trainers use cfg.n_per_period)
  max_periods         — cap on number of periods; None = use all
  dataset             — dataset name passed to data.load_periods()
  grad_accum_steps    — gradient accumulation steps
  split_frac          — completion split fraction fed to data loaders
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BaseConfig:
    """Shared hyperparameters (replaces Phase0Config)."""
    model_name:         str   = "google/flan-t5-base"
    max_input_length:   int   = 256
    max_target_length:  int   = 256
    lr:                 float = 3e-4        # trainers access as cfg.lr
    weight_decay:       float = 0.01
    batch_size:         int   = 32
    epochs_per_period:  int   = 5           # trainers access as cfg.epochs_per_period
    warmup_steps:           int   = 200
    warmup_ratio:           float = 0.06   # used when warmup_steps==0; 6% of total steps
    grad_accum_steps:       int   = 1
    max_grad_norm:          float = 1.0    # gradient clipping
    gradient_checkpointing: bool  = False  # recompute activations (saves VRAM, slower)
    use_adafactor:          bool  = False  # Adafactor instead of AdamW (lower memory)
    precision:              str   = "bf16" # "bf16" | "fp16" | "fp32"; bf16 ≈ 2× speed on
                                            #   M2+/Ampere+, fp16 unstable on T5
    ppl_eval_frac:          float = 0.05   # fraction of period used for perplexity eval
    seed:                   int   = 42
    fp16:                   bool  = False
    log_every_n_steps:      int   = 50


@dataclass
class INCAConfig(BaseConfig):
    # ── Dataset ───────────────────────────────────────────────────────
    dataset:        str            = "cc_news"   # passed to data.load_periods()
    n_per_period:   int            = 20_000      # max items loaded per period
    max_periods:    Optional[int]  = None        # None = all periods; set e.g. 3 for fast smoke
    split_frac:     float          = 0.50        # completion framing: encoder gets first half

    # ── Block-chain growth ────────────────────────────────────────────
    n_max_blocks:          int   = 8    # hard cap; raises RuntimeError if exceeded
    k_eval:                int   = 50   # evaluate saturation signals every k_eval opt steps
    patience:              int   = 5    # consecutive evals plateau must hold before firing
    min_epochs_before_grow: int  = 2    # grokking guard: never grow before this many epochs
    layers_per_block:      int   = 4    # flan-t5-base has 12 layers; 4 → 3 blocks max
    lateral_rank:          int   = 0    # 0 = no lateral adapters (Phase 2); >0 enables them

    # ── Multi-signal consensus (T1.1) ─────────────────────────────────
    rir_threshold:        float = 0.30   # min RIR for PERIOD_LEARNED to fire
    rir_negligible:       float = 0.05   # max RIR for BLOCK_FULL to fire
    grad_norm_ema_alpha:  float = 0.10   # EMA smoothing factor for grad-norm tracker
    grad_norm_decay_frac: float = 0.50   # "decayed" if current < frac × peak
    chance:               float = 0.0    # RIR baseline rate (0.25 for 4-way MCQ; 0.0 for open-answer)

    # ── CKA reference set (T1.5) ──────────────────────────────────────
    cka_ref_size:             int   = 200   # items cached at period start
    cka_saturation_threshold: float = 0.95  # CKA ≥ this → representation stable

    # ── Study-schedule replay (T1.4) ──────────────────────────────────
    buffer_max_size: int   = 2000   # max items retained per period in replay buffer
    replay_ratio:    float = 0.25   # fraction of each micro-batch drawn from replay
    n_revise:        int   = 3      # epochs in initial-pass regime (uniform sampling)
    p_hard:          float = 0.70   # refinement: fraction of hardest-loss items
    p_easy:          float = 0.20   # refinement: fraction of easiest-loss items
    p_mid:           float = 0.10   # random mid-loss  (p_hard+p_easy+p_mid must == 1.0)

    # ── Period drift safety check (T1.3) ──────────────────────────────
    period_drift_tol:    float = 0.10   # fire BLOCK_FULL early if replay_acc drops > tol

    # ── Early-stopping relabelling (T1.2) ─────────────────────────────
    min_rir_for_learned: float = 0.20   # timeout → LEARNED only if RIR ≥ this; else EXHAUSTED

    # ── Selector (E-ROUTE ablation key) ───────────────────────────────
    #   "embedding_query"  S-QKV  Q=frozen embeddings; K,V from blocks (default)
    #   "uclbr"            UCLBR  pre-gate + load-balance + uncertainty calibration
    #   "cross_attention"  S-FULL MLP gate on mean-pooled block outputs
    #   "weighted_sum"     S-WS   blind input-independent scalar per block
    selector:        str = "embedding_query"
    selector_heads:  int = 4    # attention heads for EmbeddingQuerySelector / UCLBR
    selector_hidden: int = 64   # hidden dim for CrossAttentionSelector gate MLP

    # UCLBR-specific
    uclbr_pre_gate_hidden: int   = 64
    uclbr_heads:           int   = 4
    uclbr_bias_lr:         float = 1e-3
    uclbr_top_k:           int   = 0    # 0 = soft gate; >0 = hard top-k sparsity

    # ── Replay sampling per period ─────────────────────────────────────
    replay_n_per_period: int = 2_000   # # items drawn from buffer at period start

    # ── Output ────────────────────────────────────────────────────────
    out_dir: str = "results"   # training run outputs (relative to repo root)

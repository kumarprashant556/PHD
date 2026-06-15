"""INCA-v2 configuration.

Extends Phase0Config with all INCA-specific hyperparameters so every
run is fully reproducible and comparable via a single config file.

Phase-1 additions (marked [P1]):
  - multi-signal consensus params (T1.1)
  - CKA reference-set params (T1.5)
  - study-schedule replay params (T1.4)
  - period-drift tolerance (T1.3)
  - early-stopping relabelling threshold (T1.2)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from Phase0.common.config import Phase0Config


@dataclass
class INCAConfig(Phase0Config):
    # ── Block-chain growth ────────────────────────────────────────────
    n_max_blocks: int = 8           # hard cap; raise RuntimeError if exceeded
    k_eval: int = 50                # evaluate saturation signals every k_eval opt steps
    patience: int = 5               # consecutive evals plateau must hold before firing
    min_epochs_before_grow: int = 2 # grokking guard: never grow before this many epochs

    # layers per block (number of encoder layers assigned to each INCA block)
    layers_per_block: int = 4       # flan-t5-base has 12; 4 gives 3 blocks max before OOM
    lateral_rank: int = 0           # 0 = no lateral adapters (Phase 2); >0 enables them

    # ── [P1] Multi-signal consensus (T1.1) ───────────────────────────
    rir_threshold: float = 0.30     # min RIR for period_learned to fire
    rir_negligible: float = 0.05    # max RIR for block_full to fire
    grad_norm_ema_alpha: float = 0.10   # EMA smoothing factor for grad-norm tracker
    grad_norm_decay_frac: float = 0.50  # grad-norm considered "decayed" if current < frac * peak

    # ── [P1] CKA reference set (T1.5) ────────────────────────────────
    cka_ref_size: int = 200         # items cached at period start
    cka_saturation_threshold: float = 0.95  # CKA >= this → representation stable

    # ── [P1] Study-schedule replay (T1.4) ────────────────────────────
    buffer_max_size: int = 2000     # max items retained per period in replay buffer
    replay_ratio: float = 0.25      # fraction of each micro-batch drawn from replay
    n_revise: int = 3               # epochs in initial-pass regime (uniform sampling)
    p_hard: float = 0.70            # refinement: fraction of hardest-loss items
    p_easy: float = 0.20            # refinement: fraction of easiest-loss items
    p_mid: float = 0.10             # refinement: random mid-loss items (must sum to 1.0)

    # ── [P1] Period drift safety check (T1.3) ────────────────────────
    period_drift_tol: float = 0.10  # trigger BlockFull early if replay_acc drops by > tol

    # ── [P1] Early-stopping relabelling (T1.2) ───────────────────────
    min_rir_for_learned: float = 0.20  # timeout → learned only if RIR >= this; else exhausted

    # ── Selector ─────────────────────────────────────────────────────
    # selector: which aggregation module to use (E-ROUTE ablation)
    #   "embedding_query"  — EmbeddingQuerySelector (S-QKV, recommended default)
    #                         Q = frozen original embeddings; K,V from block outputs.
    #                         Token-level; blocks compete to best answer the original input.
    #   "cross_attention"  — CrossAttentionSelector  (S-FULL)
    #                         MLP gate on mean-pooled block outputs; sequence-level.
    #   "weighted_sum"     — WeightedSumSelector      (S-WS, blind control)
    # E-ROUTE ablation — set selector in YAML to compare all variants:
    #   "embedding_query"  S-QKV  : Q=frozen embeddings; K,V from blocks (default)
    #   "uclbr"            UCLBR  : pre-gate + load-balance + uncertainty calibration
    #   "cross_attention"  S-FULL : MLP gate on mean-pooled block outputs
    #   "weighted_sum"     S-WS   : blind input-independent scalar per block
    selector: str = "embedding_query"
    selector_heads: int = 4         # attention heads for EmbeddingQuerySelector / UCLBR
    selector_hidden: int = 64       # hidden dim for CrossAttentionSelector gate MLP

    # UCLBR-specific
    uclbr_pre_gate_hidden: int   = 64     # Read-ME pre-gate MLP hidden dim
    uclbr_heads: int             = 4      # routing attention heads
    uclbr_bias_lr: float         = 1e-3   # load-balance bias update rate
    uclbr_top_k: int             = 0      # 0 = soft gate; >0 = hard top-k sparsity

    # ── Output ───────────────────────────────────────────────────────
    out_dir: str = "Phase1/results"  # root directory for training run outputs

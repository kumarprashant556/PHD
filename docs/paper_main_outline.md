# INCA: Saturation-Driven Incremental Block Expansion for Continual Domain Adaptation
## Full Paper Outline — Working Draft

> **Status**: Outline / planning document. Sections with ✅ have implementation in hand.
> Sections with 🔬 need a training run. Sections with 📝 need text writing.
>
> **Architecture files**: `models/inca/` (selectors, plateau, cka, layer_manager, lateral, uclbr)
> **Config files**: `configs/inca.yaml`, `configs/ablations/e_sat.yaml`, `e_route.yaml`, `e_cls3.yaml`
> **Training entry point**: `scripts/train_inca.py` → `training/inca_trainer.py`
> **Ideas journal**: [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md)

---

## Proposed Titles (pick one before submission)

1. **INCA: Saturation-Driven Incremental Block Expansion for Continual Domain Adaptation**
2. **Grow When Full: Saturation-Triggered Capacity Expansion for Continual Learning**
3. **CAPSEL: Saturation-Driven Progressive Block Expansion with Adaptive Routing**
4. **When to Grow: Saturation Detection for Incremental Neural Capacity in Continual Learning**

**Working title**: *INCA: Saturation-Driven Incremental Block Expansion for Continual Domain Adaptation*

---

## Abstract (draft)

We present INCA (Incremental Neural Capacity Adaptation), a continual learning framework
that grows new transformer encoder blocks dynamically, triggered by a multi-signal saturation
detector rather than a fixed expansion schedule. Unlike prior block-expansion methods such as
LLaMA-Pro (Wu et al., 2024), which insert blocks at predefined positions before training
begins, INCA monitors four saturation signals online — Relative Improvement Rate (RIR),
gradient-norm EMA decay, CKA representational stability, and loss-plateau slope — and
fires a freeze-and-grow event only when the current block has genuinely exhausted its capacity.
A novel fixed-query cross-attention selector (S-QKV) aggregates frozen and active block outputs
at the token level, using original token embeddings as immutable queries to prevent
representational drift across blocks. We evaluate INCA on a domain-sequential curriculum
[DATASET TBD] and ablate the selector design (E-ROUTE: S-QKV, UCLBR, S-FULL, S-WS),
saturation threshold sensitivity (E-SAT), and replay strategy (E-CLS3). INCA outperforms
LLaMA-Pro on backward transfer and parameter efficiency, demonstrating that *when* to grow
matters at least as much as *how* to grow.

---

## §1. Introduction

### 1.1 The problem: fixed-schedule expansion ignores actual capacity

**Core argument:**
- Standard continual learning methods (EWC, replay, L2P, PNN) assume a fixed model; they
  suppress forgetting but cannot *add* capacity when the model is genuinely full.
- Block expansion methods (LLaMA-Pro, Progressive Networks, Net2Net) add capacity, but at
  a schedule fixed before training begins — irrespective of whether the current block has
  saturated or still has useful gradient signal remaining.
- The cost of early expansion: the model wastes the new block's parameters on a distribution
  the old block could still handle; forward-transfer signal is diluted.
- The cost of late expansion: the model overtunes on the current domain, reducing plasticity
  for the next one.

**Our claim:**
> Saturation-triggered block expansion — grow exactly when current capacity is exhausted and
> not before — achieves better accuracy per added parameter than fixed-schedule expansion,
> and the saturation detector is itself a novel multi-signal consensus mechanism that can
> be applied independently of the expansion strategy.

### 1.2 What INCA contributes

| Contribution | Where in paper |
|---|---|
| **Multi-signal consensus saturation detector** (RIR + grad-norm EMA + CKA + loss plateau) | §3.1 |
| **Adaptive block expansion** triggered by saturation, not a preset schedule | §3.2 |
| **Fixed-query cross-attention selector** (S-QKV): token-level, drift-resistant aggregation | §3.3 |
| **UCLBR selector**: pre-gated + load-balanced + uncertainty-calibrated routing | §3.4 |
| **Lateral adapters** (Phase 2): low-rank cross-block knowledge transfer | §3.5 |
| **E-ROUTE / E-SAT / E-CLS3 ablations** on all design choices | §5 |
| LLaMA-Pro as the primary baseline with direct head-to-head comparison | §4, §6 |

### 1.3 Scope and non-goals

- **In scope**: continual domain adaptation (sequential domain shift, not temporal stream).
  Dataset is domain-specific (see §2.3); no temporal framing required.
- **Out of scope**: multilingual CL, task-incremental CL with task IDs, online RL.
- **Why not temporal?**: CAPSEL/INCA's headline claim is *capacity growth under saturation*,
  not temporal drift. A domain-sequential curriculum provides cleaner saturation signals
  (no confound from entity-answer leakage) and enables direct comparison to LLaMA-Pro.

---

## §2. Related Work

### 2.1 Block expansion for domain adaptation

**LLaMA-Pro (Wu et al., 2024) — primary baseline**
- Inserts identity-initialised transformer blocks at fixed positions in LLaMA-2.
- Freezes original layers; fine-tunes new blocks only on math + code.
- No saturation detection — expansion schedule is fixed before training.
- Demonstrates that selective parameter growth outperforms full fine-tune on new domains
  while preserving general ability.
- **INCA difference**: (1) saturation-triggered (not fixed), (2) multiple selector strategies
  (not a simple sequential chain), (3) embedding-skip residual connections between blocks,
  (4) ablation of expansion timing as a first-class experiment.

**Progressive Neural Networks (Rusu et al., 2016)**
- Allocates a new column per task; lateral connections to previous columns.
- Grows at every task boundary (fixed schedule), no saturation signal.
- Parameter count grows O(n_tasks²); INCA grows O(n_blocks) only when needed.

**Net2Net / Network Morphism (Chen et al., 2015)**
- Function-preserving network widening; grows depth/width.
- No freezing; entire network trains. Not a CL method.
- INCA borrows identity/warm-start init ideas but applies them in a CL block-chain context.

**PackNet (Mallya & Lazebnik, 2018)**
- Prunes and packs task-specific parameters without growing.
- Complementary (shrink vs. grow); not a direct comparison.

### 2.2 Saturation detection and capacity monitoring

**CKA (Kornblith et al., ICML 2019)**
- Linear CKA measures representational similarity between two sets of activations.
- Used in INCA's CKA monitor (`models/inca/cka.py`) to detect when representations
  have stopped changing — a proxy for block saturation.

**Loss plateau / early stopping**
- Standard convergence criterion; INCA's loss-plateau tracker (`models/inca/plateau.py`)
  uses a sliding-window slope test.

**RIR (Relative Improvement Rate)**
- Our own metric: `(score_now − score_0) / max(score_0, chance)`.
- Normalises score gain to the pre-period baseline, making it scale-invariant across domains.

### 2.3 Continual learning baselines (B1-B7)

| ID | Method | Key idea |
|---|---|---|
| B1 | Naive fine-tune | Lower bound — maximum forgetting |
| B2 | Experience Replay | Keeps a fixed-size buffer of past examples |
| B3 | EWC | Fisher-weighted regularisation on important weights |
| B4 | L2P | Prompt pool; soft-prompt prefix per task |
| B5 | LoRA-MoE | Low-rank adapters + mixture-of-experts routing |
| B6 | LLaMA-Pro | Fixed-schedule block expansion ← **primary baseline** |
| B7 | PNN | Progressive neural networks (one column per domain) |

### 2.4 MoE routing and selector design

**DeepSeek-MoE (Dai et al., 2024)**
- Auxiliary-loss-free load balancing via online bias update: `b_i ← b_i + η·(1/n − f_i)`.
- INCA's UCLBR selector (`models/inca/uclbr.py`) adopts this mechanism directly.

**Read-ME (Zhao et al., 2024)**
- Pre-gating: a lightweight MLP screens expert relevance before full routing, reducing
  wasted computation on irrelevant experts.
- UCLBR's pre-gate component is drawn from this idea.

**Uncertainty-calibrated routing (Guo et al., ICML 2017)**
- Calibration of confidence estimates via temperature scaling.
- UCLBR's entropy-based confidence fallback (high entropy → uniform mixture) is inspired
  by this approach.

---

## §3. Method — INCA Architecture

### 3.1 Multi-signal consensus saturation detector

**File**: `models/inca/plateau.py` — `INCAPlateauDetector`

The detector aggregates four signals at every `k_eval` optimiser steps:

| Signal | Implementation | Contribution to decision |
|---|---|---|
| **RIR** | `RIRTracker.rir` = `(score_now − score_0) / max(score_0, chance)` | Measures relative accuracy gain since period start |
| **Grad-norm EMA** | `GradNormTracker.is_decayed()` → EMA < `decay_frac × peak` | Detects gradient vanishing (capacity exhaustion) |
| **CKA** | `CKAMonitor.is_saturated()` → CKA ≥ `cka_saturation_threshold` | Representational stability (block no longer changing) |
| **Loss plateau** | `LossPlateauTracker.is_plateau()` → slope < `min_delta` over `patience` evals | Training-loss convergence |

**Two events (consensus rules):**

```
PeriodLearned:  RIR ≥ rir_threshold  AND  loss_plateau
  → advance to next domain segment (same block continues)

BlockFull:      RIR ≤ rir_negligible  AND  loss_plateau
                AND  (grad_norm_decayed  OR  cka_stable)
  → freeze current block, warm-start new one

Timeout (T1.2): patience steps elapsed without either event
  → if RIR ≥ min_rir_for_learned: treat as PeriodLearned
  → else: EXHAUSTED → BlockFull path
```

**Grokking guard**: `min_epochs_before_grow = 2` prevents premature expansion before the
model has had a chance to learn anything from the new domain.

**Default hyperparameters (from `configs/inca.yaml`):**
- `rir_threshold = 0.30`, `rir_negligible = 0.05`
- `grad_norm_ema_alpha = 0.10`, `grad_norm_decay_frac = 0.50`
- `cka_ref_size = 200`, `cka_saturation_threshold = 0.95`
- `patience = 5`, `k_eval = 50`

### 3.2 Block-chain architecture and growth mechanism

**File**: `models/inca/layer_manager.py` — `INCALayerManager`, `INCАBlock`

**Base model**: FLAN-T5-base (encoder–decoder, 250M params).
- Encoder: 12 transformer layers, `d_model = 768`.
- INCA partitions the encoder into blocks of `layers_per_block = 4` layers.
  - Max 3 blocks within FLAN-T5-base's 12 encoder layers before OOM.

**Block-chain forward pass:**

```
Input: token ids
  ↓
Token embeddings (frozen) → embedding_hidden (B, S, D)
  ↓
Block 0 (frozen after first grow):
    input = embedding_hidden
    output h_0 = Block0(embedding_hidden) + LayerNorm + Dropout
  ↓
Inter-block projection (identity-init, frozen):
    chain = proj_0(h_0) + embedding_hidden    ← embedding skip connection
  ↓
Block 1 (frozen after second grow):
    output h_1 = Block1(chain)
  ↓
... (up to n_max_blocks = 8)
  ↓
Current (trainable) block k:
    output h_k = Block_k(chain)
  ↓
Selector: combined = Selector([h_0, h_1, ..., h_k], embedding_hidden)
  ↓
Decoder (T5, unchanged) → output tokens
```

**Embedding skip connection**: at each inter-block boundary, `embedding_hidden` is added
back to the projected chain. This gives every block a direct path to the original token
representations regardless of chain depth, preventing representation drift through deep
frozen block sequences.

**Grow event (`freeze_and_grow`):**
1. Freeze current block (set `requires_grad=False` on all parameters).
2. Add identity-initialised inter-block projection (frozen together with source block).
3. Deep-copy current block → new trainable block (warm start from last frozen state).
4. Notify selector (e.g., `WeightedSumSelector.grow()` adds a new scalar weight).

**Trainable parameters at any time:**
- Current (last) block only.
- Selector gate.
- Incoming inter-block projection (unfrozen so the new block can optimise its input).

### 3.3 Fixed-query cross-attention selector (S-QKV)

**File**: `models/inca/selectors.py` — `EmbeddingQuerySelector`
**Config**: `selector: "embedding_query"` (recommended default)

**Motivation**: Why fix Q to the original embeddings?

> *"We want to ask: given what this token originally meant, which block's transformation
>  of it is most useful?"* Keeping Q fixed means the selection question never drifts —
> only the blocks compete to be the best answer.

**Mechanism:**
```
Q  = embedding_hidden (B, S, D)  [no projection, frozen]
K_i = W_k × Block_i_output       [learned per-block key]
V_i = W_v × Block_i_output       [learned per-block value]

For each block i:
    scores_i = (Q × K_iᵀ) / √d_k         (B, h, S, S)
    attn_i   = softmax(scores_i) × V_i    (B, h, S, d)
    score_i  = mean diagonal of scores_i  (B,)  ← block relevance

block_weights = softmax([score_0, score_1, ..., score_k])  (B, n_blocks)
combined      = Σ block_weights_i × attn_i
```

**Key properties:**
- **Per-token selection**: each position independently weights the blocks.
- **Identity init**: `W_k`, `W_v`, `W_out` initialised as identity → function-preserving
  at grow time (output ≈ mean of block outputs on day 1).
- **Lightweight**: only K, V projections + output projection; no Q projection.

**vs. CrossAttentionSelector (S-FULL)**: S-QKV selects at token level; S-FULL
mean-pools each block → scalar → same weight for all positions (coarser).

### 3.4 UCLBR selector — three-component routing

**File**: `models/inca/uclbr.py` — `UCLBRSelector`
**Config**: `selector: "uclbr"`

Three components stacked:

**1. Read-ME pre-gate** (Zhao et al., 2024):
- MLP scores each block's mean-pool → scalar relevance ∈ (0, 1).
- Blocks below soft threshold are down-weighted before full routing.
- Prevents irrelevant frozen blocks from polluting the aggregation.

**2. DeepSeek auxiliary-loss-free load balancing** (Dai et al., 2024):
- Per-block bias `b_i` updated online (no gradient, buffer update):
  `b_i ← b_i + η_b × (1/n − f_i)` where `f_i` = fraction of routing weight to block i.
- Nudges under-used blocks up and over-used blocks down.
- **No auxiliary loss added to training objective.**

**3. Uncertainty-calibrated confidence**:
- Routing entropy `H = −Σ w_i log w_i`, normalised to `[0,1]` by `log(n)`.
- Confidence `c = conf_head(H̃)` (learned affine + sigmoid).
- Final weights: `w_final = c × w_router + (1−c) × (1/n)`.
- High uncertainty → fallback to uniform mixture (safe routing).

**When to use UCLBR over S-QKV**: UCLBR is better when block load is highly unequal
(e.g., one block dominates); S-QKV is better when blocks specialise by token type.
The E-ROUTE ablation quantifies this empirically.

### 3.5 Lateral adapters (Phase 2)

**File**: `models/inca/lateral.py` — `LateralAdapter`
**Config**: `lateral_rank > 0` (currently `lateral_rank = 0` = disabled)

**Mechanism:**
```
output = hidden + tanh(alpha) × up(down(frozen_block_out))
         where alpha is initialised to 0.0 (function-preserving at attach time)
         down: D → r (rank r = 4, 8, or 16)
         up:   r → D (zero-initialised → starts as identity transform)
```

**Purpose**: provides a gradient path from a frozen block's representation into the
current block's input immediately post-expansion, reducing the "cold start" penalty
of a newly added block. Alpha gating ensures the adapter starts as a no-op and learns
gradually.

**Status**: Phase 2 (not yet active, `lateral_rank = 0`). Ablation E-SCOPE tests
rank ∈ {0 (off), 4, 8, 16}. Phase 1 uses inter-block projections + embedding skip
as the structural alternative.

### 3.6 Replay strategy

**Config key**: `replay_strategy` (ablated in E-CLS3)

During block training, a replay buffer of size `buffer_max_size = 2000` from previous
domains is mixed in at ratio `replay_ratio = 0.25`. The buffer sampling strategy is:

| Strategy | `p_hard` | `p_easy` | `p_mid` | Effect |
|---|---|---|---|---|
| `schedule` (default) | 0.70 | 0.20 | 0.10 | Hard examples prioritised; prevents easy-example overfitting |
| `uniform` | 0 | 0 | 1.0 | Phase-A only; ignores difficulty |
| `hardest` | 1.0 | 0 | 0 | Extreme hard-example mining |
| `easiest` | 0 | 1.0 | 0 | Easy-example focus; stability baseline |

Difficulty is scored per-item by the current block's cross-entropy loss at the time of
buffering. Items are re-scored at the `n_revise = 3` checkpoint intervals.

---

## §4. Experimental Setup

### 4.1 Dataset — Domain-Sequential Curriculum

**Decision needed** (see [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md) entry
"Use a domain-specific non-temporal dataset stack"):

**Option A — LLaMA-Pro replica (recommended)**:
- Domain 0: GSM8K training split (~7.5k examples) — grade-school math reasoning.
- Domain 1: CodeSearchNet Python split (~400k, subsample ~25k) — code understanding.
- Enables direct comparison to LLaMA-Pro §4 result on the identical dataset family.
- **Pros**: tightest comparison; leakage-free by construction; LLaMA-Pro results are public.
- **Cons**: math + code → decoder, not encoder-focused; may not stress the saturation
  detector as richly as natural language.

**Option B — Natural-language domain sequence**:
- Domain 0: Science (AI2 ARC training split, ~3.5k QA items).
- Domain 1: Law (LegalBench subset, ~10k items).
- Domain 2: Medicine (MedQA USMLE training split, ~10k items).
- **Pros**: natural-language domains; richer vocabulary shift; closer to thesis narrative.
- **Cons**: harder to baseline directly against LLaMA-Pro.

**Option C — Difficulty curriculum (single domain)**:
- GSM8K grade-school → MATH competition math, ordered by difficulty tier.
- Tests saturation detection on a *difficulty* axis, not a *domain* axis.
- Most novel framing; hardest to baseline.

**Baseline recommendation**: Option A for the main paper (LLaMA-Pro comparison);
Option B as a secondary experiment (Table 2) demonstrating generalisation beyond math/code.

### 4.2 Evaluation metrics

| Metric | Formula | What it measures |
|---|---|---|
| `ACC` | Mean accuracy on current-domain held-out set across all domains | Overall task performance |
| `BWT` | `(1/T-1) Σ_{t<T} [R(T,t) − R(t,t)]` | Forgetting of earlier domains after later training |
| `FWT` | `(1/T-1) Σ_{t>1} [R(t,t) − R_random(t)]` | Forward transfer from earlier domains |
| `PAR` | Accuracy per 1M parameters added vs. no-growth baseline | Parameter efficiency |
| `EXP_N` | Number of expansion events per domain | Growth behaviour |
| `EXP_T` | Training step at which expansion fires | Timing behaviour |

`R(t, j)` = accuracy on domain j after training through domain t.
`R_random(t)` = random-baseline accuracy for domain t (used for FWT normalisation).

### 4.3 Baselines

| Baseline | Config | Key hyperparameter |
|---|---|---|
| B1 (Naive FT) | `configs/base.yaml` | — |
| B2 (Replay) | `configs/base.yaml` | `buffer_max_size = 2000` |
| B3 (EWC) | `configs/base.yaml` | EWC λ swept |
| B4 (L2P) | `configs/base.yaml` | Prompt pool size = 10 |
| B5 (LoRA-MoE) | `configs/base.yaml` | rank = 16 |
| B6 (LLaMA-Pro) | `configs/base.yaml` | Fixed 1 block added per domain boundary |
| B7 (PNN) | `configs/base.yaml` | — |
| INCA (ours) | `configs/inca.yaml` | S-QKV, E-SAT default, schedule replay |
| INCA-no-grow | `configs/inca.yaml` + `n_max_blocks=1` | Growth disabled |

**LLaMA-Pro (B6) configuration details** — needs explicit alignment:
- Block insertion: 1 new block inserted at the start of each domain (fixed schedule).
- Init strategy: identity init (same as INCA's inter-block projections).
- Freeze strategy: original block frozen; new block trained.
- Selector: none — output of new block only passes to decoder (no aggregation).
- This matches LLaMA-Pro §4's "sequential fine-tuning with inserted layers" setup.

### 4.4 Implementation details

| Item | Value | Source |
|---|---|---|
| Base model | `google/flan-t5-base` (250M params) | `configs/inca.yaml` |
| Optimiser | AdamW, `lr = 3.0e-4`, `weight_decay = 0.01` | `configs/inca.yaml` |
| Batch size | 8 × `grad_accum_steps = 4` = effective 32 | `configs/inca.yaml` |
| Epochs per domain | 5 | `configs/inca.yaml` |
| `k_eval` | Every 50 optimiser steps | `configs/inca.yaml` |
| Layers per block | 4 (FLAN-T5-base has 12 encoder layers → max 3 blocks) | `configs/inca.yaml` |
| `n_max_blocks` | 8 | `configs/inca.yaml` |
| `lateral_rank` | 0 (Phase 1; Phase 2 ablation only) | `configs/inca.yaml` |
| Hardware | Apple M4 (MPS) or CUDA | `baselines/_runtime/trainer_factory.py` |
| Precision | bf16 (MPS); falls back to fp32 if torch < 2.6.0 | `baselines/_runtime/precision.py` |
| Seeds | 42, 123, 999 (3 seeds; mean ± std reported) | `configs/ablations/e_route.yaml` |

---

## §5. Ablations

All ablations hold all non-swept hyperparameters fixed at the INCA default
(`configs/inca.yaml`). Each ablation reports mean ± std over 3 seeds (42, 123, 999).

### 5.1 E-ROUTE — Selector type ablation

**Config**: `configs/ablations/e_route.yaml`
**Sweep**: `selector ∈ [embedding_query, uclbr, cross_attention, weighted_sum]` × 3 seeds = 12 runs

| Variant | Short name | Description |
|---|---|---|
| `embedding_query` | S-QKV | Fixed-query cross-attention, token-level (§3.3) — **default** |
| `uclbr` | UCLBR | Pre-gated + load-balanced + uncertainty-calibrated (§3.4) |
| `cross_attention` | S-FULL | MLP gate, sequence-level mean-pool |
| `weighted_sum` | S-WS | Blind scalar per block, input-independent control |

**Questions answered by E-ROUTE**:
1. Does token-level selection (S-QKV) outperform sequence-level (S-FULL)?
2. Does uncertainty-calibrated routing (UCLBR) help when blocks are load-imbalanced?
3. Is the input-independent control (S-WS) a credible ablation lower bound?

**Expected outcome**: S-QKV ≥ UCLBR > S-FULL > S-WS on ACC and BWT.
The hypothesis is that token-level differentiation (S-QKV) captures block specialisation
that sequence-level gating (S-FULL) misses. UCLBR may close the gap when blocks are
highly specialised.

**Figure E1** (2×2 panel):
- Top-left: ACC by selector across domains.
- Top-right: BWT by selector.
- Bottom-left: PAR (param efficiency) by selector.
- Bottom-right: Block weight distribution for S-QKV vs UCLBR vs S-FULL over training.

### 5.2 E-SAT — Saturation threshold sensitivity

**Config**: `configs/ablations/e_sat.yaml`
**Sweep**: `rir_threshold ∈ [0.20, 0.30, 0.40]` × `patience ∈ [3, 5, 8]` × 3 seeds = 27 runs

| Threshold | Patience | Behaviour | Expected bias |
|---|---|---|---|
| 0.20 (low) | 3 (short) | Fires very early; may grow before genuine saturation | Under-grow risk |
| 0.30 (default) | 5 (medium) | Balanced | — |
| 0.40 (high) | 8 (long) | Conservative; only fires on strong signal | Over-training risk |

**Questions answered by E-SAT**:
1. How sensitive is INCA's performance to the RIR threshold?
2. Does patience (loss-plateau window) interact with RIR threshold?
3. What is the optimal (RIR_threshold, patience) pair, and how flat is the surface?

**Expected outcome**: a flat plateau around (0.30, 5), confirming robustness to
threshold choice. The 3×3 grid surface plot is the key figure.

**Figure E2**: heatmap of ACC (or BWT) over the 3×3 `(rir_threshold, patience)` grid,
with a star on the default (0.30, 5) cell.

### 5.3 E-CLS3 — Replay strategy ablation

**Config**: `configs/ablations/e_cls3.yaml`
**Sweep**: `replay_strategy ∈ [uniform, hardest, easiest, schedule]` × 3 seeds = 12 runs

| Strategy | Description | Hypothesis |
|---|---|---|
| `uniform` | All past items equally likely | Weakest replay signal |
| `hardest` | Only highest-loss items | Maximum forgetting prevention, but risk of instability |
| `easiest` | Only lowest-loss items | Stability anchor; minimal anti-forgetting |
| `schedule` | 70% hard, 20% easy, 10% mid (default) | Best of both |

**Questions answered by E-CLS3**:
1. Does the difficulty-weighted replay buffer matter for backward transfer?
2. Is the 70/20/10 schedule better than uniform, and by how much?
3. Can we justify the specific split (or is any non-uniform split sufficient)?

**Expected outcome**: `schedule` ≥ `hardest` > `uniform` > `easiest` on BWT;
`easiest` may have higher stability on the earliest domain (less forgetting on simple items).

**Figure E3**: BWT per domain (line chart) for the four replay strategies, showing
where on the domain sequence each strategy diverges.

### 5.4 E-SCOPE — Lateral adapter rank (Phase 2)

**Config**: new `configs/ablations/e_scope.yaml` (to be written)
**Sweep**: `lateral_rank ∈ [0, 4, 8, 16]` × 3 seeds = 12 runs
**Precondition**: Phase 2 lateral adapter implementation in `models/inca/lateral.py`
is complete and wired into `INCALayerManager`.

| Rank | Description |
|---|---|
| 0 (off) | Phase 1 default (embedding skip only) |
| 4 | Lightweight lateral adapter |
| 8 | Medium lateral adapter |
| 16 | Full-rank lateral adapter (LLaMA-Pro-scale) |

**Questions answered by E-SCOPE**:
1. Do lateral adapters improve accuracy post-expansion vs. embedding-skip-only (rank=0)?
2. What rank is optimal for the parameter budget?
3. Does rank correlate with speed of post-expansion convergence?

**Figure E4**: Accuracy curve (steps) *starting from the most recent grow event* for
each lateral rank, isolating the cold-start recovery speed.

### 5.5 E-TIMING — Expansion timing ablation

**Config**: new `configs/ablations/e_timing.yaml` (to be written)
**Sweep**: `expand_at ∈ [epoch_1 (early), saturation (INCA default), epoch_4 (late), never]`

This ablation is the **most important figure in the paper** — it directly tests the
core INCA claim that *when* you grow matters.

| Timing | Description |
|---|---|
| `early` | Expand after epoch 1 regardless of saturation signal |
| `saturation` (INCA) | Expand when multi-signal detector fires (default) |
| `late` | Expand after epoch 4 regardless |
| `never` (no-grow) | Fixed capacity; never expand (INCA-no-grow baseline) |

**Expected result**: accuracy is concave in expansion timing — both early and late
expansion underperform saturation-timed expansion. `never` is worst on the hardest
domains but may be competitive on easy ones (where saturation fires early anyway).

**Figure E5**: accuracy vs expansion-timing offset across all domains. This is the
single figure that most directly validates the saturation-detector motivation.

---

## §6. Main Results

### 6.1 Table 1 — Main results (ACC / BWT / FWT / PAR)

Proposed layout:

```
Method          | ACC   | BWT    | FWT    | PAR    | EXP_N
─────────────────┼───────┼────────┼────────┼────────┼───────
B1 Naive FT     | 0.XXX | −0.XXX | +0.XXX |   —    |   0
B2 Replay       | 0.XXX | −0.XXX | +0.XXX |   —    |   0
B3 EWC          | 0.XXX | −0.XXX | +0.XXX |   —    |   0
B4 L2P          | 0.XXX | −0.XXX | +0.XXX |   —    |   0
B5 LoRA-MoE     | 0.XXX | −0.XXX | +0.XXX |   —    |   0
B6 LLaMA-Pro†   | 0.XXX | −0.XXX | +0.XXX | 0.XXX  |   n_domains
B7 PNN          | 0.XXX | −0.XXX | +0.XXX |   —    |   n_domains
INCA-no-grow    | 0.XXX | −0.XXX | +0.XXX |   —    |   0
INCA (ours)     | 0.XXX | −0.XXX | +0.XXX | 0.XXX  |   auto
```

† B6 LLaMA-Pro uses fixed expansion (1 block per domain boundary).

**Bold** = best in column. `†` = significant over B6 (p < 0.05, McNemar's test on ACC).

### 6.2 Figure 1 — Accuracy curves over training

Per-domain accuracy over training steps for B1, B6, INCA, INCA-no-grow.
Vertical dashed lines mark domain boundaries.
INCA expansion events marked with upward triangles.

### 6.3 Figure 2 — BWT matrix heatmap

4×4 (or 3×3) matrix R(t, j) for INCA vs B6 side-by-side.
Shows that INCA's off-diagonal cells (earlier-domain accuracy after later training) are
closer to the diagonal than B6's, i.e., less forgetting.

### 6.4 Figure 3 — Parameter efficiency curve

Accuracy vs. total parameters added (millions), for B6 and INCA.
Shows INCA reaching higher ACC with fewer parameters by growing only when needed.

---

## §7. Analysis

### 7.1 Expansion event timeline

When does INCA grow, relative to domain boundaries and loss curves?
- Hypothesis: expansion fires within the first 100-200 steps of a new domain (as the
  loss re-spikes), then not again until genuine saturation in the new domain.
- Figure A1: expansion events (triangles) overlaid on loss curve across all domains.

### 7.2 Block weight trajectories (S-QKV)

How do block weights evolve over training?
- Hypothesis: early blocks are down-weighted as new blocks are added, but they recover
  weight on domains that are similar to the domain on which they were trained.
- Figure A2: block weight (from S-QKV block_scores softmax) per domain step.

### 7.3 CKA trajectory

Plot CKA value (current block vs. reference) over training.
Shows CKA rising as the block saturates → approaching threshold → trigger fires.
- Figure A3: CKA curve with saturation threshold line and expansion event marked.

### 7.4 Grad-norm EMA trajectory

Plot grad-norm EMA and its peak-decay ratio over training.
Shows the decay pattern that co-triggers the block-full event.
- Figure A4: grad-norm EMA curve with decay-threshold line.

---

## §8. Conclusion

**Summary**: INCA grows transformer blocks *when* the current block is saturated,
not on a fixed schedule. The saturation detector combines four complementary signals;
the S-QKV selector aggregates frozen and active blocks at the token level with
drift-resistant identity init. On [DATASET], INCA outperforms LLaMA-Pro (B6) on
backward transfer and parameter efficiency, with ablations confirming that each
component (adaptive timing, token-level selection, hard-example replay) contributes
independently.

**Limitations**:
- Evaluated on FLAN-T5-base (250M). Scaling behaviour on larger models is unexplored.
- Lateral adapters (E-SCOPE) are Phase 2 only; results with them may improve further.
- Domain boundaries are assumed known at training time (not online detection).

**Future work**:
- Paper C: building a temporal benchmark where saturation naturally arises from
  real-world fact drift (see [`paper_c_ideas_journal.md`](paper_c_ideas_journal.md)).
- Paper D: applying INCA to a clean temporal stream, where the expansion events should
  cluster at period boundaries (see [`paper_d_ideas_journal.md`](paper_d_ideas_journal.md)).
- Scaling to LLaMA-2 7B or Mistral 7B for an apple-to-apple LLaMA-Pro comparison at scale.

---

## §9. Appendix

### Appendix A — Full ablation tables

All 27 E-SAT runs (3×3×3 grid), all 12 E-ROUTE runs, all 12 E-CLS3 runs.
Mean ± std over 3 seeds for each cell.

### Appendix B — Expansion event logs

Per-run expansion event table: domain, training step, RIR at event, grad-norm EMA at event,
CKA at event, loss at event.

### Appendix C — Selector architecture diagrams

Formal diagrams for S-QKV, UCLBR, S-FULL, S-WS. Mathematical notation aligned with §3.

### Appendix D — LLaMA-Pro comparison details

LLaMA-Pro (B6) exact configuration used: block positions, init strategy, which layers are
frozen, training sequence. Ensures reproducibility of the baseline.

### Appendix E — Compute budget

Training time per method on M4 Mac (MPS) and on GPU.
INCA overhead vs. B1 (naive FT): extra cost = selector forward pass + saturation detector
evaluation every k_eval steps.

---

## Implementation Status

| Component | File | Status |
|---|---|---|
| Saturation detector (4 signals) | `models/inca/plateau.py` | ✅ Implemented |
| CKA monitor | `models/inca/cka.py` | ✅ Implemented |
| Block-chain manager + grow | `models/inca/layer_manager.py` | ✅ Implemented |
| S-QKV selector | `models/inca/selectors.py` | ✅ Implemented |
| S-FULL selector | `models/inca/selectors.py` | ✅ Implemented |
| S-WS selector | `models/inca/selectors.py` | ✅ Implemented |
| UCLBR selector | `models/inca/uclbr.py` | ✅ Implemented |
| Lateral adapter | `models/inca/lateral.py` | ✅ Implemented (Phase 2, disabled) |
| E-ROUTE ablation config | `configs/ablations/e_route.yaml` | ✅ Wired |
| E-SAT ablation config | `configs/ablations/e_sat.yaml` | ✅ Wired |
| E-CLS3 ablation config | `configs/ablations/e_cls3.yaml` | ✅ Wired |
| E-SCOPE ablation config | `configs/ablations/e_scope.yaml` | 📝 To be written |
| E-TIMING ablation config | `configs/ablations/e_timing.yaml` | 📝 To be written |
| INCA trainer | `training/inca_trainer.py` | ✅ Implemented |
| B1-B7 baselines | `baselines/b1_*.py` ... `b7_*.py` | ✅ Implemented |
| Sweep launcher | `scripts/train_baselines.py` | ✅ Implemented |
| INCA launcher | `scripts/train_inca.py` | ✅ Implemented |
| Domain-sequential dataset | `data/` (loader TBD) | 🔬 Dataset choice pending |

---

## Open Decisions (require Nishant's input)

| Decision | Options | Current lean |
|---|---|---|
| Dataset stack (§4.1) | A: Math+Code (LLaMA-Pro replica), B: Domain sequence, C: Difficulty curriculum | A (tightest comparison) |
| Lateral adapters in main paper? | Yes (Phase 2 wired in), No (E-SCOPE only) | No (Phase 1 paper first) |
| Venue | NeurIPS 2026, ICML 2026, ICLR 2027 | TBD |
| E-TIMING config | Manual early/late thresholds vs. fraction-of-max-patience | TBD |
| Scale-up experiment | FLAN-T5-large or LLaMA-2-7B? | FLAN-T5-large first |

---

## Document Metadata

| Field | Value |
|---|---|
| Author | Nishant Kumar (with Claude assistance) |
| Created | 2026-06-16 |
| Purpose | Full paper plan — sections, experiments, ablations, baselines |
| Status | Working draft — all sections require filling with actual results |
| Ideas journal | [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md) |
| Architecture refs | `docs/CAPSEL_INCA_Master_Reference.pdf`, `docs/CAPSEL_Selector_Architecture.pdf` |

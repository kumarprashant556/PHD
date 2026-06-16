# Paper B — Implementation Brief
### INCA vs LLaMA-Pro: Saturation-Driven Block Expansion on a Domain-Sequential Curriculum

> **Purpose**: Concise implementation plan for converting the existing INCA code to Paper B's
> experimental setup.  All code items require **explicit permission before execution**.
>
> **Status**: Pre-implementation planning doc.  Nothing has been changed yet.
>
> **Related docs**:
> - [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md) — running ideas log
> - [`paper_main_outline.md`](paper_main_outline.md) — full paper structure (§1–§9)
> - [`configs/inca.yaml`](../configs/inca.yaml) — current training config (still on `cc_news`)

---

## 1. What the code already does (no changes needed)

| Component | File | Status |
|---|---|---|
| Block-chain growth | `models/inca/layer_manager.py` | ✅ ready |
| 4-signal saturation detector | `models/inca/plateau.py`, `cka.py` | ✅ ready |
| S-QKV selector | `models/inca/selectors.py` | ✅ ready |
| UCLBR selector | `models/inca/uclbr.py` | ✅ ready |
| Replay buffer | `models/inca/replay.py` | ✅ ready |
| LLaMA-Pro baseline (B6) | `baselines/b6_llama_pro.py` | ✅ ready |
| All other baselines B1-B7 | `baselines/b{1-7}_*.py` | ✅ ready |
| Data contract | `data/_base.py` (`finalise()`) | ✅ ready — all loaders use it |
| Ablation configs | `configs/ablations/e_sat/route/cls3.yaml` | ✅ ready — dataset key only needs changing |
| Training entry | `scripts/train_inca.py`, `training/inca_trainer.py` | ✅ ready |

The entire INCA architecture is Paper B–ready.  The only gaps are (a) the dataset, (b) the
Paper B config, and (c) the memory/efficiency evaluation metrics.

---

## 2. Dataset design

### 2.1 Goal

A **3-domain sequential curriculum** that:
- mirrors LLaMA-Pro's math+code spirit (domain = distinct capability, not temporal slice)
- is small enough to run on a single GPU / M-series Mac in < 6 hours per full sweep
- produces clean BWT/FWT (no leakage hazard — domains don't share answer strings)
- fits the existing `data/_base.py` completion-framing contract (`"complete: " + first_half → second_half`)

### 2.2 Proposed domains and HF sources

| Period | Domain | HF dataset | Why |
|---|---|---|---|
| `P1` | **Math** | `lighteval/MATH` | 12,500 problem+solution pairs; free-text format (problem statement concatenated with step-by-step solution); LLaMA-Pro's headline domain |
| `P2` | **Code** | `bigcode/the-stack-smol` (Python subset) | ~200k Python files; completion framing on function body works naturally; LLaMA-Pro's second domain |
| `P3` | **Science** | `allenai/sciqa` (SciQ) | 13,679 science passages + explanations; adds a third domain for multi-period BWT evidence; free-text format |

**Optional P4** (if 3 domains give inconclusive BWT): `wikipedia` (English, "Category:Science")
as a general-knowledge anchor — establishes baseline forgetting on a broad domain.

### 2.3 Scale (resource-constrained)

| Parameter | Value | Rationale |
|---|---|---|
| `n_per_period` | **2 000** | ~200 MB RAM per period; single MacBook-class run in < 2 h/domain |
| Domains | **3** (P1/P2/P3) | Minimum for BWT table (need ≥ 2 post-P1 periods to compute backward transfer) |
| Seeds | **3** (42, 123, 999) | Standard for Q1 variance reporting |
| Total training items | ~6 000 | Fits comfortably in 8 GB VRAM / 16 GB unified memory |

**Why this is enough for Q1**: the paper's claim is about *architecture* (saturation detector +
adaptive expansion), not about scale.  LLaMA-Pro's own ablations showed that the timing of
block insertion is measurable even at small scales.  With 3 domains × 3 seeds × 8 methods
(B1-B7 + INCA) = 72 runs, the table has the density expected for a systems/CL paper.

### 2.4 Format mapping (completion framing)

The existing `data/_base.py` `apply_completion()` takes a single `text` string and splits
it at `split_frac = 0.50`.  For each source:

| Domain | `text` construction |
|---|---|
| Math | `problem + "\n\n" + solution` (both fields in `lighteval/MATH`) |
| Code | raw Python file content (docstring + function body) |
| Science | `support + " " + question + " " + answer` from SciQ |

After `finalise()` these become standard `{input_text, target_text, period}` rows — no change
to any downstream code.

### 2.5 New file needed

**`data/domain_sequential.py`** — one new loader following the `_base.py` contract.
Returns `{"P1_math": Dataset, "P2_code": Dataset, "P3_science": Dataset}`.

⚠️ **Permission required before writing this file.**

---

## 3. Configuration changes

### 3.1 New primary config: `configs/paper_b.yaml`

Inherits everything from `configs/inca.yaml`; overrides:

```yaml
dataset:          "domain_sequential"   # new loader
n_per_period:     2000
max_periods:      3
epochs_per_period: 3                    # fewer epochs needed (small data saturates faster)
batch_size:       8
grad_accum_steps: 4                     # effective batch 32 — unchanged
k_eval:           25                    # check saturation more frequently (small dataset)
patience:         3                     # plateau patience tighter (fewer steps per epoch)
min_epochs_before_grow: 1              # allow growth after 1 epoch (not 2) — small data
out_dir: "results/paper_b"
```

### 3.2 Ablation config updates

Each ablation config in `configs/ablations/` has `dataset: cc_news` hardcoded.
That one key needs updating to `domain_sequential` and `n_periods: 3`.

Files to update: `e_sat.yaml`, `e_route.yaml`, `e_cls3.yaml`.

⚠️ **Permission required before editing these files.**

### 3.3 LLaMA-Pro (B6) baseline config: `configs/baselines/b6_paper_b.yaml`

B6 already works (`baselines/b6_llama_pro.py` — zero-init copy of last block per period,
freeze old, train new).  For Paper B the comparison must be apples-to-apples:

```yaml
# configs/baselines/b6_paper_b.yaml
dataset:          "domain_sequential"
n_per_period:     2000
max_periods:      3
epochs_per_period: 3
batch_size:       8
grad_accum_steps: 4
initial_trainable_blocks: 1            # LLaMA-Pro: exactly 1 new block per period boundary
seed: 42
out_dir: "results/paper_b/b6"
```

⚠️ **Permission required before creating this file.**

---

## 4. Memory and efficiency evaluation

This is the key differentiator from a pure ACC/BWT paper.  The claim: INCA adds capacity
**only when needed**, so it reaches comparable accuracy to LLaMA-Pro with fewer parameters
and lower peak memory.

### 4.1 New metrics (beyond ACC / BWT / FWT already in the outline)

| Metric | Symbol | Formula / collection point | What it shows |
|---|---|---|---|
| **Peak training memory** | `MEM_TRAIN` | `torch.cuda.max_memory_allocated()` after each period | INCA vs LLaMA-Pro memory cost per domain |
| **Inference memory** | `MEM_INFER` | `torch.cuda.memory_allocated()` on a single forward pass | Static memory after all domains |
| **Total parameters added** | `PAR` | `sum(p.numel() for p in model.parameters() if p.requires_grad)` diff per period | Already in outline — collect per period |
| **Memory efficiency** | `ACC/MB` | `delta_ACC_period / MEM_TRAIN_period` | Accuracy gain per MB of peak memory |
| **Training wall time** | `TIME_PER_PERIOD` | `time.perf_counter()` around `trainer.train()` | Speed comparison |
| **Blocks used** | `EXP_N` | Already tracked by `layer_manager.py` | How many grow events fired vs LLaMA-Pro's fixed 1/period |

### 4.2 Expected story

| Metric | INCA (expected) | LLaMA-Pro / B6 (expected) | Advantage |
|---|---|---|---|
| ACC | ≥ B6 (tied or +1-2%) | Fixed expansion | — |
| BWT | Less negative than B6 (saturation guard prevents over-fitting) | Fixed expansion at boundary | INCA |
| PAR | Fewer blocks on easy domains (no grow event if not needed) | Always +1 block/period | INCA |
| MEM_TRAIN | Lower on easy domains (no new block allocated) | Always allocates new block | INCA |
| ACC/MB | Higher | Baseline | INCA — **headline efficiency figure** |
| EXP_N | ≤ n_periods (some periods may not saturate) | Always = n_periods | INCA |

This produces **Figure 2** in the paper: ACC/MB scatter plot with methods as points.
INCA should sit in the upper-left (high accuracy, low memory cost).

### 4.3 New file needed

**`training/memory_tracker.py`** — lightweight wrapper that logs per-period:
- `peak_train_mb`: peak GPU/MPS memory during `trainer.train()`
- `infer_mb`: model static memory on a dummy forward pass
- `param_delta`: trainable params added vs previous period
- `wall_time_s`: training wall time in seconds
- Writes to `results/paper_b/<run_id>/memory_log.json`

The wrapper is called from `training/inca_trainer.py` (and from each baseline's `main()`).

⚠️ **Permission required before writing this file.**

---

## 5. LLaMA-Pro comparison details

### 5.1 What B6 currently does (code audit)

From `baselines/b6_llama_pro.py`:
- Period 0: freeze all params; unfreeze last `initial_trainable_blocks=1` encoder blocks.
- Period N > 0: `deepcopy` last block → zero output projections → `append` to encoder block
  list → freeze all → unfreeze new last block.
- Trains with `standard_trainer` (HF `Seq2SeqTrainer`).

This is a faithful LLaMA-Pro reproduction at the FLAN-T5 scale.  **No change needed to the
B6 logic** — it is already correct for Paper B.

### 5.2 Scale argument (reviewer pre-emption)

LLaMA-Pro used LLaMA-2-7B; INCA uses FLAN-T5-base (250M).  The scale difference is a
known reviewer concern.  Pre-emption strategy (to log in `paper_b_ideas_journal.md`):

- Report the **block expansion ratio** (blocks added / total blocks) rather than raw block
  count.  For FLAN-T5-base with 3 blocks and 1 added: ratio = 33%.  For LLaMA-2-7B with 32
  layers and 8 added: ratio = 25%.  Comparable.
- Cite LLaMA-Pro §5.4 (ablation on smaller models) if it exists.
- Frame INCA as "the CL method"; LLaMA-Pro as "the fixed-schedule expansion paradigm" — scale
  is a deployment variable, not the primary comparison axis.

---

## 6. Ordered code items (all need permission)

| # | Item | File | Estimated size | Depends on |
|---|---|---|---|---|
| 1 | Domain-sequential loader | `data/domain_sequential.py` | ~120 lines | Nothing |
| 2 | Register loader in registry | `data/__init__.py` (3-line addition) | 3 lines | Item 1 |
| 3 | Paper B primary config | `configs/paper_b.yaml` | ~40 lines | Item 1 |
| 4 | LLaMA-Pro baseline config | `configs/baselines/b6_paper_b.yaml` | ~15 lines | Item 3 |
| 5 | Update ablation configs | `configs/ablations/e_{sat,route,cls3}.yaml` | 2 lines each | Item 3 |
| 6 | Memory tracker | `training/memory_tracker.py` | ~80 lines | Nothing |
| 7 | Wire memory tracker into INCA trainer | `training/inca_trainer.py` (5-line hook) | 5 lines | Item 6 |
| 8 | Wire memory tracker into B6 | `baselines/b6_llama_pro.py` (5-line hook) | 5 lines | Item 6 |
| 9 | E-SCOPE ablation config | `configs/ablations/e_scope.yaml` | ~15 lines | Item 3 |
| 10 | E-TIMING ablation config | `configs/ablations/e_timing.yaml` | ~15 lines | Item 3 |

**Start order**: 1 → 2 → 3 → 4 → 5 (dataset + config pipeline first, so smoke tests
are possible before touching any model code).

Items 6–8 (memory tracker) are independent and can be written in parallel with 1–5.
Items 9–10 (ablation configs) are the final step; they produce "THE most important figures."

---

## 7. Open decisions (not blocking item 1)

| Decision | Options | Recommendation |
|---|---|---|
| Domain 4 (optional) | Add Wikipedia general / law / medicine as P4 | Add only if 3-domain BWT is inconclusive |
| SciQ vs SciDocs | SciQ (simpler, text passages) vs AllenAI S2ORC (larger, richer) | SciQ — smaller, cleaner text |
| Memory tracker platform | MPS (Mac) vs CUDA (GPU server) | Keep both code paths; MPS uses `torch.mps.current_allocated_memory()` |
| Lateral adapters in main table | Include E-SCOPE (rank>0) runs in Table 1 vs appendix only | Appendix-only for paper 1; main table stays B1-B7+INCA |
| Venue | ACL 2027 / EMNLP 2026 / ICLR 2027 | Decide after seeing E-TIMING results |

---

## Document metadata

| Field | Value |
|---|---|
| Author | Nishant Kumar (with Claude assistance) |
| Created | 2026-06-16 |
| Purpose | Actionable implementation brief for Paper B experiments |
| Rule | All code items require Nishant's explicit approval before execution |
| Related | `paper_b_ideas_journal.md`, `paper_main_outline.md`, `configs/inca.yaml` |

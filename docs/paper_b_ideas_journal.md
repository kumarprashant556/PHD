# Main Paper — Ideas Journal
### INCA: Saturation-Driven Incremental Block Expansion for Continual Domain Adaptation

> **This is the main paper journal.** "Paper A" (temporal CL framing) is superseded — see
> [`paper_a_ideas_journal.md`](paper_a_ideas_journal.md) for the archived entries.
> The full paper plan is in [`paper_main_outline.md`](paper_main_outline.md).
>
> A running log of ideas, framings, experiments, and reviewer-defences for the **main
> CAPSEL/INCA paper**: saturation-driven block-expansion evaluated on a domain-specific
> (non-temporal) dataset, with LLaMA-Pro as the primary baseline.  See
> [`docs/CAPSEL_INCA_Master_Reference.pdf`](CAPSEL_INCA_Master_Reference.pdf),
> [`docs/CAPSEL_Implementation_Guide.pdf`](CAPSEL_Implementation_Guide.pdf), and
> [`docs/CAPSEL_Selector_Architecture.pdf`](CAPSEL_Selector_Architecture.pdf) for the
> architecture spec.
>
> **Reverse-chronological** (newest entry at top).  Append-only — mark stale entries
> `Status: Rejected` or `Status: Superseded`, never delete.

---

## Scope — what this paper is and is not

| This paper IS                                               | This paper is NOT                                     |
|-------------------------------------------------------------|-------------------------------------------------------|
| Architecture: saturation detection → block expansion        | A temporal-drift / forgetting benchmark paper         |
| Selector ablation: RIR vs grad-norm vs CKA vs plateau       | A dataset construction paper                          |
| Domain-specific dataset (math + code, or similar)           | Dependent on temporal stream or BWT matrix            |
| Direct comparison to LLaMA-Pro (Wu et al., 2024)            | A temporal CL survey                                  |
| Block init strategy experiments                             | Paper A (which uses CC-News / TiC-LM for BWT claims) |
| Plasticity–stability tradeoff on saturation signal          | Paper C (temporal dataset) or Paper D (temporal CL)   |

**Relationship to other papers:**
- Paper A uses INCA on a temporal stream → BWT/FWT headline.
- Paper B isolates the *architecture* claim from the temporal framing.
- Paper C builds the clean temporal dataset Paper A and D need.
- Paper D applies CL methods (B1-B7 + INCA) to Paper C's dataset.

---

## Conventions

Same template as Paper A's journal:

```
### YYYY-MM-DD · [Source] · short title
- **Source**: User-instructed | Claude-proposed (approved YYYY-MM-DD) | Joint-session
- **Status**: Open | Approved-for-paper | In-draft | Rejected | Superseded | Deferred
- **Where it lands**: §X.Y of Paper B, or "Appendix", or "Thesis Ch. N", or "Decision only"
- **Tags**: #architecture #experiment #ablation #framing #reviewer-defence #figure
- **Why it matters** (1-2 sentences)
- **Body** (free-form)
```

### Source rules
- **User-instructed** — Nishant told me; goes in immediately.
- **Claude-proposed** — I noticed something; I ask first; added only with explicit approval.
- **Joint-session** — emerged from conversation; both parties agreed to log it.

### Tag glossary

| Tag                 | Use when …                                                               |
|---------------------|--------------------------------------------------------------------------|
| `#architecture`     | The idea is about INCA/CAPSEL's growth mechanism itself.                 |
| `#ablation`         | The idea is a specific ablation (selector type, init strategy, etc.).    |
| `#experiment`       | A full experiment — dataset, training run, metric.                       |
| `#framing`          | Changes the paper's narrative or contribution claim.                     |
| `#reviewer-defence` | Pre-empts a specific anticipated reviewer objection.                     |
| `#related-work`     | A paper/result to cite or position against.                              |
| `#figure`           | A specific figure or table that should appear in Paper B.                |
| `#dataset`          | Dataset choice or preprocessing decision.                                |

---

## Entries

<!-- Newest entry goes immediately below this comment. -->

### 2026-06-16 · User-instructed · Establish Paper B ideas journal
- **Source**: User-instructed
- **Status**: Approved-for-paper (this file is the artefact)
- **Where it lands**: Process artefact — not in paper
- **Tags**: #architecture
- **Why it matters**: Centralises Paper B's idea pool so architecture and ablation decisions
  don't get mixed with Paper A's temporal CL decisions.
- **Body**:
  - Paper B is the clean capacity-paper: INCA architecture + selector ablations on a
    domain-specific dataset, no temporal framing required.
  - Claude does **not** add Claude-proposed entries without Nishant's explicit approval.
  - See sibling journals: [`paper_a_ideas_journal.md`](paper_a_ideas_journal.md),
    [`paper_c_ideas_journal.md`](paper_c_ideas_journal.md),
    [`paper_d_ideas_journal.md`](paper_d_ideas_journal.md).

### 2026-06-16 · User-instructed · Use a domain-specific (non-temporal) dataset stack following LLaMA-Pro
- **Source**: User-instructed
- **Status**: Open — dataset stack needs finalisation
- **Where it lands**: Paper B §3 (Experimental Setup)
- **Tags**: #dataset #framing #reviewer-defence
- **Why it matters**: Decouples the architecture claim from temporal framing. If the headline
  is "saturation-driven capacity growth," the dataset only needs to saturate the base model and
  admit a clean train/eval split — a temporal axis is unnecessary and introduces leakage hazard.
  LLaMA-Pro (Wu et al., 2024) used math+code on exactly this argument.
- **Body**:
  - **Candidate stacks** (in order of preference):
    1. **Math + Code** (LLaMA-Pro replica): GSM8K training split (math reasoning) +
       CodeSearchNet (code understanding). This maximises the direct LLaMA-Pro comparison.
    2. **Domain sequence** (sequential domain shift): science → law → medicine using
       domain-specific subsets of RedPajama or The Pile. Each domain = one "period."
    3. **Difficulty curriculum** (hardness-driven saturation): easy → hard within a single
       domain (e.g., GSM8K grade-school → competition math). Growth triggered by loss plateau
       within difficulty tier.
  - **Recommendation**: Stack 1 (math + code) is the tightest LLaMA-Pro comparison. Stack 2
    is the strongest "we generalise beyond math" story. Stack 3 is most original but hardest
    to baseline.
  - **Key advantage over temporal**: no probe-answer leakage hazard; BWT is clean by
    construction (math answers don't repeat across domains/difficulties); direct comparison to
    LLaMA-Pro §4 possible.
  - **Action**: check LLaMA-Pro §4 for exact dataset sizes and saturation behaviour before
    committing to Stack 1.
  - **Cross-link**: research-domain framing logged at
    [`research_ideas_journal.md`](research_ideas_journal.md)
    `2026-06-16 · Should CAPSEL use LLaMA-Pro's dataset stack`.

### 2026-06-16 · User-instructed · Selector ablation: e_sat vs e_route vs e_cls3
- **Source**: User-instructed
- **Status**: Open — configs exist; needs training runs
- **Where it lands**: Paper B §4 (Ablations); Figure B1
- **Tags**: #ablation #experiment #figure
- **Why it matters**: The selector is CAPSEL/INCA's core novelty — it decides *when* and
  *where* to grow. The ablation quantifies how much each signal type contributes.
- **Body**:
  - **Three configs already wired** (see `configs/ablations/`):
    - `e_sat.yaml` — saturation-based selector (RIR + grad-norm EMA + loss plateau): the
      proposed CAPSEL mechanism.
    - `e_route.yaml` — routing-based selector: growth decision from a learned gating module.
    - `e_cls3.yaml` — 3-class classifier selector: binary "grow / don't grow" replaced by
      "grow / freeze / shrink."
  - **What to measure per selector**:
    - Block-expansion events: how many, at which layers, at which training steps.
    - Final task accuracy vs. a fixed-capacity (no-growth) baseline.
    - Parameter efficiency: accuracy per parameter added.
    - Training stability: loss curve smoothness, grad-norm behaviour.
  - **Expected result**: e_sat should trigger growth at the right moment (after saturation,
    before divergence); e_route may over-grow; e_cls3 may under-grow if the classifier
    trains slowly.
  - **Reviewer defence**: "why not just train a bigger model from scratch?" → parameter
    efficiency plot (Figure B1b): INCA reaches same accuracy with fewer total parameters
    because blocks are added only where needed.
  - **Figure plan**:
    - B1a: Loss curves for e_sat / e_route / e_cls3 with expansion events marked.
    - B1b: Accuracy vs total parameters for INCA variants vs LLaMA-Pro vs fixed-capacity
      FLAN-T5-base.

### 2026-06-16 · User-instructed · LLaMA-Pro as primary baseline and comparison target
- **Source**: User-instructed
- **Status**: Open — needs LLaMA-Pro §4 read + reproduction estimate
- **Where it lands**: Paper B §2 (Related Work) + §5 (Main Results, Table B1)
- **Tags**: #related-work #experiment #reviewer-defence
- **Why it matters**: LLaMA-Pro (Wu et al., 2024) is the closest published analogue — block
  expansion for domain adaptation, non-temporal dataset. A direct comparison is the strongest
  possible positioning for Paper B.
- **Body**:
  - **What LLaMA-Pro did**: froze all original LLaMA-2 layers, added identity-initialised
    transformer blocks interleaved at fixed positions, fine-tuned new blocks only on
    math+code. No saturation signal — expansion is fixed at paper-design time.
  - **What CAPSEL/INCA adds**: saturation-driven expansion timing (not fixed), selector
    ablation (not a single strategy), lateral connections between old and new blocks.
  - **Comparison axes**:
    - Accuracy on held-out math/code tasks.
    - Parameter efficiency (accuracy per parameter added).
    - Training stability (loss curve shape).
    - Catastrophic forgetting on *general* tasks (does adding math blocks hurt general NLU?).
  - **Potential issue**: LLaMA-Pro used LLaMA-2-7B; INCA is FLAN-T5-base (250M). Scale
    difference is a reviewer concern — pre-empt with a "same relative block expansion
    ratio" argument and cite LLaMA-Pro's own ablation on smaller scales if available.
  - **Action**: read LLaMA-Pro §4 to extract: exact dataset sizes, block positions, init
    strategy (identity init?), evaluation tasks, and whether they report per-param efficiency.

### 2026-06-16 · User-instructed · Block expansion timing as the headline claim
- **Source**: User-instructed
- **Status**: Open — depends on ablation results
- **Where it lands**: Paper B §1 (Introduction) + §4 (Results)
- **Tags**: #architecture #framing #figure
- **Why it matters**: The core claim that differentiates CAPSEL from LLaMA-Pro and
  progressive-networks ancestors is *when* blocks are added. Saturate-then-grow is the
  hypothesis — early-growth hurts (model hasn't saturated current capacity), late-growth
  wastes compute. This claim needs an experiment to land.
- **Body**:
  - **The experiment**: train INCA with expansion triggered at step T for T ∈ {early, on-time
    (saturation-detected), late, never}. Plot final accuracy + training efficiency vs T.
    "On-time" is defined by the saturation detector; "early" and "late" are fixed offsets.
  - **Saturation detector signals** (from `models/inca/`):
    - `plateau.py` — loss plateau slope (Δloss over a window).
    - `cka.py` — CKA drift between representation snapshots (stabilisation = saturation).
    - `selectors.py` — multi-signal selector combining RIR, grad-norm EMA, and plateau.
    - `layer_manager.py` — expansion decision and new-block insertion.
  - **Expected shape**: accuracy is concave in T — early and late expansion both underperform
    saturation-timed expansion.
  - **Figure plan**: B2 — accuracy curve vs expansion timing offset (T_early, T_sat, T_late,
    T_never). This is the single most important figure in Paper B.
  - **Related claim**: `lateral.py` connections let new blocks share representations with
    frozen old blocks immediately post-expansion — this is the init strategy that makes
    early saturation detection useful (blocks can start contributing right away).

### 2026-06-16 · User-instructed · Lateral connections as the init strategy that makes growth safe
- **Source**: User-instructed
- **Status**: Open
- **Where it lands**: Paper B §3.2 (Architecture — Block Expansion); Appendix B
- **Tags**: #architecture #experiment #reviewer-defence
- **Why it matters**: A naive identity-init new block (LLaMA-Pro's approach) takes many
  steps to leave the identity regime. Lateral connections (`models/inca/lateral.py`)
  provide a gradient path from the frozen representation immediately, reducing the "cold
  start" penalty of a newly added block.
- **Body**:
  - **Ablation**: lateral on vs. lateral off, holding all other settings fixed. Expected
    benefit: faster convergence post-expansion + better early-expansion accuracy.
  - **Connection to LLaMA-Pro**: they freeze old layers and fine-tune only new ones. INCA's
    lateral connections let frozen-layer representations inform new-block gradients — a
    structural advantage worth calling out explicitly.
  - **Reviewer concern**: "lateral connections add parameters on top of the block expansion."
    Pre-empt: report parameter counts with and without laterals; show the accuracy-per-param
    curve improves with laterals.

### 2026-06-16 · User-instructed · UCLBR regularisation as the stability mechanism
- **Source**: User-instructed
- **Status**: Open — needs ablation
- **Where it lands**: Paper B §3.3 (Architecture — Stability Mechanism); Table B2 ablation
- **Tags**: #architecture #ablation
- **Why it matters**: `models/inca/uclbr.py` implements the continual learning regulariser
  that prevents previously-frozen blocks from drifting when new blocks are added. It is the
  "stability" half of the plasticity–stability tradeoff and needs its own ablation.
- **Body**:
  - **Ablation**: UCLBR on vs. off, same dataset and expansion timing. Metric: performance
    on the *pre-expansion* task after growth (forgetting signal).
  - **Connection to EWC (B3)**: UCLBR is a structured alternative to EWC. The ablation
    table should include a row for B3-EWC as the prior-art regulariser baseline.
  - **Expected result**: UCLBR > no-reg > EWC on the stability metric, because UCLBR is
    designed for the post-expansion regime specifically (not for sequential fine-tuning).

---

## Document metadata

| Field        | Value                                                                        |
|--------------|------------------------------------------------------------------------------|
| Author       | Nishant Kumar (with Claude assistance)                                       |
| Created      | 2026-06-16                                                                   |
| Purpose      | Running log of Paper B ideas — capacity-driven architecture + ablations      |
| Append rule  | New entries go at the top of "Entries"; never delete, only mark status       |
| Owner        | Nishant — all Claude-proposed entries require explicit approval before adding |
| Sibling docs | [`paper_a_ideas_journal.md`](paper_a_ideas_journal.md) — temporal CL paper   |
|              | [`paper_c_ideas_journal.md`](paper_c_ideas_journal.md) — temporal dataset    |
|              | [`paper_d_ideas_journal.md`](paper_d_ideas_journal.md) — temporal CL on wiki |
|              | [`research_ideas_journal.md`](research_ideas_journal.md) — domain-level      |

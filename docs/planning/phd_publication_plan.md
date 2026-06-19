# CAPSEL / INCA — PhD Publication Plan
### Three Q1 Papers from One Unified System

**Author:** Nishant Kumar  
**Last updated:** 2026-06-19  
**System:** CAPSEL/INCA — Saturation-driven continually-growing transformer LM  
**Master reference:** `docs/CAPSEL_INCA_Master_Reference.pdf`  
**Task list:** `docs/TASKS.md`

---

## The System in One Paragraph

INCA grows a transformer encoder one block at a time. When the current trainable block has genuinely exhausted its learning capacity — detected by a four-signal consensus inside the model — that block is frozen and a warm-started copy becomes the new trainable block. Because saturation-driven growth deliberately allows each block to absorb **multiple domains before freezing**, a block is not a single-domain expert but a compressed multi-period memory. A token-level router (S-QKV) then retrieves the right period's knowledge from each block at inference time. A per-block replay buffer prevents within-block forgetting while a period is still being trained. When the chain grows long enough, representationally redundant blocks are merged using a certified algorithm. The whole system is grounded in Complementary Learning Systems (CLS) theory from neuroscience.

---

## The Story Arc Across All Three Papers

```
Paper 1:  "When to grow, and how to retrieve from a multi-period block"
          └─ Core architecture · timing · S-QKV routing · replay · domain-sequential

Paper 2:  "How to route with calibrated uncertainty over a growing chain on a real temporal stream"
          └─ UCLBR · lateral adapters · CLS experiments · TiC-LM

Paper 3:  "How to stay bounded: principled merging with a certified routing error guarantee"
          └─ Block merging · TIES/SLERP · error bound theorem · Fisher pruning
```

Each paper builds directly on the previous. Each is self-contained. No contribution is duplicated.

---

---

# Paper 1

## Title

**Saturation-Driven Block Expansion for Continual Domain Adaptation: When to Grow and How to Retrieve**

---

## Central Claim

> Saturation-driven block expansion timing outperforms fixed-schedule expansion (LLaMA-Pro), and token-level routing is the **necessary and sufficient** condition for retrieving domain-specific knowledge from multi-period blocks that fixed-schedule architectures never create.

Two claims locked together:
1. **Timing claim:** Growing when the model says it is ready (multi-signal consensus) outperforms growing on a fixed schedule.
2. **Routing claim:** Because saturation-driven growth allows one block to absorb multiple domains, retrieval from that block requires token-level routing. Sequence-level routing fails in exactly this setting.

The claims need each other. Claim 1 creates multi-period blocks on purpose. Claim 2 explains how to read from them. LLaMA-Pro (one block per period, fixed schedule) never faces the retrieval problem because no block is ever multi-period. INCA creates the problem and solves it in the same design.

---

## Core Argument / Framing

**The framing against LLaMA-Pro:**

LLaMA-Pro argues that adding identity-initialised blocks improves domain adaptation. It grows on a fixed schedule: one block per period, always, regardless of whether the current block is saturated. This wastes parameters on easy periods and underfits hard ones.

INCA's counter-argument:
> *"The right question is not how many blocks to add, but when. A block trained past saturation accumulates noise, not knowledge. A block grown before saturation discards the warm-start advantage of its predecessor. Saturation-driven timing produces blocks that are full without being over-trained — and because we grow less often, each block absorbs more domains, making parameter efficiency a natural consequence of the design."*

**The routing framing — two orthogonal retrieval problems:**

Fixed-schedule expansion (LLaMA-Pro) enforces one block per period, so no block is ever multi-period and no period ever spans multiple blocks. Routing is trivially simple: one block, one domain. Saturation-driven growth abandons this assumption entirely, and in doing so creates two distinct retrieval challenges that must be solved simultaneously:

*Scenario A — Multi-period block (small or medium datasets):*
Block capacity exceeds the per-period data volume. One block absorbs P1 then P2 before saturating. At inference, a math query must retrieve math knowledge from a block that also contains code knowledge.

*Scenario B — Multi-block period (large datasets):*
Per-period data volume exceeds block capacity. A single large domain (e.g. full MetaMath, 395K examples) saturates Block 0 mid-period; Block 1 continues on the remaining examples of the same domain. At inference, a math query must choose between Block 0's math representations and Block 1's math representations.

Crucially, Scenario B does not produce two redundant copies of math knowledge. Because saturation-driven growth fires when Block 0 can no longer improve on its current training data, **Block 1 necessarily trained on the examples that Block 0 found hardest**. The blocks partition the domain by difficulty level — emergently and automatically, with no explicit difficulty signal. Block 0 learned the easy math; Block 1 learned the residually harder math conditioned on Block 0's representations through the chain.

> *"In both scenarios, the routing problem cannot be solved by sequence-level routing (one scalar per block per sequence). Sequence-level routing produces one routing weight for the entire input regardless of which tokens drive the query — a complex math query and a trivial math query receive identical block weights if they are the same length. S-QKV's token-level mechanism is the architectural choice that solves both cases with the same mechanism: Q = frozen original embeddings captures both domain type (math tokens vs code tokens) and query complexity (simple math notation vs dense chain-of-thought notation). K and V are computed fresh on the current query — the block's own self-attention activates domain-relevant and complexity-appropriate patterns first, and S-QKV then attends over those already-filtered representations."*

**The emergent difficulty routing prediction:**

A falsifiable consequence of the multi-block period argument: in a large-dataset curriculum, S-QKV routing weights should correlate with question difficulty across blocks — Block 0 receives higher weight for questions below its saturation threshold, Block 1 for harder questions — without any explicit difficulty label during training or inference. This is testable by sorting MetaMath questions by difficulty tier and plotting mean routing weight to each block against difficulty.

**The unified routing table:**

| Scenario | Cause | Retrieval problem | S-QKV signal |
|---|---|---|---|
| Multi-period block | Small dataset, block absorbs multiple domains before saturating | Which domain within this block? | Domain-different Q: math tokens ≠ code tokens in frozen embedding space |
| Multi-block period | Large dataset, one domain saturates multiple blocks | Which difficulty stratum within this domain? | Complexity-different Q: simple math ≠ dense proof notation in frozen embedding space |
| Mixed (realistic) | Large multi-domain curriculum | Both simultaneously | Q encodes both domain type and query complexity; handles both with no modification |

Sequence-level routing (S-FULL, S-WS) collapses the sequence dimension before routing, so neither domain type nor query complexity is visible to the router. E-ROUTE is the experiment that proves this collapse causes accuracy degradation in both scenarios — not just one.

**The biological framing:**

Per-block study-schedule replay maps onto hippocampal CLS theory. Frozen block weights = neocortical long-term memory. Replay buffer = hippocampal short-term episodic memory. Buffer cleared at freeze = hippocampal pruning after consolidation. This framing is backed by E-CLS3 (study-schedule vs uniform) and appears as a §2 discussion, not a separate paper.

---

## Contributions

| # | Contribution | Type |
|---|---|---|
| C1 | Multi-signal saturation consensus (RIR + GradNorm EMA + CKA + loss plateau) as the block expansion trigger | Architecture / method |
| C2 | Empirical proof that saturation-driven timing outperforms early, late, and fixed-schedule expansion | Experiment |
| C3 | Identification of two orthogonal retrieval problems created by saturation-driven growth: (a) multi-period blocks where one block absorbs multiple domains, (b) multi-block periods where large datasets cause one domain to span multiple blocks with emergent difficulty stratification | Problem statement |
| C4 | Proof that token-level routing (S-QKV) is necessary and sufficient for both retrieval problems; sequence-level routing (S-FULL, S-WS) fails in both cases because it collapses the token signal that encodes domain type and query complexity | Architecture + ablation |
| C4a | Emergent difficulty-aware routing: in multi-block periods, S-QKV routing weights correlate with question difficulty across blocks without any explicit difficulty label — a falsifiable prediction tested by routing analysis on difficulty-stratified MetaMath | Experiment |
| C5 | Per-block study-schedule replay (CLS-grounded): two-phase uniform → hard/easy schedule, cleared at freeze, bounded buffer | Method |
| C6 | Growth primitive comparison (G-VERT / G-HORIZ / G-EXPERT) with function-preserving init for G-VERT | Architecture + ablation |

---

## What This Paper Covers

### Dataset
- Domain-sequential 5-period curriculum: MetaMath (P1) → Evol-CodeAlpaca (P2) → SciQ (P3) → MedMCQA (P4) → CommonsenseQA (P5)
- Same math and code data LLaMA-Pro used in SFT → dataset-controlled comparison
- FLAN-T5-large (780 M parameters)
- Metric: token-level F1 (unified across all 5 domains; SQuAD-style; non-binary → smooth RIR signal)

### Baselines
- **Primary:** B6 LLaMA-Pro (fixed-schedule vertical expansion) — the paper-to-beat
- **Secondary:** B1 naive fine-tuning · B2 replay-only · B3 EWC · B4 L2P · B5 LoRA-MoE · B7 PNN
- **Bounds:** frozen FLAN-T5-large (forward-transfer floor) · joint training on all periods (upper bound)

### Main Results Table
Rows: B1–B7 + INCA. Columns: P1–P5 token F1 · Avg-F1 · BWT · FWT · PAR(M) added.  
Key finding: INCA highest Avg-F1, lowest |BWT|, fewest parameters added on average across 3 seeds.

---

## Ablations

| ID | Config | What it tests | Why it matters |
|---|---|---|---|
| **E-TIMING** | `e_timing.yaml` ✅ | early vs saturation vs late vs never | **Headline figure** — directly proves timing claim |
| **E-ROUTE** | `e_route.yaml` ✅ | S-QKV vs S-FULL vs S-WS (vs UCLBR deferred to P2) | **Proves routing claim** — run on both small-dataset (multi-period block) and large-dataset (multi-block period) configs; S-FULL should fail in both, S-QKV succeeds in both |
| **E-PRIM** | `e_prim.yaml` ✅ | G-EXPERT vs G-VERT vs G-HORIZ | Function-preserving init advantage |
| **E-SAT** | `e_sat.yaml` ✅ | rir_threshold × patience sensitivity | Detector robustness |
| **E-CLS3** | `e_cls3.yaml` ✅ | study-schedule vs uniform vs hardest-only | Replay strategy CLS justification |

**Key experiment design for E-ROUTE:**  
Run S-QKV and S-FULL/S-WS on the same trained checkpoints. On single-period blocks (early in training): performance similar — no period ambiguity. On multi-period blocks (after first saturation event): S-QKV >> S-FULL >> S-WS. This pattern directly validates C3 and C4.

---

## Figures Plan

| Figure | Content | Claim it supports |
|---|---|---|
| Figure 1 | Bar chart: INCA vs B6 on each domain + Avg-F1 (hero figure, analog of LLaMA-Pro Fig 7) | C1, C2 |
| Figure 2 | **E-TIMING:** Avg-F1 vs expansion timing (early / saturation / late / never) — the concave curve | C1, C2 |
| Figure 3 | **E-ROUTE:** routing method vs accuracy across three regimes: single-period blocks (baseline) · multi-period blocks (small dataset) · multi-block periods (large dataset). S-FULL degrades in both multi-X regimes; S-QKV holds in all three. | C3, C4 |
| Figure 3b | **Emergent difficulty routing:** mean S-QKV routing weight per block plotted against MetaMath question difficulty tier (easy → competition). Block 0 weight peaks on easy; Block 1 weight peaks on hard — with no explicit difficulty signal. | C4a |
| Figure 4 | F1 curves over training with saturation events marked (↑ arrows) + CKA of frozen blocks per period | C1, C5 |
| Figure 5 | Efficiency scatter: Avg-F1 vs parameters added; blob size = peak memory | C1 |

---

## What Is Needed to Complete This Paper

**Code** (already done):
- INCA trainer with multi-signal saturation detector ✅
- S-QKV, S-FULL, S-WS selectors ✅
- G-EXPERT, G-VERT, G-HORIZ primitives ✅
- Study-schedule replay, T1.3 drift check ✅
- B6 LLaMA-Pro baseline ✅
- All 5 ablation configs ✅

**Before first run:**
- Fix `chance: 0.25` for MCQ periods (P4_medical, P5_commonsense) in `configs/paper_b.yaml`

**Experiments to run:**
- Main sweep: INCA vs B1–B7 + B6 · 3 seeds each · 5 periods
- E-TIMING · E-ROUTE · E-PRIM · E-SAT · E-CLS3 ablation sweeps

**Writing:** ~20 pages; no new systems required.

---

## What This Paper Does NOT Need

- SPRT (patience-based check is sufficient to demonstrate timing)
- Fisher pruning
- UCLBR (S-QKV is sufficient for Paper 1; UCLBR is Paper 2's contribution)
- Temporal stream data (domain-sequential is self-contained)
- Block merging

---

## Target Venue and Timeline

**Target:** TMLR (Transactions on Machine Learning Research) — rolling submission, no deadline pressure, Q1-equivalent, accepts strong empirical papers without a required theorem. Fallback: Neural Networks (Elsevier) or IEEE TNNLS.

**Timeline:** 4–6 months from first experiment run.

---

---

# Paper 2

## Title

**CAPSEL: Uncertainty-Calibrated Growing Expert Chains for Temporal Continual Learning**

---

## Central Claim

> Over a growing chain of frozen expert blocks on a real temporal stream, uncertainty-calibrated load-balanced routing (UCLBR) with low-rank lateral adapters produces better forward and backward transfer than standard attention routing — and the per-block replay mechanism mirrors hippocampal consolidation in measurable, quantified ways.

---

## Core Argument / Framing

Paper 1 proved **when to grow** and showed S-QKV works for within-block retrieval on a controlled domain-sequential benchmark. Paper 2 asks: **how to route robustly when you do not know which period a query belongs to, when periods overlap, and when the stream is temporal rather than domain-sequential?**

**The routing problem at scale:**

On a temporal stream (TiC-LM: 12+ monthly Common Crawl slices), period boundaries are not clean. A query about an event may partially match several frozen blocks. S-QKV routes correctly when the Q signal is strong (clearly math vs. clearly code). On a temporal stream, queries often match multiple blocks equally — the router needs to express *uncertainty* and fall back gracefully toward equal weighting rather than confidently routing to the wrong block.

> *"S-QKV is a strong router when period representations are distant. On temporal streams where adjacent periods are similar (same domain, different month), S-QKV's confidence is miscalibrated — it routes confidently but wrongly. UCLBR adds three components to address this: a Read-ME pre-gate that filters irrelevant blocks before attention, a DeepSeek-style load-balance bias that prevents routing collapse onto a single block, and an uncertainty-calibrated fallback that detects when the router is confused and backs off toward equal weighting."*

**The lateral adapter framing:**

Paper 1 showed that S-QKV reads from the block's query-conditioned output. But the trainable block's input is still only the previous block's output through the chain. On long chains, the chain deepens representation drift. Lateral adapters give the trainable block a **direct, low-rank path from any frozen block** — bypassing the chain distortion. This is the parameter-light equivalent of a direct synapse from neocortex to the current trainable region.

**The CLS biological grounding:**

Paper 2 runs the full CLS experimental program (E-CLS1–E-CLS5) that Paper 1 alludes to but does not have space to validate:
- E-CLS1: Does removing replay hurt? By how much, across buffer sizes?
- E-CLS2: Do frozen block representations monotonically specialise? (linear probe per block per period)
- E-CLS4: Does replay loss drop before validation loss rises? (consolidation dynamics)
- E-CLS5: Does forgetting follow a power law across frozen blocks? (forgetting curves)

These experiments transform the biological analogy from a framing device into a measurable, falsifiable claim.

---

## Contributions

| # | Contribution | Type |
|---|---|---|
| C1 | UCLBR: three-component routing — Read-ME pre-gate + DeepSeek aux-loss-free load balance + uncertainty calibration | Architecture / method |
| C2 | SPRT-based saturation detection replacing fixed patience (α=0.05, β=0.10; thresholds A≈2.89, B≈−2.94) | Method / theory |
| C3 | Lateral adapters (G-LAT): rank-r cross-block knowledge transfer, α_k=0 init (function-preserving) | Architecture |
| C4 | Empirical CLS validation: E-CLS1–E-CLS5 prove the biological analogy is measurable, not metaphorical | Experiment |
| C5 | Dataset-agnostic saturation detection: same config fires correctly across RealtimeQA (~30% F1), StreamingQA (~40%), TemporalWiki (~50%) | Experiment |
| C6 | Full temporal stream evaluation on TiC-LM (12 periods) vs B1–B8 + SEEKR + Online-LoRA | Experiment |

---

## What This Paper Covers

### Dataset
- **Primary:** TiC-LM — 12 monthly Common Crawl slices (2.9T tokens; Li et al. ACL 2025 Oral)
- **Secondary:** TRACE · TemporalWiki · RealtimeQA · StreamingQA
- **Same model:** FLAN-T5-large (780M) for direct continuity with Paper 1

### Baselines
- All B1–B8 (including B8 BlockStack) on TiC-LM
- **Modern secondaries:** SEEKR (EMNLP 2024) · Online-LoRA (WACV 2025) · DER++ (NeurIPS 2020)
- **Bounds:** frozen pretrained (FT floor) · joint training (upper bound)

---

## Ablations

| ID | Config | What it tests |
|---|---|---|
| **E-ROUTE (full)** | `e_route.yaml` ✅ | S-QKV vs S-FULL vs S-WS vs UCLBR on temporal stream |
| **E-SCOPE** | `e_scope.yaml` ✅ | Lateral adapter rank r ∈ {0, 4, 8, 16} |
| **E-SIG** | `e_sig.yaml` ❌ needs config | 4 signals ablated independently + in pairs |
| **E-SAT-AGNOSTIC** | `e_sat_agnostic.yaml` ❌ needs config | Same config, 3 different temporal datasets |
| **E-CLS1** | `e_cls1.yaml` ❌ needs config | Replay off vs buffer sizes {100, 500, 2000, all} |
| **E-CLS2** | `e_cls2.yaml` ❌ needs config | Linear probe per frozen block per period |
| **E-CLS4** | `e_cls4.yaml` ❌ needs config | Consolidation dynamics (replay loss vs val loss timing) |
| **E-CLS5** | `e_cls5.yaml` ❌ needs config | Forgetting curves: power-law fit on BWT per frozen block |
| **E-SCALE** | `e_scale.yaml` ❌ needs config | FLAN-T5-large (Track A) + Pythia-160M/GPT-2-Medium (Track B) |

---

## What Is Needed to Complete This Paper

**New code:**
- SPRT accumulator in `models/inca/plateau.py` (P1.3)
- `models/inca/growth_chooser.py` — SPRT margin + CKA drift → primitive selection (P2.3)
- Probe scripts: `scripts/probe_frozen_blocks.py` · `scripts/forgetting_curves.py` (P3.1)
- Per-epoch replay loss logging in `training/inca_trainer.py` for E-CLS4
- B8 BlockStack baseline `baselines/b8_block_stack.py`
- SEEKR + Online-LoRA + DER++ modern baselines

**New configs:** e_sig · e_sat_agnostic · e_cls1 · e_cls2 · e_cls4 · e_cls5 · e_scale

**Data:** TiC-LM pipeline (already partially set up in `data/tic_lm.py`)

**Timeline:** 12–18 months from Paper 1 submission.

**Target:** NeurIPS / ICLR / JMLR

---

## What Paper 2 Cites from Paper 1

Paper 2 cites Paper 1 for: INCA architecture · multi-signal saturation consensus · study-schedule replay · S-QKV token-level routing · growth primitive design. Paper 2's contributions are entirely additive: UCLBR routing, lateral adapters, SPRT, temporal evaluation, and the CLS experimental program.

---

---

# Paper 3

## Title

**Bounded Continual Growth: Principled Block Merging with Certified Post-Merge Routing Error**

---

## Central Claim

> A continually-growing block chain can be compressed at regular intervals by merging representationally redundant frozen blocks using a three-stage algorithm (TIES geometric merge → distillation refinement → router recalibration), and the degradation in routing quality after merging is bounded by a certifiable function of the pre-merge CKA similarity between the merged blocks.

---

## Core Argument / Framing

Papers 1 and 2 establish that INCA grows to fit the data. Left unconstrained, the chain grows without bound — a practical deployment problem and a theoretical unsatisfying result. Paper 3 asks: **can you bound the chain length without degrading performance, and can you certify how much damage the compression causes?**

**The merging argument:**

If two frozen blocks become representationally redundant (CKA ≥ 0.90), they have converged to similar knowledge. Keeping both wastes parameters and makes routing harder (two similar blocks compete in the selector, causing load-balance drift). Merging them reduces chain length while preserving the combined knowledge — IF the merge is done correctly and the router is recalibrated afterward.

> *"Standard knowledge distillation merges two models without regard for routing. We show this is insufficient for a growing chain because the router's internal state is calibrated to the pre-merge block positions. We propose a three-stage merge that first geometrically merges block weights (TIES), then distills the merged block to recover lost knowledge, then recalibrates the router on a short held-out stream pass. We prove that the output distribution shift from this procedure is bounded by a constant depending on the pre-merge CKA similarity and the merge quality."*

**The theoretical angle:**

The post-merge error bound is the paper's unique theoretical contribution. Assumptions: router R is L-Lipschitz in feature embedding; merged block B_m satisfies ‖B_m(x) − (B_i(x)+B_j(x))/2‖ ≤ ε_m; CKA(B_i, B_j) ≥ 1−ε_c. Result: router output distribution after recalibration differs from pre-merge by at most a constant × (ε_m + ε_c) × L. This bound:
1. Gives a principled criterion for WHEN to merge (τ_merge = 0.90 is not arbitrary — it sets ε_c ≤ 0.10)
2. Justifies the three-stage procedure (each stage reduces ε_m)
3. Provides a falsifiable prediction (BWT loss ≤ 2 pp empirically if the bound is tight)

**The Fisher pruning connection:**

Before freezing a block, prune its lowest-Fisher-score neurons and attention heads. This is not just parameter efficiency — it pre-conditions the block for future merging by removing redundant capacity before it accumulates. Fisher pruning at freeze + CKA-triggered merging are two sides of the same bounded-chain principle.

---

## Contributions

| # | Contribution | Type |
|---|---|---|
| C1 | Three-stage block merging: TIES geometric merge + distillation refinement + router recalibration | Algorithm |
| C2 | Post-merge router error bound theorem: error ≤ f(ε_m + ε_c) × L | Theory |
| C3 | Fisher-based structured pruning before freeze: removes lowest-Fisher heads + MLP neurons, conditions block for future merging | Method |
| C4 | E-MERGE: merge cadence K ∈ {2,3,5} — chain length vs BWT tradeoff | Experiment |
| C5 | E-TAU: merge threshold τ ∈ {0.85, 0.90, 0.95} — compression vs accuracy | Experiment |
| C6 | E-PRUNE: Fisher vs magnitude pruning; p ∈ {0,10,20,30}%; sweet spot 10–20% | Experiment |
| C7 | Long-horizon comparison (24 periods, Pythia-160M) vs SEEKR · D-MoLE · LLaMA-MoE · CL-MoE | Experiment |

---

## What This Paper Covers

### Dataset
- **Primary:** TiC-LM Track B — 24 monthly periods (longer than Paper 2's 12), Pythia-160M
- **Secondary:** TRACE (8-task benchmark, instruction-following degradation)
- Long horizon is essential: merging only demonstrates its value when the chain would otherwise grow to 8+ blocks

### Baselines
- Unbounded INCA (Paper 1 + 2) — the "no-merge" upper bound
- **Modern secondaries (all needed here):** SEEKR · D-MoLE · Online-LoRA · InfLoRA · MIGU · LLaMA-MoE · CL-MoE

---

## Ablations

| ID | Config | What it tests |
|---|---|---|
| **E-MERGE** | needs config | Merge cadence K ∈ {2,3,5}: accuracy drop vs chain length |
| **E-TAU** | needs config | Merge threshold τ ∈ {0.85, 0.90, 0.95}: compression vs BWT |
| **E-PRUNE** | `e_prune.yaml` ❌ needs config | Fisher vs magnitude pruning; p sweep |

---

## What Is Needed to Complete This Paper

**New code:**
- `models/inca/block_merge.py` — TIES trim-and-elect + SLERP fallback + distillation + router recalibration (P5.1)
- `diagonal_fisher()` · `prune_heads()` · `prune_neurons()` in layer manager (P1.4)
- Modern secondary baselines B9+ (SEEKR, D-MoLE, etc.) (P5.0)
- Written proof of post-merge error bound (P5.2)

**New configs:** e_merge · e_tau · e_prune

**Timeline:** 20–30 months from now (Year 2 per roadmap).

**Target:** ICML / NeurIPS (theory + experiments track) / JMLR

---

---

## Cross-Paper Dependency Map

```
Paper 1 (done when experiments run)
├── Establishes: INCA architecture · timing · S-QKV routing · study-schedule replay
├── Proves: timing claim · routing claim on domain-sequential
└── Cited by: Papers 2 and 3 for the architecture and timing mechanism

Paper 2 (builds on P1 architecture, new routing + temporal evaluation)
├── Requires from P1: INCA architecture, S-QKV as baseline routing
├── Adds: UCLBR · SPRT · lateral adapters · CLS validation · TiC-LM evaluation
└── Cited by: Paper 3 for UCLBR routing component

Paper 3 (builds on P1+P2 system, adds merging + theory)
├── Requires from P1: block chain architecture, growth primitives
├── Requires from P2: UCLBR routing, temporal evaluation pipeline
├── Adds: block merging algorithm · error bound · Fisher pruning · long-horizon eval
└── Terminal paper — feeds directly into thesis Ch 5
```

---

## What Each Paper Needs That the Others Don't

| Component | Paper 1 | Paper 2 | Paper 3 |
|---|---|---|---|
| Multi-signal saturation consensus | ✅ core | cited | cited |
| Growth primitives (G-VERT/HORIZ/EXPERT) | ✅ ablated | cited | cited |
| Study-schedule replay (per-block CLS) | ✅ ablated | cited, extended | cited |
| S-QKV token-level routing | ✅ core + proven | cited as baseline | cited |
| UCLBR routing | ❌ | ✅ core | cited |
| SPRT saturation detection | ❌ | ✅ core | cited |
| Lateral adapters (G-LAT) | ❌ | ✅ ablated | cited |
| CLS probe experiments (E-CLS1–5) | ❌ | ✅ core | cited |
| TiC-LM temporal stream | ❌ | ✅ core | extended (24p) |
| Block merging algorithm | ❌ | ❌ | ✅ core |
| Post-merge error bound theorem | ❌ | ❌ | ✅ core |
| Fisher pruning | ❌ | ❌ | ✅ ablated |
| Modern secondary baselines (SEEKR etc.) | ❌ | partial | ✅ full |

---

## Timeline

| Milestone | Target |
|---|---|
| Paper 1 experiments complete | Month 4–5 |
| Paper 1 submitted (TMLR rolling) | Month 5–6 |
| Paper 2 code complete | Month 10–12 |
| Paper 2 experiments complete | Month 13–15 |
| Paper 2 submitted (NeurIPS / ICLR) | Month 15–18 |
| Paper 3 code complete | Month 20–22 |
| Paper 3 submitted (ICML / JMLR) | Month 24–28 |
| Thesis submitted | Month 32–36 |

---

## The Unifying Thesis Statement

> A continually-learning language model should allocate architectural capacity as a principled, multi-signal decision — not a schedule — growing when a consensus of calibrated signals declares its current capacity exhausted, sharing knowledge across frozen experts through function-preserving lateral adapters and token-level uncertainty-calibrated routing, replaying memories under a biologically-grounded schedule, and bounding its own size through principled merging of representationally-redundant experts with certified routing error guarantees.

Paper 1 covers the first clause. Paper 2 covers the middle. Paper 3 covers the last.

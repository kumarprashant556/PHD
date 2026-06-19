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

### 2026-06-19 · User-instructed · CAPSEL master reference gap audit — full implementation status check
- **Source**: User-instructed (cross-referenced against `docs/CAPSEL_INCA_Master_Reference.pdf`)
- **Status**: Open — gaps logged here and in `docs/TASKS.md`
- **Where it lands**: Decision only — drives implementation priority
- **Tags**: #architecture #ablation #experiment
- **Why it matters**: First full audit against the master reference since code stabilised. Identifies
  what is code-complete, what has a config but no code, and what is entirely absent.

---

#### DONE — matches spec ✅

| Spec section | What's implemented |
|---|---|
| §3.1–3.2 Architecture | Sequential chain + embedding skip + identity inter-block projections |
| §3.3 Growth primitives | G-EXPERT ✅, G-VERT ✅, G-HORIZ ✅ (Net2Net), G-LAT ✅ (lateral.py) — all four |
| §4.2 Multi-signal consensus | RIR + GradNorm EMA + CKA + loss plateau — all four signals |
| §4.3 Grokking guard (epoch) | `min_epochs_before_grow` — epoch-count guard only (MI part absent — see gaps) |
| §5 Replay / CLS | Per-block buffer, `clear_all()` at every freeze, two-phase study-schedule sampling |
| §5.5 T1.3 drift check | Period-transition drift check wired in trainer |
| §6 Selector variants | S-QKV, UCLBR, S-FULL, S-WS — all four |
| §10.1 Baselines B1–B7 | Code present for all seven |
| Paper B pipeline | INCA trainer + B6 LLaMA-Pro + domain_sequential + 4 configs (MPS + CUDA) |
| Ablation configs | `e_sat`, `e_route`, `e_scope`, `e_cls3`, `e_timing`, `e_prim` — all runnable |
| `--expand_at` (E-TIMING) | Wired in `scripts/train_inca.py` |

---

#### GAP 1 — Config bug (fix before any Paper B run) 🔴

**`chance` not set for MCQ domains in `configs/paper_b.yaml`**

The spec (§14 key numbers) says `chance = 0.25` for 4-way MCQ tasks because
RIR = (score − score_0) / max(score_0, **chance**). With `chance = 0.0` (current default),
the RIR denominator for P4_medical (medmcqa, 4-way MCQ) is wrong — a model scoring
25% by chance will report non-zero RIR, causing BLOCK_FULL to fire spuriously or not at all.

P3_science (sciq) is also 4-choice MCQ. P5_commonsense (commonsense_qa) is 5-choice.
Fix: add `chance: 0.25` (or 0.20 for P5) per-period, or expose as a period-level config key.

---

#### GAP 2 — §4.3 Grokking guard — MI part absent 🟡

The spec requires a second condition alongside epoch count:
> `MI_t − MI_0 ≥ δ_I (~0.05 nats)` estimated via MINE.

Only the epoch-count branch is implemented. The MINE estimator and MI progress floor are
completely absent. Until MINE is added, the grokking guard is a pure epoch count (correct
in spirit but not spec-faithful). Relevant for E-GROK ablation.

---

#### GAP 3 — §4.4 SPRT extension not implemented 🟡

Completely absent from `models/inca/plateau.py`. The spec defines:
- Λ_t = Σ log[p1(s_τ) / p0(s_τ)] — log-ratio accumulator over the 4-signal vector
- Declare H1 (block full) if Λ_t ≥ A ≈ 2.89; H0 (capacity remains) if Λ_t ≤ B ≈ −2.94

Currently, the multi-signal consensus uses a hard threshold rule, not SPRT. TASKS.md P1.3.

---

#### GAP 4 — §3.3 / §4.4 Growth primitive auto-selector absent 🟡

The spec says the SPRT margin + frozen-block CKA drift *select which primitive fires*
(G-VERT for large margin + low drift, G-HORIZ for large margin + high drift, G-LAT for
small-but-persistent margin). Right now `growth_primitive` is a fixed config value with no
auto-selection logic. `models/inca/growth_chooser.py` does not exist. TASKS.md P2.3.

---

#### GAP 5 — §7 Fisher-based structured pruning absent 🟡

Completely absent. The spec says at every freeze event:
1. Compute diagonal Fisher over the replay buffer on the current block
2. Prune bottom p% of MLP neurons and attention heads by Fisher score (structured, not unstructured)

No `diagonal_fisher()`, `prune_heads()`, or `prune_neurons()` anywhere in the codebase.
E-PRUNE ablation config (`configs/ablations/e_prune.yaml`) also missing. TASKS.md P1.4 / P2.7.

---

#### GAP 6 — §9 Block merging (Paper B proper) absent 🔵

Intentionally deferred to Year 2 per roadmap (§13). But listed here for completeness:
TIES/SLERP geometric merge, distillation refinement, router recalibration, post-merge
error bound theorem — all unimplemented. `models/inca/block_merge.py` does not exist.
TASKS.md P5.1.

---

#### GAP 7 — B8 BlockStack baseline absent 🟡

TASKS.md P0.1 lists B8 by name but `baselines/b8_*.py` does not exist. B8 is described
in §10.1 as "frozen stack + trainable current + pluggable selector — Paper A internal
reference." Should be added before the Phase 0 baseline sweep.

---

#### GAP 8 — §10.2 Modern secondary baselines not implemented 🟢

SEEKR, Online-LoRA, InfLoRA, D-MoLE, MIGU, DER++, LLaMA-MoE, CL-MoE — none present.
These are needed for Phase 5 E-TIC-B comparison (§10.2 says "for 2026 review"). Low
priority until Paper A is submitted. No TASKS.md entry yet.

---

#### GAP 9 — 10 ablation configs missing 🟡

The following ablation configs have no `configs/ablations/` file yet:

| Config | Spec ref | What it tests |
|---|---|---|
| `e_prune.yaml` | §8 E-PRUNE | Fisher vs magnitude; p ∈ {0,10,20,30}% |
| `e_trans.yaml` | §8 E-TRANS | T1.3 drift check with vs without |
| `e_grok.yaml` | §8 E-GROK | Grokking guard on modular arithmetic |
| `e_scale.yaml` | §8 E-SCALE | FLAN-T5-large (Track A) + Pythia-160M/GPT-2-Medium (Track B) |
| `e_sig.yaml` | §8 E-SIG | Each of 4 signals ablated independently + in pairs |
| `e_sat_agnostic.yaml` | §8 E-SAT-AGNOSTIC | Same config on RealtimeQA / StreamingQA / TemporalWiki |
| `e_cls1.yaml` | §8 E-CLS1 | Replay necessity — buffer sizes {100, 500, 2000, all} |
| `e_cls2.yaml` | §8 E-CLS2 | Linear probing frozen blocks per period |
| `e_cls4.yaml` | §8 E-CLS4 | Consolidation dynamics (per-epoch train/val/replay loss + CKA) |
| `e_cls5.yaml` | §8 E-CLS5 | Forgetting curves — power-law fit on BWT per frozen block |

Note: `e_cls3.yaml` ✅ exists.

---

#### GAP 10 — Probe and analysis scripts missing 🟡

For the CLS experiments to run, these scripts are needed:
- `scripts/probe_frozen_blocks.py` — linear probing on each frozen block's mean-pooled reps (E-CLS2)
- Forgetting curve fitting script (power-law fit on BWT per block over time) (E-CLS5)
- Consolidation dynamics logging (per-epoch replay loss + CKA alongside train/val loss) (E-CLS4)

TASKS.md P3.1 covers `evaluation/probes.py` but the dedicated scripts are not listed.

---

#### Summary table

| Gap | Severity | Already in TASKS.md? | Phase |
|---|---|---|---|
| `chance: 0.25` for MCQ domains | 🔴 Fix before run | No | Paper B config |
| SPRT implementation | 🟡 | Yes — P1.3 | Phase 1 |
| Fisher pruning | 🟡 | Yes — P1.4 | Phase 1 |
| Grokking guard (MI part) | 🟡 | Partially — P3.5 | Phase 3 |
| `growth_chooser.py` (auto-select) | 🟡 | Yes — P2.3 | Phase 2 |
| 10 missing ablation configs | 🟡 | Partially | Phase 2–4 |
| Probe / analysis scripts | 🟡 | Partially — P3.1 | Phase 3 |
| B8 BlockStack baseline | 🟡 | Name only — P0.1 | Phase 0 |
| Modern baselines (B9+) | 🟢 | No | Phase 5 |
| Block merging (§9) | 🔵 | Yes — P5.1 | Phase 5 |

### 2026-06-17 · User-instructed · LLaMA-Pro §4 full read → INCA Paper B experimental design decisions
- **Source**: User-instructed (full §4 of Wu et al. 2024 reviewed)
- **Status**: Approved-for-paper — these decisions drive the Paper B experimental setup
- **Where it lands**: Paper B §4 (Experiments) — structure, figures, ablations, framing
- **Tags**: #experiment #ablation #framing #figure #reviewer-defence #related-work
- **Why it matters**: Reading §4 end-to-end reveals exactly which decisions LLaMA-Pro made,
  which ones we must mirror for fair comparison, and which gaps INCA fills.

---

#### A. What LLaMA-Pro's §4 actually does (concise mapping)

| LLaMA-Pro §4 section | What they did | What it means for INCA |
|---|---|---|
| §4.1 pretrain | Stack-Dedup Python + Proof-Pile-2 (80B tokens, 16 H800 GPUs) | We cannot replicate this scale. Our sequential fine-tuning IS our "pretraining." |
| §4.1 SFT | MetaMath + Evol-CodeAlpaca + ShareGPT + WizardLM + SlimOrca (~1M samples) | Use MetaMath (P1) + Evol-CodeAlpaca (P2) — same data, directly comparable |
| §4.1 eval | LM-Eval-Harness (general), BigCode Eval (code, pass@1 greedy) | We use token-level F1 — see 2026-06-17 metrics entry |
| Table 1 | Pretrained + SFT comparisons across 9 benchmarks | Our Table 1: B1–B7 + INCA across 5 domains, token F1 + BWT/FWT |
| Figure 4 | Code avg vs language avg scatter, blob = tokens trained | Our Figure 2: ACC vs peak memory scatter, blob = params added |
| Figure 5 | Training loss curves per block count | Our Figure 3a: F1 + loss curves per period with saturation events marked |
| Figure 6 | Token distribution shift (92.6% unshifted) | Our Figure 3b: CKA drift of frozen blocks across periods |
| Figure 7 | Same SFT data: LLaMA-Pro > LLaMA2-7B on all tasks | Our **main claim figure**: same data, INCA > B6 (fixed schedule) |
| §4.5 Table 5 | Block count / LoRA / MoE / fine-tuning / stacking ablation | Our E-TIMING ablation table (early/sat/late/never) |

---

#### B. Dataset decisions (confirmed)

- **P1 → MetaMath** (`meta-math/MetaMath`, Yu et al. 2023, 395K):
  Same dataset LLaMA-Pro used in SFT. Full CoT solutions. No n_per_period constraint.
  Directly comparable: "using the same math data, INCA achieves X."

- **P2 → Evol-CodeAlpaca** (`theblackcat1992/evol-codealpaca-v1`, ~111K):
  Same dataset LLaMA-Pro used in SFT. Instruction → Python code. Wide difficulty range.
  Directly comparable: "using the same code data, INCA achieves Y."

- **P3–P5 → sciq / medmcqa / commonsense_qa** (INCA-original):
  Not in LLaMA-Pro. This is where INCA generalises *beyond* LLaMA-Pro's math+code scope.
  Narrative: *"We extend beyond LLaMA-Pro's two-domain setting to a 5-domain curriculum,
  demonstrating that saturation-driven growth generalises to diverse domain sequences."*

- **TriviaQA stays as fallback** in case MetaMath implementation blocks a training run.

---

#### C. Experimental structure for Paper B §4

```
§4.1  Experimental Settings
       Model: FLAN-T5-large (780M), 24 enc + 24 dec layers, max 8 blocks
       Dataset: 5-domain sequential (MetaMath → Evol-CodeAlpaca → SciQ → MedMCQA → CommonsenseQA)
       Hyperparams: lr=3e-4, batch 4×8=32 effective, 3 epochs/period, n_per_period=8000
       Metric: token-level F1 (SQuAD-style), unified across all domains
       Seeds: 42 / 123 / 999 (3 seeds × 8 methods = 24 INCA runs + baseline runs)

§4.2  Main Results (Table 1)
       Rows: B1 FT-all / B2 FT-frozen / B3 EWC / B4 ER / B5 ProgressNet / B6 LLaMA-Pro /
             B7 PackNet / INCA (ours)
       Columns: P1-Math / P2-Code / P3-Sci / P4-Med / P5-CS / Avg-F1 / BWT / FWT / PAR(M) / ACC/MB
       Key finding: INCA highest Avg-F1 with lowest |BWT| and fewest PAR(M) on average

§4.3  Efficiency Analysis (Figure 2 — analog of their Figure 4)
       X-axis: peak training memory (MB)   Y-axis: Avg F1 across 5 domains
       Points: B1–B7 + INCA (labeled). INCA should be upper-left (high F1, low memory).
       Bubble size: total parameters added across all periods.

§4.4  Ablation Study (Table 3 — analog of their Table 5)
       E-TIMING: expand_at ∈ {early, saturation, late, never}   [analog: their N-blocks]
       E-ROUTE:  selector ∈ {embedding_query, sigmoid_gate, threshold_only}
       E-SAT:    rir_threshold ∈ {0.20, 0.30, 0.40}
       Also include: LoRA row, FT row (cite their Table 5 as motivation for including these)

§4.5  Representation Analysis (Figure 3 — analog of their Figures 5+6)
       3a: F1 curves over training steps, saturation events marked (↑ arrow)
       3b: CKA of frozen blocks before/after grow events (analog of their Figure 6)
           Expected: ≥ 0.95 CKA = "general capability preserved" across domains
```

---

#### D. The single most important figure: INCA analog of LLaMA-Pro Figure 7

LLaMA-Pro Figure 7 argument:
> "Same SFT data applied to LLaMA2-7B and LLaMA-Pro. LLaMA-Pro consistently outperforms
> LLaMA2-7B (+8.56 HumanEval, +10.2 MBPP). Proves block expansion encodes domain knowledge
> during pretraining — it's not just more parameters."

INCA's version:
> "Same domain data applied to B6 (fixed-schedule expansion, 1 block/period) and INCA
> (saturation-driven expansion). INCA consistently outperforms B6 across all 5 domains.
> Proves that saturation-driven timing encodes more domain knowledge than a fixed schedule —
> it's not just about whether to add blocks, but *when*."

This figure is Paper B's Figure 1 (or the hero figure in §1 Introduction). Numbers come
directly from Table 1. No extra experiment needed — it's a bar chart of Table 1 subset.

---

#### E. Reviewer defences drawn from LLaMA-Pro §4

| Anticipated reviewer objection | Drawn from LLaMA-Pro §4 | INCA response |
|---|---|---|
| "Why not just LoRA?" | §4.5: "LoRA struggles to model distribution of new domain" | Same finding in our B4 row. Cite LLaMA-Pro + our own ablation. |
| "Why not full fine-tuning?" | §4.5: "full fine-tuning results in more significant drop in general performance" | B1 row shows same BWT degradation. Cite LLaMA-Pro. |
| "Why not MoE?" | §4.5: "MoE comparable to 4-block expansion, but with more parameters" | B5 (ProgressNet, MoE-style) should underperform INCA on ACC/MB. |
| "Why interleaved not top?" | §4.5: top-stacking gives lower domain specialisation | We use interleaved by design; cite LLaMA-Pro for justification. |
| "8 blocks seems arbitrary" | §4.5: 8 is optimal; 16≈8 (diminishing returns) | INCA finds the right count adaptively (0–8). `n_max_blocks=8` is a ceiling, not a target. |
| "Scale difference (780M vs 8.3B) unfair" | LLaMA-Pro §4.1: 7B scale, 80B tokens | Block expansion ratio argument: INCA 4 layers/24 = 17% depth increase ≈ LLaMA-Pro 8/32 = 25%. Comparable architecture intervention at different compute budgets. |

---

#### F. What INCA Paper B does that LLaMA-Pro §4 does NOT

1. **Explicit BWT/FWT matrix** — LLaMA-Pro measures forgetting only implicitly via
   "general perplexity barely changes." We report per-domain BWT numbers.

2. **Sequential domains** — LLaMA-Pro trains math+code simultaneously. We train them
   sequentially with period boundaries. This is a harder and more realistic CL scenario.

3. **Saturation measurement** — LLaMA-Pro never measures whether blocks were actually
   saturated. We report: median EXP_T (steps to saturation), EXP_N (blocks grown),
   BLOCK_FULL vs EXHAUSTED rate.

4. **5 domains** — LLaMA-Pro: 2 (math + code). INCA: 5 (math, code, science, medical,
   commonsense). Demonstrates generalisation of the architecture beyond math/code.

5. **Adaptive block count** — LLaMA-Pro fixes 8 blocks. INCA's count is 0–8 per run
   depending on saturation. Report mean ± std of blocks grown per seed.

### 2026-06-17 · User-instructed · Dataset upgrade: MetaMath (P1) + Evol-CodeAlpaca (P2) from LLaMA-Pro's SFT corpus
- **Source**: User-instructed
- **Status**: Open — evaluate against current P1_trivia / P2_code before switching
- **Where it lands**: Paper B §3 (Experimental Setup — Dataset Stack); Table footnote (dataset provenance)
- **Tags**: #dataset #experiment #framing #reviewer-defence
- **Why it matters**: LLaMA-Pro's SFT corpus includes MetaMath (math) and Evol-CodeAlpaca
  (code) — the very datasets that drove their GSM8K score of 78.4 and MATH score of 30.3.
  Using the same datasets for Paper B's P1 and P2 makes the INCA vs LLaMA-Pro comparison
  maximally direct and pre-empts the reviewer objection "different training data, unfair comparison."
- **Body**:
  - **MetaMath** (Yu et al., 2023 — `meta-math/MetaMath`):
    - 395,000 examples — no n_per_period constraint (vs. GSM8K's 7,473 cap of ~7,000)
    - Augmented from GSM8K + MATH competition problems via: rephrasing, self-verification,
      backward reasoning, FOBAR (filling-in-the-blank). Each source problem yields multiple
      reformulations with full chain-of-thought solutions.
    - Format: `"query": <question>  "response": <CoT reasoning> #### <answer>` — same
      `####` delimiter as raw GSM8K → `_load_gsm8k`-style stripping of `<<>>` annotations
      applies directly (minor rename of column keys needed).
    - LLaMA-Pro SFT result: **GSM8K 78.4**, MATH 30.3 (surpassing Mistral-7B at 77.7 / 28.2)
    - Why better than raw GSM8K for our purposes: 53× more examples → smoother saturation
      curves, more reliable RIR signal, BLOCK_FULL fires cleanly.
    - Why better than competition_math: full CoT targets (median ~60 words) → non-binary
      token F1 from step 1; RIR is non-zero.
  - **Evol-CodeAlpaca** (`theblackcat1992/evol-codealpaca-v1` or `nickrosh/Evol-Instruct-Code-80k-v1`):
    - ~111,272 code instruction pairs (or 80k depending on variant)
    - Generated via WizardCoder's Evol-Instruct pipeline: seed code problems evolved to
      increasing difficulty/complexity. Wide diversity of Python tasks.
    - Format: instruction → code solution — maps directly to our `input_text / target_text`
      structure used in `_load_p2_code`.
    - Larger than flytech/python-codes-25k (25k) and more diverse.
    - LLaMA-Pro used this in SFT → using it in Paper B means our P2 domain is drawn from
      the same distribution LLaMA-Pro trained on.
  - **Positioning argument for the paper:**
    > *"To enable a direct dataset-controlled comparison with LLaMA-Pro (Wu et al., 2024),
    > we adopt MetaMath (Yu et al., 2023) as our math domain (P1) and Evol-CodeAlpaca
    > as our code domain (P2) — the same instruction-tuning datasets used in LLaMA-Pro's
    > SFT stage. The key difference is that LLaMA-Pro applies these in a single combined
    > fine-tuning step with a fixed block schedule, while INCA presents them as sequential
    > periods with saturation-driven block expansion."*
  - **Tradeoffs vs current config:**
    | | Current | Proposed |
    |---|---|---|
    | P1 | TriviaQA (138K, factual QA) | MetaMath (395K, math CoT) |
    | P2 | flytech/python-codes-25k (25K) | Evol-CodeAlpaca (~80–111K) |
    | LLaMA-Pro comparability | Indirect | **Direct** |
    | Dataset loader change | — | New `_load_metamath`, update `_load_p2_code` |
    | n_per_period constraint | None | None (both well above 8,000) |
  - **Action items (pending decision):**
    1. Check MetaMath HF card for train split size and column names (`query`/`response`).
    2. Verify Evol-CodeAlpaca variant (80k vs 111k) — prefer the one with instruction +
       clean Python solution columns.
    3. Implement `_load_metamath` following `_load_gsm8k` pattern (strip `<<>>`, full CoT).
    4. Update `DEFAULT_PERIODS` and `configs/paper_b.yaml` if switching P1/P2.
    5. Keep TriviaQA and flytech loaders intact as fallback (do not delete).
  - **Note on remaining periods**: P3_science, P4_medical, P5_commonsense are INCA-original
    (not drawn from LLaMA-Pro). This is intentional — we demonstrate that INCA's saturation
    detector generalises beyond math+code to a broader domain sequence.

### 2026-06-17 · User-instructed · Metric choice: token-level F1 as the unified Paper B metric
- **Source**: User-instructed (from LLaMA-Pro paper analysis, 2026-06-17)
- **Status**: Approved-for-paper
- **Where it lands**: Paper B §3 (Experimental Setup — Evaluation Protocol); §5 (Main Results footnote); Reviewer-defence appendix
- **Tags**: #experiment #framing #reviewer-defence #dataset
- **Why it matters**: LLaMA-Pro uses per-domain binary metrics (exact-match for math, pass@1
  for code). INCA cannot use these directly. Token-level F1 is the correct unified substitute
  and is itself a contribution — it is what makes the saturation detector work.
- **Body**:
  - **What LLaMA-Pro reports:**
    - Math (GSM8K, GSM8K-PoT, MATH): exact-match accuracy on the final extracted number.
      Binary: 0 or 1 per example.
    - Code (HumanEval, MBPP): pass@1 — generate code, *execute it* against unit test cases.
      Binary per test case.
    - LLaMA-Pro uses different metrics per domain with no unified signal.
  - **Why we cannot use the same metrics:**
    1. Our code dataset (flytech/python-codes-25k) has no unit test cases → pass@1 is
       undefined. HumanEval/MBPP have baked-in tests; our training split does not.
    2. Binary metrics (exact-match / pass@1) are fatal to the saturation detector:
       early in training F1 = 0 → RIR = 0 → BLOCK_FULL never fires → every period
       ends in EXHAUSTED fallback. This is exactly the failure mode we observed with
       competition_math (P1_math). GSM8K final-number-only extraction has the same problem.
  - **Token-level F1 (SQuAD-style) — our unified metric:**
    - Bag-of-words partial credit: fraction of reference tokens present in hypothesis.
    - Smooth over the course of training → non-zero RIR from step 1 → BLOCK_FULL fires
      when learning genuinely plateaus.
    - Applied uniformly across all 5 domains (trivia QA, code, science, medical, commonsense).
    - For GSM8K (if used as P1): target is the *full chain-of-thought solution* (median
      47 words). Correct reasoning steps score partial F1; correct `#### N` answer
      scores additional F1. Far smoother than extracting only the final number.
  - **Framing this as a contribution (not a limitation):**
    > *"Unlike LLaMA-Pro, which uses task-specific binary metrics (pass@1 for code,
    > exact-match for math) that cannot serve as saturation signals, INCA adopts a
    > unified token-level F1 across all domains. This provides a smooth, differentiable
    > proxy for learning progress that enables the saturation detector to operate without
    > task-specific evaluation harnesses, making INCA domain-agnostic by design."*
  - **What to call it in tables**: always "token F1" (never "accuracy") so reviewers do
    not directly compare numbers to LLaMA-Pro's exact-match or pass@1 scores.
  - **Secondary metric (optional)**: for code, CodeBLEU or rough pass@1 on a small
    HumanEval subset could be reported as an additional column in Table B1 to situate
    our F1 numbers in the broader code-generation landscape. Deferred to camera-ready.
  - **Archive note**: dataset files in `docs/archive/` (gap analysis, research report,
    engineering schema) are designated for Paper A/C/D (temporal stream). They are
    explicitly held out of Paper B. See note in each archive file.

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

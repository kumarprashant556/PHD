# Paper D — Ideas Journal
### Temporal-Driven Continual Learning on a Clean Temporal Benchmark

> A running log of ideas, framings, experiments, and reviewer-defences for Paper D.
> Paper D is the **temporal CL benchmark sweep**: training B1-B7 baselines and INCA on
> the TemporalWiki dataset (and/or Paper C's clean leakage-controlled dataset), producing
> clean BWT/FWT results on a temporal stream that is not confounded by probe-answer leakage.
>
> **Pre-conditions**:
> - Paper C's leakage-controlled dataset (or TemporalWiki with a leakage audit) must exist
>   before Paper D's headline numbers are credible.
> - TemporalWiki leakage audit via `scripts/analyze_probe_leakage.py` must be run first.
>
> **Reverse-chronological** (newest entry at top).  Append-only — mark stale entries
> `Status: Rejected` or `Status: Superseded`, never delete.

---

## Scope — what this paper is and is not

| This paper IS                                               | This paper is NOT                                      |
|-------------------------------------------------------------|--------------------------------------------------------|
| A CL method comparison on a clean temporal stream           | An architecture paper (that's Paper B)                 |
| BWT / FWT / ACC / REG on TemporalWiki or Paper C dataset    | A dataset construction paper (that's Paper C)          |
| B1-B7 baselines + INCA evaluated on leakage-free probes     | Dependent on CC-News leaky probes for headlines         |
| The "temporal CL benchmark" for our series of papers        | The paper that introduces the dataset (Paper C does)   |
| Connecting temporal drift to INCA's capacity growth         | Paper A (different dataset + different headline claim)  |

**Relationship to other papers:**
- Paper C builds the clean benchmark → Paper D uses it.
- Paper B establishes INCA architecture → Paper D evaluates it in a temporal context.
- Paper A is Paper D's predecessor on CC-News (leaky pilot); Paper D is the clean version.
- Thesis Chapter 4 is "the temporal CL results chapter" — Paper D feeds into it directly.

---

## Conventions

Same template as Paper A's journal. Additional tag `#temporal` for temporal-stream-specific
design decisions.

### Tag glossary

| Tag                 | Use when …                                                                |
|---------------------|---------------------------------------------------------------------------|
| `#temporal`         | A decision specific to the temporal-stream framing.                       |
| `#sweep`            | A full B1-B7 + INCA training sweep design decision.                       |
| `#metric`           | A BWT/FWT/ACC metric design choice.                                       |
| `#baseline`         | A specific baseline (B1-B7) configuration or result.                      |
| `#dataset`          | Which dataset(s) to use or compare on.                                    |
| `#reviewer-defence` | Pre-empts a specific anticipated reviewer objection.                      |
| `#figure`           | A specific figure or table for Paper D.                                   |

---

## Entries

<!-- Newest entry goes immediately below this comment. -->

### 2026-06-16 · Joint-session · B1 CC-News result: BWT ≈ 0, confirms leaky-benchmark hypothesis
- **Source**: Joint-session (B1 sweep completed 2026-06-16; analysis conducted same session)
- **Status**: Approved-for-paper
- **Where it lands**: Paper D §5 (Discussion) — CC-News side of Figure D4; corroborates
  the "leaky benchmark → flat BWT" narrative; cross-reference to the CC-News vs TemporalWiki
  comparison entry (2026-06-16 · CC-News (leaky) vs TemporalWiki (clean))
- **Tags**: #baseline #temporal #metric #reviewer-defence #figure
- **Why it matters**: B1 (naive sequential fine-tuning on CC-News) produced BWT = −0.0018 —
  essentially zero forgetting. This is not a positive result for INCA; it is the expected
  artefact of a leaky benchmark and exactly what the CC-News vs TemporalWiki story predicts.
  It provides concrete empirical backing for the claim that leakage suppresses measurable BWT.
- **Body**:
  - **Run**: `python scripts/train_baselines.py --config configs/base.yaml --device mps`
    baseline `b1_finetune`; CC-News v2, 4 half-year periods, 25 000 items/period.
    Results dir: `results/sweep_20260615_234028/b1_finetune_20260615_234028/`
  - **Headline numbers**:
    - ACC = 0.2435
    - BWT = −0.0018  ← near-zero forgetting for naive sequential fine-tuning
    - FWT = +0.2271
  - **BWT matrix R[t, j]** (accuracy of model-after-period-t on probes-of-period-j):
    ```
                 2017_H1  2017_H2  2018_H1  2018_H2
    after 2017_H1  0.2083    —        —        —
    after 2017_H2  0.2188   0.2316    —        —
    after 2018_H1  0.2448   0.2316   0.2312    —
    after 2018_H2  0.2188   0.2211   0.2258   0.3085
    ```
  - **Key observations**:
    1. **BWT = −0.0018**: catastrophic forgetting is absent. Naive fine-tuning on a
       homogeneous temporal domain does not overwrite earlier representations.
    2. **Positive backward transfer**: R[2,0] = 0.2448 > R[0,0] = 0.2083 — training on
       2018_H1 *increased* accuracy on 2017_H1 probes. Domain coherence lets later training
       reinforce earlier period representations.
    3. **Growing FWT**: 0.1895 → 0.2258 → 0.2660. The model progressively better
       anticipates the next period before training on it — a strong positive forward transfer
       signal consistent with CC-News cross-period vocabulary overlap.
    4. **Diagonal trend** 0.2083 → 0.2316 → 0.2312 → 0.3085. Current-period accuracy
       increases across the curriculum, especially in P4, reflecting warm-start benefits.
    5. **Loss trend** 3.3990 → 3.1621 → 3.1331 → 3.0716. Monotone decrease across periods.
  - **Interpretation for Paper D narrative**: The expected pattern from the
    "leaky benchmark → flat BWT" hypothesis is confirmed: CC-News leakage (75–83%) means
    probe answers are present in multiple periods' training text, so the model can "re-learn"
    forgotten answers from cross-period leakage rather than exhibiting true forgetting.
    B1's BWT = −0.0018 is not evidence that naive fine-tuning solves forgetting — it is
    evidence that the CC-News probes are too leaky to measure forgetting.
  - **Implication for INCA on CC-News**: INCA's BWT advantage over B1 on CC-News will be
    near-zero for the same reason. The architecture advantage must be measured on
    TemporalWiki (or Paper C's clean dataset) where leakage is controlled.
  - **B2 status (as of entry date)**: still running (period 2, step ~900/1270). B2 period 1
    is identical to B1 (replay doesn't activate until period 2). Period 2 loss trajectory is
    slightly noisier than B1 (expected: mixed replay batches inflate effective dataset) but
    converging to the same range. Final B2 BWT expected to differ from B1 by < 0.005 given
    that B1 already does not forget.
  - **Action**: when B2–B7 complete, add a follow-up entry confirming that all baselines
    show flat BWT on CC-News. This makes Figure D4 (left panel) a clean negative control.

### 2026-06-16 · User-instructed · Establish Paper D ideas journal
- **Source**: User-instructed
- **Status**: Approved-for-paper (this file is the artefact)
- **Where it lands**: Process artefact — not in paper
- **Tags**: #temporal
- **Why it matters**: Separates the temporal-CL comparison paper from the architecture paper
  (B) and dataset paper (C) so each has a clean independent story.
- **Body**:
  - Claude does **not** add Claude-proposed entries without Nishant's explicit approval.
  - See sibling journals: [`paper_a_ideas_journal.md`](paper_a_ideas_journal.md),
    [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md),
    [`paper_c_ideas_journal.md`](paper_c_ideas_journal.md).

### 2026-06-16 · User-instructed · TemporalWiki as the primary benchmark dataset
- **Source**: User-instructed
- **Status**: Open — leakage audit required before committing
- **Where it lands**: Paper D §3 (Experimental Setup — Dataset)
- **Tags**: #dataset #temporal #reviewer-defence
- **Why it matters**: TemporalWiki is the most cited temporal CL benchmark that has clean
  temporal snapshots (Wikipedia state at T=0 and T=1 annual). It is the natural choice for
  "clean temporal CL" if its leakage is low. If leakage is high, Paper C's dataset replaces it.
- **Body**:
  - **TemporalWiki structure**:
    - Two Wikipedia snapshots: 2020 and 2021 (annual granularity).
    - ~10k probes (entity cloze from Wikipedia entity pages).
    - Training documents: Wikipedia article text for each snapshot.
    - Probe types: entity cloze — "X was born in <mask>" → city/year.
  - **Leakage risk**: Wikipedia entity names are highly persistent across years. A probe
    "Marie Curie was born in <mask>" has the same answer in 2020 and 2021 — this is
    *intended* for stable probes, but may inflate context-leakage numbers because the entity
    mention appears in both years' training text. Run `scripts/analyze_probe_leakage.py`
    on TemporalWiki to quantify.
  - **Decision tree**:
    - If TemporalWiki leakage is low (target L < 0.30): use as Paper D's headline dataset.
    - If TemporalWiki leakage is high (target L ≥ 0.30): use Paper C's dataset as headline,
      TemporalWiki as secondary comparison.
    - If TemporalWiki leakage is unknown: audit first, then decide (this is the current state).
  - **Fallback**: TiC-LM Track A (if leakage audit passes; already planned for Paper A).
  - **Action**: run `scripts/analyze_probe_leakage.py --dataset temporalwiki --periods ...`
    as the first experiment for Paper D.

### 2026-06-16 · User-instructed · B1-B7 + INCA sweep on TemporalWiki
- **Source**: User-instructed
- **Status**: Open — depends on leakage audit result
- **Where it lands**: Paper D §4 (Results) — Main Table D1 (BWT/FWT/ACC matrix)
- **Tags**: #sweep #experiment #figure
- **Why it matters**: Paper D's empirical contribution is the first clean B1-B7 + INCA
  comparison on a leakage-audited temporal stream. All methods use the same dataset, probes,
  and evaluation protocol; results are directly comparable.
- **Body**:
  - **Methods to sweep** (7 baselines + INCA):
    - B1: Naive sequential fine-tune (lower bound for forgetting).
    - B2: Experience Replay.
    - B3: EWC (Elastic Weight Consolidation).
    - B4: L2P (Learning to Prompt).
    - B5: LoRA-MoE.
    - B6: LLaMA-Pro (block expansion, fixed timing — the Paper B comparison target).
    - B7: PNN (Progressive Neural Networks).
    - INCA: saturation-driven block expansion (our method; uses configs/inca.yaml).
  - **Metrics** (per the TiC-LM/CAPSEL roadmap definitions):
    - `ACC` — average accuracy on current-period probes across all periods.
    - `BWT` — backward transfer: does training on period T hurt period T-k accuracy?
    - `FWT` — forward transfer: does training on earlier periods help period T?
    - `REG` — regret matrix diagonal vs off-diagonal (full matrix in Appendix D).
    - `PAR` — parameters added (INCA, B6, B7 only): accuracy per added parameter.
  - **Figure plan**:
    - D1: Main results table (ACC / BWT / FWT / PAR for all 8 methods).
    - D2: BWT matrix heatmap for INCA vs B1 (shows forgetting distribution across periods).
    - D3: Accuracy curves over training (period-by-period, all methods overlaid).
  - **Infrastructure**: `scripts/train_baselines.py` already runs B1-B7 sweep with
    `--baselines b1,b2,b3,b4,b5,b6,b7`. Need a `--dataset temporalwiki` flag once
    TemporalWiki loader is wired. `scripts/train_inca.py` runs INCA. Both write to
    `results/sweep_<timestamp>/` with standardised output format.

### 2026-06-16 · User-instructed · CC-News (leaky) vs TemporalWiki (clean) as the methodology comparison
- **Source**: User-instructed
- **Status**: Open — requires both datasets to be swept
- **Where it lands**: Paper D §5 (Discussion) + Appendix D
- **Tags**: #temporal #metric #reviewer-defence #figure
- **Why it matters**: The strongest story in Paper D is: "on CC-News (leaky), B1 shows no
  forgetting; on TemporalWiki (clean), forgetting is real and our method reduces it."
  This directly validates the Paper C leakage-control contribution.
- **Body**:
  - **Expected pattern** (hypothesis):
    - CC-News: all methods have flat BWT (≈ 0.00 ± 0.05). INCA shows no advantage.
    - TemporalWiki: B1 shows negative BWT (real forgetting). INCA shows less forgetting
      than B1 and competitive with B3-EWC and B6-LLaMA-Pro.
  - **If the hypothesis is wrong** (INCA does not improve BWT on TemporalWiki either):
    - Option A: INCA's advantage is FWT (not BWT) — capacity growth helps forward transfer.
    - Option B: the model is too small (FLAN-T5-base 250M) to show meaningful growth events
      on TemporalWiki's scale. Reframe as a scale experiment.
  - **Figure D4**: side-by-side BWT bar chart — B1-B7 + INCA on CC-News (left) vs
    TemporalWiki (right). The contrast between flat-BWT-leaky and real-BWT-clean is the
    visual headline of the paper.
  - **Cross-link**: CC-News leakage finding and the case for TemporalWiki are documented
    in [`paper_a_methodology_note_probe_leakage.md`](paper_a_methodology_note_probe_leakage.md)
    §5 and the research ideas journal.

### 2026-06-16 · User-instructed · INCA block-expansion timing on a temporal stream
- **Source**: User-instructed
- **Status**: Open
- **Where it lands**: Paper D §4.2 (INCA Analysis) + Appendix D
- **Tags**: #temporal #architecture #figure
- **Why it matters**: Paper B studies expansion timing on a static domain dataset. Paper D
  studies the same question on a temporal stream. The temporal axis adds a new dimension:
  does INCA grow more blocks after a period boundary (new distribution) than mid-period?
- **Body**:
  - **Hypothesis**: INCA triggers block expansion at or shortly after each period boundary,
    because the new period's distribution causes a saturation signal spike (CKA drift rises,
    loss plateau breaks). This is a *testable* prediction from the saturation theory.
  - **Measurement**: log expansion events (layer index, training step, period) during the
    TemporalWiki sweep. Plot expansion event count vs step number, with period boundaries
    marked.
  - **Expected result**: expansion events cluster within the first few hundred steps of each
    new period, then taper as the model saturates. This pattern validates the CAPSEL theory
    specifically in the temporal setting.
  - **Alternative (negative) result**: expansion events are uniform across training steps —
    the period boundary has no special effect. This would suggest saturation is data-volume-
    driven, not distribution-shift-driven. Still publishable as a finding.
  - **Figure D5**: expansion event timeline across all periods for INCA on TemporalWiki.

### 2026-06-16 · User-instructed · TemporalWiki loader — infrastructure prerequisite
- **Source**: User-instructed
- **Status**: Open — needs implementation before any sweep
- **Where it lands**: `data/temporalwiki.py` + `baselines/_runtime/data.py`; not in paper
- **Tags**: #sweep #dataset
- **Why it matters**: The current `data/__init__.py` registers `temporalwiki` as a
  dataset (`data.temporalwiki.load_temporalwiki_periods`) but the loader itself may not
  be implemented yet. This is the critical path item for Paper D.
- **Body**:
  - **Check**: does `data/temporalwiki.py` exist and implement `load_temporalwiki_periods`?
    (File not found in current repo tree — needs to be written or ported from
    `legacy/Phase0/data/download_temporalwiki.py` + `preprocess_temporalwiki.py`.)
  - **What the loader must return**: `Dict[period_id, Dataset]` with columns
    `input_text`, `target_text`, `period` — standard CAPSEL loader contract
    (see `data/_base.py`).
  - **Period IDs**: TemporalWiki uses annual snapshots; natural IDs are `2020`, `2021`.
  - **Action**: check `data/temporalwiki.py` existence; if missing, port from
    `legacy/Phase0/data/` and wire to `baselines/_runtime/data.py`'s dataset switch.
  - **Note**: `scripts/analyze_probe_leakage.py` will also need a TemporalWiki mode
    (different probe JSONL path; different period list). This is a 15-line extension.

---

## Document metadata

| Field        | Value                                                                         |
|--------------|-------------------------------------------------------------------------------|
| Author       | Nishant Kumar (with Claude assistance)                                        |
| Created      | 2026-06-16                                                                    |
| Purpose      | Running log of Paper D ideas — temporal CL sweep on clean temporal benchmark  |
| Append rule  | New entries go at the top of "Entries"; never delete, only mark status        |
| Owner        | Nishant — all Claude-proposed entries require explicit approval before adding  |
| Sibling docs | [`paper_a_ideas_journal.md`](paper_a_ideas_journal.md) — temporal CL (CC-News)|
|              | [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md) — capacity growth      |
|              | [`paper_c_ideas_journal.md`](paper_c_ideas_journal.md) — dataset construction |
|              | [`research_ideas_journal.md`](research_ideas_journal.md) — domain-level       |

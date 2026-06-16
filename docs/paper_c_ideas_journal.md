# Paper C — Ideas Journal
### Building a Leakage-Controlled Temporal Benchmark Dataset

> A running log of ideas, framings, experiments, and reviewer-defences for Paper C.
> Paper C is the **dataset paper**: constructing a clean temporal CL benchmark informed by
> the problems found in CC-News v2 (75-83% probe leakage) and TemporalWiki's approach.
> The contribution is both the dataset artifact and the leakage-controlled construction
> methodology.
>
> **Background**: CC-News v2 has 75-83% probe-answer leakage across periods — see
> [`paper_a_methodology_note_probe_leakage.md`](paper_a_methodology_note_probe_leakage.md).
> That finding is the direct motivation for this paper.
>
> **Reverse-chronological** (newest entry at top).  Append-only — mark stale entries
> `Status: Rejected` or `Status: Superseded`, never delete.

---

## Scope — what this paper is and is not

| This paper IS                                               | This paper is NOT                                      |
|-------------------------------------------------------------|--------------------------------------------------------|
| A new temporal benchmark dataset for CL                     | Another CL algorithm paper                             |
| Leakage-controlled probe construction methodology           | A replication of CC-News or TemporalWiki               |
| A critique of existing temporal benchmarks (leakage audit)  | Paper A (which *uses* this dataset as Phase 2 data)    |
| Source-agnostic construction pipeline (CC-News / wiki / arXiv) | Paper D (which *trains* CL models on this dataset) |
| The standard eval artefact Paper D and Paper A both need    | A CL algorithm comparison                              |

**Relationship to other papers:**
- Paper A is blocked on a clean benchmark → Paper C unblocks it.
- Paper D trains and evaluates CL methods on Paper C's dataset.
- Paper B does not use this dataset (it uses domain-specific non-temporal data).
- Thesis Chapter 3 can be "the dataset paper."

---

## Conventions

Same template as Paper A's journal. Additional tag `#pipeline` for construction-pipeline
design decisions.

### Tag glossary

| Tag                 | Use when …                                                                |
|---------------------|---------------------------------------------------------------------------|
| `#pipeline`         | A construction-pipeline design decision (how probes are generated/filtered)|
| `#leakage-control`  | A design choice specifically to prevent probe-answer leakage.             |
| `#source-selection` | Which raw text source(s) to draw from.                                    |
| `#probe-design`     | How cloze / QA / MCQ probes are constructed or annotated.                 |
| `#benchmark-scope`  | Scale, period count, domains, languages, etc.                             |
| `#reviewer-defence` | Pre-empts a specific anticipated reviewer objection.                      |
| `#experiment`       | A validation experiment (e.g., leakage audit on a proposed source).       |

---

## Entries

<!-- Newest entry goes immediately below this comment. -->

### 2026-06-16 · User-instructed · Establish Paper C ideas journal
- **Source**: User-instructed
- **Status**: Approved-for-paper (this file is the artefact)
- **Where it lands**: Process artefact — not in paper
- **Tags**: #pipeline
- **Why it matters**: Separates dataset-paper decisions from the algorithm papers (A, B, D)
  so the construction methodology can be developed independently.
- **Body**:
  - Claude does **not** add Claude-proposed entries without Nishant's explicit approval.
  - See sibling journals: [`paper_a_ideas_journal.md`](paper_a_ideas_journal.md),
    [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md),
    [`paper_d_ideas_journal.md`](paper_d_ideas_journal.md).

### 2026-06-16 · User-instructed · Leakage-controlled probe construction as the key contribution
- **Source**: User-instructed
- **Status**: Open — methodology needs formalisation
- **Where it lands**: Paper C §3 (Construction Methodology) — the paper's core contribution
- **Tags**: #pipeline #leakage-control #reviewer-defence
- **Why it matters**: No existing temporal CL benchmark ships a leakage matrix or a
  leakage-controlled probe split as a standard artefact. Making this a first-class
  construction step — not a post-hoc audit — is a methodology contribution with leverage
  across the whole field.
- **Body**:
  - **The problem**: in CC-News v2, 75-83% of any later period's probe answers appear as
    training targets in the earliest period. Any CL method trained on this stream cannot
    *forget* these answers because they keep being re-taught. BWT is uninterpretable.
  - **The fix we propose**: at construction time, before finalising probes, run a
    cross-period leakage check and:
    1. **Discard** probes whose answer appears in ≥ 1 other period's training stream
       (target leakage check).
    2. **Flag** probes whose answer appears as a context substring in ≥ 1 other period
       (context leakage flag — kept but annotated, so researchers can filter).
    3. **Report** the final leakage matrix as a benchmark artefact (not a caveat).
  - **The tool already exists**: `scripts/analyze_probe_leakage.py` runs this check.
    Paper C is essentially: (a) apply it at *construction* time, not post-hoc, and
    (b) run it on multiple sources (CC-News, TemporalWiki, our new stream) to show
    that leakage is endemic unless you control for it.
  - **Specific novelty vs. TemporalWiki**: TemporalWiki controls for entity-popularity bias
    but does NOT report or control for cross-period probe-answer leakage. Our methodology
    adds that missing step.
  - **Leakage threshold**: probe is "leakage-free" if target leakage L[t,j] < 0.10 for
    all off-diagonal (t,j). This threshold needs a justification (empirical or principled).

### 2026-06-16 · User-instructed · Source selection — TemporalWiki + CC-News as the comparison baseline
- **Source**: User-instructed
- **Status**: Open — source audit needed before committing
- **Where it lands**: Paper C §2 (Related Work + Source Audit) + §3 (Construction)
- **Tags**: #source-selection #experiment
- **Why it matters**: Paper C needs to (a) show existing sources are leaky, (b) propose a
  construction methodology that fixes the leak, and (c) apply it to build the actual dataset.
  We already have the CC-News v2 leakage audit. TemporalWiki is the second most natural
  comparison.
- **Body**:
  - **Sources to audit** (in order of priority):
    1. **CC-News v2** (already done): 75-83% target leakage / 88-95% context leakage.
       Result is in `results/leakage/` and `paper_a_methodology_note_probe_leakage.md`.
    2. **TemporalWiki** (top priority): Wikipedia snapshots with entity-change probes. High
       overlap risk because entity names repeat constantly across monthly snapshots. Run
       `scripts/analyze_probe_leakage.py` on TemporalWiki probe set.
    3. **StreamingQA** (binary QA format, likely high leakage): news corpus, 2007-2020.
       Harder to audit because format differs from our JSONL; may need adapter.
    4. **Our new stream** (after construction): build from CC-News with the leakage-control
       pipeline applied at source; report the resulting leakage matrix and show it's below
       threshold.
  - **What "our new stream" looks like**:
    - Same CC-News raw data, but probes are filtered at construction time (step 1-3 above).
    - Cross-period probe deduplication: normalised answers that appear in multiple periods
      are either dropped (strict) or flagged (lenient). Paper C proposes both splits.
    - Period granularity: quarterly (3-month) gives more periods + more probe diversity.
    - Target: 20-40 leakage-free probes per period × 12-20 periods = 240-800 clean probes.
  - **Why not arXiv or Wikipedia revisions?**:
    - ArXiv abstracts: low entity-name diversity, high leakage risk (same authors, same
      topics repeat); harder to generate natural cloze probes.
    - Wikipedia revisions: TemporalWiki has already done this; building our own would
      duplicate effort without adding novelty.
  - **Action**: run `scripts/analyze_probe_leakage.py --periods ...` on TemporalWiki as
    the first experiment for Paper C.

### 2026-06-16 · User-instructed · Cross-period probe stability annotation
- **Source**: User-instructed
- **Status**: Open — annotation protocol needs design
- **Where it lands**: Paper C §3.3 (Probe Stability Annotation)
- **Tags**: #probe-design #pipeline
- **Why it matters**: A probe is not just "clean" or "leaky" — it may be *stable* (same
  answer across all periods), *updated* (answer changed at period T), or *deprecated*
  (fact no longer probed after period T). The stability label turns a leakage-free probe
  set into a *fact-dynamics* dataset, which is a much stronger contribution.
- **Body**:
  - **Three stability classes** (already in `preprocessing/temporal.py` schema):
    - `stable` — correct answer does not change across any period in the stream.
    - `updated` — correct answer changes at a known period boundary (the "drift" signal).
    - `deprecated` — the entity/fact disappears from the stream and the probe is retired.
  - **Annotation method**:
    - Automatic (first pass): cross-period answer diff. If the most-common answer for a
      probe entity changes between period T and T+1, flag as `updated`.
    - Human validation (second pass): a human reviews flagged probes to confirm the
      answer change is genuine (not a tokenisation artefact or NER error).
  - **What this enables**: an `updated`-only probe subset directly measures a CL model's
    ability to *overwrite* old knowledge with new knowledge — the purest forgetting-vs-update
    signal in the literature.
  - **Downstream use**: Paper D uses the stability-annotated probe set. Paper A uses the
    leakage-free subset. Paper C ships both as dataset splits.
  - **Scale estimate**: at 20-40 leakage-free probes/period × 20 periods, expect ~10-30
    `updated` probes per period if CC-News has typical entity-mention dynamics.

### 2026-06-16 · User-instructed · Minimum viable scale for a credible temporal CL benchmark
- **Source**: User-instructed
- **Status**: Open — needs literature scan
- **Where it lands**: Paper C §1 (Introduction) + §2 (Related Work) + §4 (Dataset Statistics)
- **Tags**: #benchmark-scope #reviewer-defence #experiment
- **Why it matters**: "40 probes per period is too small" is the obvious reviewer objection.
  The counter-argument needs to be evidence-based: what scale do the accepted benchmarks use,
  and what is the minimum N for statistically-interpretable BWT?
- **Body**:
  - **Current landscape**:
    - TemporalWiki: ~10k probes total across 2 periods (annual snapshots).
    - TiC-LM: ~thousands of probes per period but extremely heavy to run.
    - CC-News v2: ~190 scored probes per period (small, but manageable).
    - StreamingQA: binary QA, ~tens of thousands total across 13 years.
  - **The claim to defend**: "20-40 leakage-free probes per period is sufficient to compute
    a statistically-interpretable BWT confidence interval, because BWT is a *relative*
    metric (difference of two accuracy estimates) and paired-sample tests apply."
  - **Evidence needed**: a simulation showing that with N=30 probes/period,
    a BWT difference of ±0.10 is detectable at p<0.05. This is a standard sample-size
    calculation (McNemar's test for matched binary outcomes).
  - **Alternative framing**: "we sacrifice per-period N for leakage cleanliness — a
    40-probe leakage-free benchmark is more informative than a 190-probe leaky one."
    This is the dataset paper's headline argument.
  - **Action**: run the sample-size simulation; cite TemporalWiki §4 on their N choice.

### 2026-06-16 · User-instructed · Paper C as the foundation that unlocks Papers A and D
- **Source**: User-instructed
- **Status**: Open — dependency ordering decision
- **Where it lands**: Paper C §1 (Introduction); cross-refs from Papers A and D
- **Tags**: #framing #reviewer-defence
- **Why it matters**: A dataset paper without downstream users is weak. The fact that
  Paper A and Paper D both depend on Paper C's clean probe set is the strongest
  reviewer-deflection: the dataset is not an end in itself, it's the precondition for
  reproducible CL evaluation.
- **Body**:
  - **Dependency graph**:
    - Paper C → Paper A: clean probes replace CC-News v2 leaky probes in §4.
    - Paper C → Paper D: clean probes are the eval benchmark for the B1-B7 + INCA sweep.
    - Paper B does not depend on Paper C (uses domain-specific non-temporal data).
  - **Sequencing options**:
    1. Submit Paper C first, cite in Papers A and D as "under review."
    2. Combine Paper C's dataset construction into Paper A as §3 + Appendix B.
    3. Combine Paper C + Paper D into one "temporal CL benchmark + baseline sweep" paper.
  - **Recommendation (open)**: Option 1 maximises paper count and is standard for datasets.
    Option 2 is faster but buries the methodology contribution. Decision should be made
    after checking the conference venue's data-track policy.

---

## Document metadata

| Field        | Value                                                                         |
|--------------|-------------------------------------------------------------------------------|
| Author       | Nishant Kumar (with Claude assistance)                                        |
| Created      | 2026-06-16                                                                    |
| Purpose      | Running log of Paper C ideas — temporal benchmark dataset construction         |
| Append rule  | New entries go at the top of "Entries"; never delete, only mark status        |
| Owner        | Nishant — all Claude-proposed entries require explicit approval before adding  |
| Sibling docs | [`paper_a_ideas_journal.md`](paper_a_ideas_journal.md) — temporal CL (BWT)    |
|              | [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md) — capacity growth      |
|              | [`paper_d_ideas_journal.md`](paper_d_ideas_journal.md) — temporal CL on wiki  |
|              | [`research_ideas_journal.md`](research_ideas_journal.md) — domain-level       |
|              | [`paper_a_methodology_note_probe_leakage.md`](paper_a_methodology_note_probe_leakage.md) — leakage finding |

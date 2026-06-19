> **SUPERSEDED 2026-06-16** — "Paper A" (temporal CL on CC-News / TiC-LM) no longer exists as a
> standalone paper. The CAPSEL/INCA capacity-driven architecture paper is now the **main paper**
> (see [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md) and
> [`paper_main_outline.md`](paper_main_outline.md)). The temporal-dataset work is tracked in
> [`paper_c_ideas_journal.md`](paper_c_ideas_journal.md) and
> [`paper_d_ideas_journal.md`](paper_d_ideas_journal.md).
> Retained here as archived evidence of the temporal-CL framing. Do not add new entries.

# Paper A — Ideas Journal (SUPERSEDED — see paper_b_ideas_journal.md)

> A running log of ideas, framings, experiments, and reviewer-defences for the CAPSEL/INCA
> Paper A. **Reverse-chronological** (newest entry at top). Append-only — never delete an
> entry; mark it `Status: Rejected` or `Status: Superseded` with a reason instead.

---

## Conventions

Every entry follows the same template so they're filterable by eye:

```
### YYYY-MM-DD · [Source] · short title
- **Source**: User-instructed | Claude-proposed (approved YYYY-MM-DD) | Joint-session
- **Status**: Open | Approved-for-paper | In-draft | Rejected | Superseded | Deferred
- **Where it lands**: §X.Y of Paper A, or "Appendix A", or "Thesis Ch. N", or "Decision only"
- **Tags**: #methodology #experiment #framing #reviewer-defence #related-work #figure
- **Why it matters** (1-2 sentences)
- **Body** (free-form — claim, evidence, action items, open questions)
```

### Source rules

- **User-instructed** — Nishant told me to add it. Goes in immediately.
- **Claude-proposed** — I noticed something while working. I ask first; if approved, I add it
  with the approval date in parentheses. **Never added without permission.**
- **Joint-session** — emerged from a conversation; both parties agreed it should be logged.

### Lifecycle rules

- New ideas start at `Status: Open`.
- When the relevant Paper A section is being drafted and the idea is committed, move to
  `Status: In-draft`.
- When the section is finalised and cites/uses the idea, move to `Status: Approved-for-paper`.
- If the idea is dropped, mark `Status: Rejected` with a one-line reason in the body.
- If a later entry supersedes this one, mark `Status: Superseded` and link to the new entry.

### How to add a new entry

1. Add the new entry at the **top** of the "Entries" section (below the divider).
2. Keep the template fields in the same order.
3. If the entry references a file, link to the path relative to the repo root (e.g.,
   `docs/literature_survey_t5_base_em_scores.md`).
4. Cross-link related entries by date+slug (e.g., "see 2026-06-16 · leakage-finding").

### Tag glossary

| Tag                 | Use when …                                                             |
|---------------------|------------------------------------------------------------------------|
| `#methodology`      | The idea changes how an experiment is run or how a metric is computed. |
| `#experiment`       | The idea is a new experiment, baseline, or ablation.                   |
| `#framing`          | The idea changes the paper's narrative or claim structure.             |
| `#reviewer-defence` | The idea pre-empts a specific anticipated reviewer objection.          |
| `#related-work`     | The idea is a paper/result to cite or position against.                |
| `#figure`           | The idea is a specific figure or table that should appear in the paper.|
| `#title-abstract`   | The idea affects the title, abstract, or first paragraph.              |
| `#thesis-only`      | Only relevant to thesis chapter, not Paper A.                          |

---

## Entries

<!-- Newest entry goes immediately below this comment. -->

### 2026-06-16 · User-instructed · Consider LLaMA-Pro's dataset stack as an alternative to a temporal stream
- **Source**: User-instructed
- **Status**: Open — needs scoping decision before §3 (Datasets) is finalised
- **Where it lands**: Paper A §3 (Datasets) — or, more likely, §1 framing if we adopt it
- **Tags**: #framing #experiment #methodology
- **Why it matters**: CAPSEL/INCA is fundamentally a **capacity-driven** contribution (see
  `docs/capsel-inca-project.md` and the CAPSEL roadmap PDF) — block-expansion / saturation-driven
  growth, not temporal CL. The temporal-drift framing was inherited from the CC-News pilot and
  is now wobbling under (a) the leakage finding and (b) the broader dataset-curation pain.
  LLaMA-Pro (Wu et al., 2024) demonstrated block expansion on a **math + code** corpus with no
  temporal framing at all, which is the closest analogue in the literature to what we're
  actually building.
- **Body**:
  - **Hypothesis to log**: if Paper A's headline is "saturation-driven capacity growth," the
    dataset only needs to (i) saturate the base model and (ii) admit a clean BWT measurement.
    Temporal stratification is *one* way to get (ii) but not the only one.
  - **Pros**: aligns the dataset choice with the actual contribution; sidesteps the entire
    probe-leakage hazard (math/code answers don't repeat across train slices the way news
    entities do); direct comparison to LLaMA-Pro becomes possible.
  - **Cons**: surrenders the "continual learning over time" narrative; loses the TiC-LM Track A
    plan (see 2026-06-16 · tic-lm-headline); requires re-tooling the preprocessing pipeline for
    math/code data.
  - **Decision pending**: should be settled jointly with the methodology-paper-vs-architecture
    decision in `paper_a_methodology_note_probe_leakage.md` §6.

### 2026-06-16 · User-instructed · Consider building our own small curated temporal dataset
- **Source**: User-instructed
- **Status**: Open — needs scoping; high cost-risk
- **Where it lands**: Paper A §3 (Datasets) — or deferred to a separate dataset paper
- **Tags**: #experiment #methodology #reviewer-defence
- **Why it matters**: Every third-party temporal stream we've tried has a known defect —
  CC-News v2 has 75-83% probe leakage (see 2026-06-16 · leakage-finding); TiC-LM is heavy /
  slow / not yet leakage-audited; StreamingQA and TemporalWiki have their own scale-vs-clean
  tradeoffs. Building our **own** small curated stream from a single source might be cheaper
  than continuing to fight third-party data.
- **Body**:
  - **Candidate sources**: (i) Wikipedia revision history slices (TemporalWiki-style but
    leakage-controlled at construction time); (ii) one news site's RSS archive with strict
    de-duplication across periods; (iii) ArXiv abstracts by month with cited-fact probes.
  - **Pros**: full control over leakage, probe design, train/eval format alignment, scale —
    every defect we've spent time diagnosing on CC-News disappears at construction.
  - **Cons**: dataset construction is notoriously time-eating (months, not weeks); reviewers
    will push back ("yet another bespoke benchmark"); no direct comparison to prior CL work.
  - **Risk**: this becomes the whole PhD if not scoped tightly. Hard time-box required.
  - **Lighter-weight alternative**: instead of a new *training* stream, build a small
    **leakage-corrected probe set** on top of an existing stream (CC-News or TiC-LM). This is
    a 1-2 week effort, not a 6-month one, and is essentially the drift-only-probe-subset entry
    (see 2026-06-16 · drift-only-subset) generalised.
  - **Cross-link**: the broader question "should this be its own paper?" is tracked in
    [`docs/research_ideas_journal.md`](research_ideas_journal.md) (#benchmark-design).

### 2026-06-16 · Joint-session · CC-News leakage matrix as Appendix A
- **Source**: Joint-session (this conversation, immediately after `analyze_probe_leakage.py` run)
- **Status**: Approved-for-paper
- **Where it lands**: Paper A Appendix A (Figures A1 + A2); §3.2 Datasets paragraph cites it
- **Tags**: #methodology #figure #reviewer-defence
- **Why it matters**: The CC-News v2 stream has 75-83% probe-answer leakage across periods. If
  we don't disclose it, a reviewer will. If we *do* disclose it, it becomes a methodology
  contribution rather than a hole in the paper.
- **Body**:
  - Target-leakage matrix and context-leakage matrix verbatim from `results/leakage/`.
  - Captions and §3.2 text already drafted in
    [`paper_a_methodology_note_probe_leakage.md`](paper_a_methodology_note_probe_leakage.md)
    §8-§9 — paste-ready.
  - Implication: any BWT-on-CC-News claim is invalid; TiC-LM Track A becomes the only headline
    BWT venue (see 2026-06-16 · tic-lm-headline).

### 2026-06-16 · Joint-session · TiC-LM Track A as the only BWT venue
- **Source**: Joint-session
- **Status**: Approved-for-paper
- **Where it lands**: Paper A §5 (Main results); referenced from §3.2 disclosure
- **Tags**: #framing #experiment #methodology
- **Why it matters**: Forces the paper's architecture — CC-News is demoted to "pilot benchmark,
  Section 3", and TiC-LM Track A holds all forgetting/regret claims. Plan-B (CC-News headline)
  contingency is closed off by the leakage finding.
- **Body**:
  - Precondition: re-run `scripts/analyze_probe_leakage.py` against TiC-LM Track A probes
    before committing. If TiC-LM shows similar leakage, paper pivots to methodology
    contribution (see 2026-06-16 · methodology-paper-alt).
  - CC-News still appears in §4 for convergence-speed and representation-stability comparisons
    — those are not confounded by leakage.

### 2026-06-16 · Joint-session · Report (EM, token-F1, substring-match) triplet on diagonals
- **Source**: Joint-session (raised in `literature_survey_t5_base_em_scores.md` §4)
- **Status**: Approved-for-paper
- **Where it lands**: Paper A Table 1 (diagonals only); Thesis Ch. 4 §4.2
- **Tags**: #methodology #reviewer-defence
- **Why it matters**: Closes off the reviewer objection "your absolute numbers are low" — same
  predictions, three metrics, ~5-10 point spread. Documented for T5-base by the HF model card
  showing EM 17 vs subset-match 24.5 on identical predictions.
- **Body**:
  - Triplet only on the *diagonal* — full off-diagonal matrix stays single-metric for clarity.
  - Eval code already supports `eval_mode: combined` (em_weight + f1_weight); needs a small
    extension to also emit substring-match. Not yet implemented.

### 2026-06-16 · Claude-proposed (approved 2026-06-16) · Drift-only probe subset as the leakage-corrected eval
- **Source**: Claude-proposed (approved in same conversation)
- **Status**: Open — needs `scripts/build_drift_only_probes.py` written
- **Where it lands**: Paper A §4 (CC-News results) as the leakage-corrected sub-evaluation
- **Tags**: #experiment #methodology
- **Why it matters**: Even if CC-News is not the headline, a leakage-corrected sub-evaluation
  on CC-News makes the literature-band comparison apples-to-apples and rescues a "real CL
  signal" claim from the pilot benchmark.
- **Body**:
  - Filter v2 probes to those whose normalised answer does NOT appear in any other period's
    training stream (target or context).
  - Estimated N: 30-60 probes per period (extrapolating from the 0.17-0.25 unique-to-period
    fraction).
  - Eval-only re-run on existing checkpoints; no training cost.
  - Detailed plan in
    [`paper_a_methodology_note_probe_leakage.md`](paper_a_methodology_note_probe_leakage.md)
    §7 item 1.

### 2026-06-16 · User-instructed · Establish this ideas journal
- **Source**: User-instructed
- **Status**: Approved-for-paper (this file is the artefact)
- **Where it lands**: Process artefact — not in paper
- **Tags**: #methodology
- **Why it matters**: Centralises the paper's idea pool so nothing slips between sessions.
- **Body**:
  - File is append-only; entries are reverse-chronological.
  - Claude does not add Claude-proposed entries without Nishant's explicit approval (consistent
    with the broader workflow preference recorded in MEMORY.md about not auto-running smoke
    tests).

---

## Document metadata

| Field        | Value                                                                       |
|--------------|-----------------------------------------------------------------------------|
| Author       | Nishant Kumar (with Claude assistance)                                      |
| Created      | 2026-06-16                                                                  |
| Purpose      | Running log of Paper A ideas — claims, experiments, framings, defences      |
| Append rule  | New entries go at the top of "Entries"; never delete, only mark status      |
| Owner        | Nishant — all Claude-proposed entries require explicit approval before adding|

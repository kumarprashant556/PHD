# Research Ideas Journal — Domain & Open Questions

> A running log of **research-domain** ideas: open questions, candidate directions, gaps in
> the literature, and cross-paper threads that don't belong to any single paper. Scope is
> deliberately broader than [`paper_a_ideas_journal.md`](paper_a_ideas_journal.md), which is
> Paper A-specific.
>
> **Reverse-chronological** (newest entry at top). Append-only — never delete an entry; mark
> it `Status: Rejected`, `Status: Superseded`, or `Status: Resolved` with a reason.

---

## Scope — what belongs here vs. paper-specific journals

| Goes here (research_ideas_journal)            | Goes in paper_a_ideas_journal                       |
|------------------------------------------------|------------------------------------------------------|
| Open questions in the field at large           | Specific claim/experiment/figure for Paper A         |
| Candidate dataset / benchmark directions       | A decision about which dataset *Paper A* uses        |
| Cross-paper threads (Paper A → Paper B → thesis)| Single-paper architectural decision                  |
| Literature gaps spotted while reading          | A specific citation to add to Paper A's related work |
| Methodology ideas not yet tied to a paper      | Methodology already approved for Paper A             |
| Domain framings / taxonomies / surveys-of-self | Paper A title/abstract/§1 wording                    |

When an entry here matures enough to commit to a specific paper, **cross-link** to the
paper-specific journal entry (don't move the entry — append a status update).

---

## Conventions

Same entry template as Paper A's journal, plus a `Domain` tag at the top so a single sweep
shows which research thread an idea belongs to:

```
### YYYY-MM-DD · [Source] · short title
- **Domain**: one of {online-temporal-data, selective-forgetting, capacity-driven-cl,
                       benchmark-design, evaluation-methodology, cross-cutting}
- **Source**: User-instructed | Claude-proposed (approved YYYY-MM-DD) | Joint-session
            | Literature (paper citation)
- **Status**: Open | In-draft | Approved-for-paper | Rejected | Superseded | Resolved
            | Deferred
- **Where it might land**: Paper A § X, Paper B, Thesis Ch. N, or "scoping only"
- **Tags**: #online-temporal-data #selective-forgetting #capacity-driven-cl #benchmark-design
          #evaluation-methodology #open-question #literature-gap #thesis-candidate
          #survey-needed
- **Why it matters** (1-2 sentences)
- **Body** (free-form — claim, evidence, papers to check, open sub-questions)
```

### Source rules (same discipline as Paper A's journal)

- **User-instructed** — Nishant told me; goes in immediately.
- **Claude-proposed** — I noticed something; I ask first; if approved, added with the approval
  date in parentheses. **Never added without permission.**
- **Joint-session** — emerged from a conversation; both parties agreed it should be logged.
- **Literature** — pulled from a paper I (or Nishant) read; cite the paper in the body.

### Lifecycle rules

- New ideas start at `Status: Open`.
- When the idea gets committed to a specific paper, move to `Status: Approved-for-paper`
  *and* cross-link to that paper's journal entry.
- If the idea turns out to be already-solved in the literature, mark `Status: Resolved` with
  the citation that resolves it.
- If the idea is dropped, mark `Status: Rejected` with a one-line reason.
- If a later entry supersedes this one, mark `Status: Superseded` and link.

### How to add a new entry

1. Add the new entry at the **top** of the "Entries" section.
2. Keep template field order.
3. If the idea was prompted by a specific paper, cite it (`Author Year, Section X`).
4. Cross-link related entries by date+slug.

### Domain tag glossary

| Domain                       | What's in scope                                                        |
|------------------------------|------------------------------------------------------------------------|
| `online-temporal-data`       | Streaming temporal datasets, drift benchmarks, news/wiki streams.      |
| `selective-forgetting`       | Targeted unlearning, forgetting-by-design, privacy-driven CL.          |
| `capacity-driven-cl`         | Block expansion, saturation-driven growth, CAPSEL/INCA's home turf.    |
| `benchmark-design`           | What makes a CL benchmark valid; leakage, BWT gameability, probe QA.   |
| `evaluation-methodology`     | Metrics — EM/F1/subset-match, BWT/regret, calibration under shift.     |
| `cross-cutting`              | Ideas that span ≥ 2 domains.                                           |

### Cross-cutting tag glossary

| Tag                     | Use when …                                                          |
|-------------------------|---------------------------------------------------------------------|
| `#open-question`        | The idea is a question the field has not answered.                  |
| `#literature-gap`       | The idea identifies missing work (no one has measured / built this).|
| `#thesis-candidate`     | The idea is a thesis-chapter candidate, not a paper.                |
| `#survey-needed`        | Resolving this requires a literature scan before committing.        |
| `#dataset-candidate`    | The idea proposes a specific dataset or stream.                     |
| `#metric-candidate`     | The idea proposes a specific metric or eval protocol.               |

---

## Entries

<!-- Newest entry goes immediately below this comment. -->

### 2026-06-16 · User-instructed · Paper D direction — temporal CL sweep on a clean temporal stream
- **Domain**: online-temporal-data
- **Source**: User-instructed
- **Status**: Open — blocked on TemporalWiki leakage audit (Paper C precondition)
- **Where it might land**: Paper D (standalone); Thesis Ch. 4 §4.4
- **Tags**: #open-question #dataset-candidate #metric-candidate
- **Why it matters**: The cleanest possible test of whether CL methods actually suppress
  forgetting on a temporal stream requires a benchmark that is not confounded by
  probe-answer leakage. Paper D is the full B1-B7 + INCA sweep on such a benchmark.
- **Body**:
  - **Core contribution**: first clean BWT/FWT comparison of B1-B7 baselines + INCA on
    a leakage-audited temporal stream (TemporalWiki or Paper C dataset).
  - **Headline hypothesis**: on a leaky stream (CC-News), all methods show flat BWT;
    on a clean stream (TemporalWiki/Paper C), forgetting is real and method differences
    are measurable. This validates both Paper C's dataset and INCA's forgetting-suppression
    claim simultaneously.
  - **Dependency**: TemporalWiki leakage audit must be run first (using
    `scripts/analyze_probe_leakage.py`). If TemporalWiki is also leaky, Paper C's dataset
    becomes the only credible temporal benchmark and Paper D's value rises accordingly.
  - **Infrastructure needed**: TemporalWiki loader (`data/temporalwiki.py`) — check
    `legacy/Phase0/data/download_temporalwiki.py` and `preprocess_temporalwiki.py` for
    the starting point.
  - **Cross-link**: paper-specific design decisions in
    [`paper_d_ideas_journal.md`](paper_d_ideas_journal.md).

### 2026-06-16 · User-instructed · Paper C direction — build a leakage-controlled temporal benchmark
- **Domain**: benchmark-design
- **Source**: User-instructed
- **Status**: Open — methodology clear; implementation not started
- **Where it might land**: Paper C (standalone dataset paper); Thesis Ch. 3
- **Tags**: #dataset-candidate #benchmark-design #literature-gap #open-question
- **Why it matters**: The CC-News v2 leakage finding (75-83% probe-answer leakage) shows
  that no existing temporal CL benchmark ships leakage control as a standard step. Paper C
  makes this a first-class construction criterion and ships a clean dataset that Papers A
  and D both depend on.
- **Body**:
  - **Core contribution**: a temporal CL benchmark where leakage control is applied at
    *construction time*, not discovered post-hoc. The `scripts/analyze_probe_leakage.py`
    tool runs the check; the construction pipeline uses it as a filter.
  - **Novelty vs TemporalWiki**: TemporalWiki controls for entity-popularity bias but does
    not measure or control cross-period probe-answer leakage. Paper C adds that step.
  - **Dataset design**:
    - Source: CC-News or Wikipedia revision history with quarterly period slicing.
    - Filter: probes whose normalised answer appears in any other period's training targets
      or context are discarded at source.
    - Stability annotation: probes labelled `stable` / `updated` / `deprecated` per
      the cross-period diff protocol in `preprocessing/temporal.py`.
    - Target scale: 20-40 leakage-free probes per period × 12-20 periods.
  - **Cross-link**: paper-specific decisions in [`paper_c_ideas_journal.md`](paper_c_ideas_journal.md);
    leakage finding in
    [`paper_a_methodology_note_probe_leakage.md`](paper_a_methodology_note_probe_leakage.md).

### 2026-06-16 · User-instructed · Paper B direction — capacity-driven growth on a domain-specific dataset
- **Domain**: capacity-driven-cl
- **Source**: User-instructed
- **Status**: Approved-for-paper — this IS the main paper. See [`paper_main_outline.md`](paper_main_outline.md).
- **Where it might land**: **Main paper** (was "Paper B"); Thesis Ch. 2
- **Tags**: #open-question #survey-needed #dataset-candidate
- **Why it matters**: CAPSEL/INCA's core contribution is saturation-driven block expansion,
  not temporal CL. A domain-specific dataset (math + code, following LLaMA-Pro) lets the
  architecture paper stand independently from the temporal framing, sidesteps the leakage
  hazard entirely, and enables a direct comparison to LLaMA-Pro (Wu et al., 2024).
- **Body**:
  - **The argument**: Paper B isolates the architecture claim. If block expansion triggered
    by saturation detection outperforms fixed-schedule expansion (LLaMA-Pro) and no-growth
    baselines on a domain-specific dataset, that result holds regardless of whether the
    dataset is temporal.
  - **Candidate dataset stacks**:
    1. Math + Code (LLaMA-Pro replica): GSM8K training + CodeSearchNet. Direct LLaMA-Pro §4
       comparison.
    2. Sequential domain shift: science → law → medicine from RedPajama subsets. Each
       domain = one "period."
    3. Difficulty curriculum: easy → hard within a single domain.
  - **Ablations already wired**: `configs/ablations/e_sat.yaml`, `e_route.yaml`, `e_cls3.yaml`.
  - **Action**: read LLaMA-Pro §4; check if their block init is identity-init vs INCA's
    lateral-connection init.
  - **Cross-link**: paper-specific decisions in [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md).

### 2026-06-16 · User-instructed · Establish this research-ideas journal
- **Domain**: cross-cutting
- **Source**: User-instructed
- **Status**: Approved-for-paper (this file is the artefact)
- **Where it might land**: Process artefact — not in any paper
- **Tags**: #thesis-candidate
- **Why it matters**: Centralises the domain-level idea pool so cross-paper threads (Paper A,
  Paper B, thesis chapters) have one place to live, separate from per-paper journals.
- **Body**:
  - Append-only; reverse-chronological.
  - Claude does **not** add Claude-proposed entries without Nishant's explicit approval
    (consistent with the broader workflow preference in `MEMORY.md`).
  - Seeds below were added at user request to demonstrate the template — most are flagged
    `#survey-needed` because they are domain hypotheses that need a literature scan before
    they harden into commitments.

### 2026-06-16 · User-instructed · Should we build our own small online temporal dataset?
- **Domain**: online-temporal-data
- **Source**: User-instructed
- **Status**: Open — needs scoping decision; high time-cost
- **Where it might land**: Paper A §3 (if scoped tight) **or** a separate dataset paper
- **Tags**: #dataset-candidate #benchmark-design #survey-needed #open-question
- **Why it matters**: Every third-party temporal dataset we've tried has a defect (leakage,
  scale, format friction). If we own the construction, we own the defect-correction. But
  dataset construction is a known time-sink; needs scope discipline.
- **Body**:
  - **Candidate sources**: (i) Wikipedia revision-history slices with leakage-controlled probe
    construction (TemporalWiki-style); (ii) one news site's RSS archive with strict cross-period
    deduplication; (iii) ArXiv abstracts by month with citation-style fact probes;
    (iv) Github commit-message stream by date.
  - **Open sub-questions**:
    - What's the minimum viable scale for a credible CL temporal benchmark? (TemporalWiki-style
      bench has ~10k probes; CC-News v2 has ~200/period scored — large gap.)
    - Can we publish a *probe-only* benchmark on top of someone else's training stream?
      (Cheaper, less reviewer pushback, lets us critique CC-News / TiC-LM directly.)
  - **Cross-link**: paper-A-specific framing is logged at
    [paper_a_ideas_journal.md `2026-06-16 · User-instructed · Consider building our own small
    curated temporal dataset`].
  - **Action**: literature scan needed — survey what's wrong with existing temporal CL
    benchmarks; TemporalWiki §3, StreamingQA §2, TiC-LM §3 are the obvious starting points,
    plus the Lazaridou 2021 "Mind the Gap" paper.

### 2026-06-16 · User-instructed · Should CAPSEL use LLaMA-Pro's dataset stack instead of a temporal stream?
- **Domain**: capacity-driven-cl
- **Source**: User-instructed
- **Status**: Open — high-leverage scoping decision
- **Where it might land**: Paper A §1 framing + §3 dataset choice
- **Tags**: #open-question #survey-needed
- **Why it matters**: CAPSEL/INCA is fundamentally about **capacity growth under saturation**,
  not about temporal drift. The temporal-drift framing was inherited from the CC-News pilot.
  LLaMA-Pro (Wu et al., 2024) is the closest published analogue to what we're actually
  building and used a non-temporal (math + code) corpus. The dataset choice should follow the
  contribution, not the inherited pilot.
- **Body**:
  - **The genuine question**: do CAPSEL/INCA's headline claims (block-expansion timing,
    saturation detection, plasticity-vs-stability tradeoff) require a temporal axis at all?
    If not, math + code is methodologically cleaner because the leakage hazard disappears.
  - **What to check in the literature**:
    - LLaMA-Pro 2024 §4 — exact corpus mix and saturation behaviour.
    - Block-expansion ancestors (Net2Net 2015, Progressive Networks 2016) — what they used.
    - Whether any prior work has done block expansion on a temporal stream specifically.
  - **Cross-link**: paper-A-specific framing logged at
    [paper_a_ideas_journal.md `2026-06-16 · User-instructed · Consider LLaMA-Pro's dataset
    stack as an alternative to a temporal stream`].

### 2026-06-16 · Joint-session · Probe-leakage measurement is not a standard step in temporal-CL benchmark construction
- **Domain**: benchmark-design
- **Source**: Joint-session (emerged from the `analyze_probe_leakage.py` finding on CC-News v2)
- **Status**: Open — candidate methodology contribution
- **Where it might land**: Paper A Appendix A (concrete instance) + a methodology paper
- **Tags**: #literature-gap #metric-candidate #open-question
- **Why it matters**: No CL temporal benchmark we've encountered ships with a leakage matrix or
  a leakage-corrected probe split as a standard artefact. If we make the case that this should
  be a benchmark-publishing norm, it's a method contribution with leverage beyond Paper A.
- **Body**:
  - **Already in hand**: target_leakage and context_leakage matrices for CC-News v2 (75-83% /
    88-95% off-diagonal) and the analyzer script.
  - **Open sub-questions**:
    - Does TiC-LM Track A leak the same way? (Precondition for using it as headline.)
    - Does TemporalWiki leak? Lazaridou 2021's setup?
    - Is there a principled threshold ("L < 0.30 → publishable as drift bench") or is it
      necessarily benchmark-specific?
  - **Survey needed**: read TemporalWiki, TiC-LM, StreamingQA, Lazaridou 2021, Dhingra 2022
    methodology sections and check whether any of them measured train/test answer overlap at
    all.
  - **Cross-link**: paper-A-specific use logged at
    [paper_a_ideas_journal.md `2026-06-16 · Joint-session · CC-News leakage matrix as Appendix A`].

### 2026-06-16 · Joint-session · "What to forget" is rarely defined operationally in CL literature
- **Domain**: selective-forgetting
- **Source**: Joint-session (came up in CAPSEL roadmap context)
- **Status**: Open — thesis-chapter candidate
- **Where it might land**: Thesis Ch. 5 (probably); not Paper A
- **Tags**: #open-question #thesis-candidate #survey-needed
- **Why it matters**: Most CL papers treat forgetting as *uniformly bad*. But in a
  capacity-bounded model, forgetting *something* may be necessary to learn the next thing —
  and forgetting **stale** content is positively desirable for a temporal stream. The field
  has no standard operationalisation of "stale" vs "still-relevant" facts.
- **Body**:
  - **Open sub-questions**:
    - Is there a clean operationalisation of "stale" for a temporal-fact stream? (Fact has
      been overwritten in the world, or just hasn't been re-mentioned?)
    - Has unlearning / machine-unlearning literature (Bourtoule 2021, Kurmanji 2024) been
      connected to temporal CL? My read: not really.
    - Can saturation-detection (CAPSEL's core mechanism) be repurposed to detect *which*
      previously-learned facts should be released?
  - **Survey needed**: Bourtoule SISA, Kurmanji TOFU, plus the "forgetting is computation"
    line of work.

### 2026-06-16 · Joint-session · Backward-transfer (BWT) is gameable on leaky streams
- **Domain**: evaluation-methodology
- **Source**: Joint-session (direct corollary of CC-News leakage finding)
- **Status**: Open — methodology contribution candidate
- **Where it might land**: Paper A Appendix A discussion + methodology paper
- **Tags**: #open-question #metric-candidate #literature-gap
- **Why it matters**: A flat or *positive* BWT row on a leaky stream looks like a CL win but
  is actually re-exposure to the same answers. Any CL paper reporting BWT on a temporal news
  stream without a leakage audit is vulnerable to this critique.
- **Body**:
  - **Open sub-questions**:
    - What's the right "leakage-corrected BWT"? Drift-only subset BWT? Leakage-weighted BWT?
    - How many published CL temporal results are vulnerable to this critique? (Survey
      question — needs scan of recent CL benchmarks.)
  - **Cross-link**: tied to the leakage-measurement entry above; ideally both go into the same
    methodology paper.

---

## Document metadata

| Field        | Value                                                                                |
|--------------|--------------------------------------------------------------------------------------|
| Author       | Nishant Kumar (with Claude assistance)                                               |
| Created      | 2026-06-16                                                                           |
| Purpose      | Running log of research-domain ideas, open questions, and cross-paper threads        |
| Append rule  | New entries go at the top of "Entries"; never delete, only mark status               |
| Owner        | Nishant — all Claude-proposed entries require explicit approval before adding        |
| Sibling docs | [`paper_a_ideas_journal.md`](paper_a_ideas_journal.md) — Paper A: temporal CL on CC-News/TiC-LM |
|              | [`paper_b_ideas_journal.md`](paper_b_ideas_journal.md) — Paper B: capacity-driven growth        |
|              | [`paper_c_ideas_journal.md`](paper_c_ideas_journal.md) — Paper C: temporal benchmark dataset     |
|              | [`paper_d_ideas_journal.md`](paper_d_ideas_journal.md) — Paper D: temporal CL on clean stream    |

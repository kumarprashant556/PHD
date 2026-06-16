> **ARCHIVED 2026-06-16** — Superseded by [`docs/dataset_strategy_FINAL.md`](../dataset_strategy_FINAL.md),
> which merges this document's problem statement with the research report and engineering schema.
> Retained here as evidence/background only. Do not edit.

# Dataset Gap Analysis for CAPSEL/INCA Research

## What the existing benchmarks were designed for

Every temporal QA dataset in the literature was built to evaluate a model's *retrieval* accuracy at a fixed point in time — not to study how a model learns and forgets as time passes. That design choice creates a fundamental mismatch with what CAPSEL/INCA actually needs.

**StreamingQA** (2007–2020, WMT news) phrases every question as "is this fact currently true or outdated?" The binary framing is useful for retrieval systems but useless for measuring backward transfer. There are no period-indexed training documents to drive per-period fine-tuning, and the knowledge "conflict" is always between the frozen model and a test-time fact — never between two things the model was explicitly trained on in sequence.

**TempLAMA** sources everything from Wikidata edits. That gives clean timestamps but only a single knowledge source, and Wikidata edits are not uniform: some years have hundreds of occupation-change triples, others almost none. More importantly, most TempLAMA facts are *replacements* (person X held position Y from 2010–2014, then Z from 2014–present) rather than *updates with rationale*, so the dataset can't tell you whether a model understood *why* the fact changed — only whether it memorised the new value.

**ArchivalQA** is the largest (1M+ pairs) but it's news-only and treats every period as independent. It has no cross-period probes; you cannot construct a BWT matrix from it without substantial re-annotation.

**TimeQA** is human-annotated and high-quality, but annotation cost caps period coverage at roughly 12 annual snapshots of Wikipedia. Per-period training volume is far below what INCA needs to trigger a single block-growth event under realistic saturation thresholds.

**DriftMedQA** is the only domain-specific temporal benchmark, and it comes closest to what you need — it explicitly models fact *revision* (HIV and diabetes guideline changes) rather than fact replacement. But 195 QA pairs across two diseases is training noise rather than a training set. You couldn't train a single epoch on it without seeing every probe multiple times.

**ConflictMedQA** studies *internal* knowledge conflicts (contradictions already present in the model's weights) rather than temporal drift. Useful for a different problem.

---

## The structural gap

None of these datasets simultaneously provide all four things CAPSEL needs:

**1. Per-period training documents** — raw text the model can actually learn from. QA probe pairs alone are insufficient; the model needs the supporting corpus that *generates* those facts, otherwise seq2seq fine-tuning overfits immediately and you're measuring memorisation speed, not continual learning.

**2. Cross-period probe sets** — probes from period *t* that are re-evaluated after training on periods *t+1, t+2, …* These are the only way to compute an honest BWT matrix. No existing dataset ships these; researchers have to construct them post-hoc by hand, introducing annotation drift.

**3. Genuine fact update with supporting rationale** — not just "fact X became Y in year Z" but "the clinical guideline changed because study S showed outcome O." Without rationale, there's no way to distinguish a model that understood the update from one that simply overwrote its weights.

**4. Sufficient volume per period.** With FLAN-T5-small, you need roughly 200–500 training items per period to get a meaningful loss curve. StreamingQA can supply this for news; nothing domain-specific can.

---

## What the ideal dataset looks like

The dataset type that fills this gap is a **domain-specific temporal fact corpus with paired probes and rationale** — structured around a domain where facts update on a predictable, documented schedule and where the update process generates both (a) new training documents and (b) unambiguous old-vs-new probe pairs.

Medical/clinical is the most tractable domain because guideline bodies (WHO, CDC, NIH, NICE) publish versioned documents with explicit supersession relationships, dates, and reasoning. Drug label updates from the FDA have a similar structure. Each regulatory document revision is a period boundary; the old label is the training corpus for period *t*, the new label is the training corpus for period *t+1*, and the probes ask questions whose ground-truth answer flips between versions.

This gives you something no existing dataset provides: **the forgetting signal is built into the ground truth**. A probe whose correct answer was "150 mg twice daily" in period 3 and "200 mg once daily" in period 5 directly measures whether the model updated cleanly (learns new) or catastrophically (loses old context). Your BWT matrix entry at (5, 3) has a semantically meaningful interpretation.

### Practical structure

| Property | Target |
|---|---|
| Temporal periods | 15–30 (quarterly or semi-annual, 5–10 years) |
| Training documents per period | 500–2,000 (full guideline text, drug label sections, clinical summaries) |
| Probes per period | 100–300 |
| Probe types | Stable facts (forward transfer check), updated facts (core signal), deprecated facts (catastrophic forgetting check) |
| Construction method | Diff between consecutive guideline versions → candidate probes → human validation pass on ambiguous cases |

### Why this is a contribution

The closest precedent is DriftMedQA's construction process, but at ~20× scale and with the training corpus included. CAPSEL needs this dataset, and building it establishes both the benchmark and the research claim simultaneously — no existing benchmark can be used to reproduce CAPSEL/INCA results, which is a strong motivation for the community.

---

## Summary comparison

| Dataset | Training docs | Cross-period probes | Update rationale | Volume/period | Domain |
|---|---|---|---|---|---|
| StreamingQA | No | No | No | High | News |
| TempLAMA | No | No | No | Low-medium | Wikipedia |
| ArchivalQA | No | No | No | High | News |
| TimeQA | No | No | No | Low | Wikipedia |
| DriftMedQA | No | No | Partial | Very low (195 total) | Medical |
| ConflictMedQA | No | No | No | Low | Medical |
| **Ideal (proposed)** | **Yes** | **Yes** | **Yes** | **Medium-high** | **Medical/domain** |

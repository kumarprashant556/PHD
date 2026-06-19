# CAPSEL/INCA — Dataset Strategy (FINAL / canonical)

Date: 2026-06-15 · Status: **decision document — use this one**

This supersedes and merges three inputs (all retained as evidence):
- `docs/dataset_gap_analysis.md` — the problem statement (what's missing in existing benchmarks).
- `docs/dataset_research_report.md` — web research + evidence (Claude).
- `docs/dataset_strategy_report.md` — engineering schemas + DailyMed path (Codex).

Two independent analyses converged on the same plan; this doc is the reconciled, actionable version.

---

## The decision in one screen

**Reframe:** stop treating "the dataset" as one artifact. For CAPSEL the unit is a **temporal data system**:
```
period documents ─► training tasks   (completion + salient-span)
period documents ─► probe generation (cloze / QA / MCQ)
probes from period t ─► re-evaluated after every future period  (BWT matrix)
versioned updates    ─► high-precision update + rationale tests   (DailyMed/FDA)
```

**Training format:** **temporal text-to-text completion = primary**, **salient-span denoising = auxiliary**
(decide the mix by ablation, then keep it). Every input is period-prefixed (`period: 2018-H1`).

**Evaluation format:** cloze + QA + (limited) MCQ probes, **frozen per period, stored separately, re-run across
all later periods**. BWT/FWT computed over **probe accuracy**, never perplexity. Adopt TiC-LM's regret-matrix
definitions so numbers are comparable to the 2025 ACL benchmark.

**Dataset stack (3 stages, gated):**
1. **Phase 1 pilot → CC-News** (already wired) + generated probes.
2. **Paper A main → TiC-LM / TiC-CC** (+ its TiC-Wiki/StackExchange/CodeDocs evals). **FineWeb** is the drop-in
   fallback if reproducing TiC-CC via Apple's pipeline is too heavy.
3. **Later extension → DailyMed/FDA SPL** versioned drug labels for genuine fact-update-with-rationale probes;
   NICE/CDC/WHO guidelines only if rationale becomes a headline contribution.

**Rejected:** mC4 / C4 / OSCAR / multilingual CC — multilingual ≠ temporal; wrong axis, adds a confound.

---

## 1. Why format is constrained by the architecture (not just preference)

INCA does not merely score final accuracy; it needs **learning curves, saturation signals, freeze/grow
decisions, and BWT/FWT matrices**. A valid training format must satisfy five constraints:

1. **Dense learning signal** — many target tokens/example, not one short answer.
2. **Stable loss curves** — enough examples/period to read plateau / grad-norm / CKA signals.
3. **Temporal locality** — every example belongs to exactly one period.
4. **Probe derivability** — the same raw stream yields period-tied evaluation probes.
5. **Cross-period re-evaluation** — period-`t` probes are stored and re-tested after `t+1…T`.

The decisive point: **the saturation detector (RIR, grad-norm EMA, CKA drift, loss-plateau) is a function of a
dense, smooth scalar loss.** Completion and span-corruption provide it; cloze/MCQ/QA accuracy is too sparse and
noisy at FLAN-T5-base's ~30–35% F1 ceiling to *train* on. → sparse formats are **evaluation only**.

---

## 2. Training format

### Primary — prefix-to-continuation completion
Dense supervised tokens, label-free, real temporal drift, mirrors to causal next-token for Track B (Pythia).
Completion beats summarization/QA *for training* because those compress the document into a tiny target the
model can memorize without learning the period distribution. Period-prefix every input (cheap temporal handle;
TempLAMA shows time-prefixed models refresh ~30× cheaper without degrading old periods — arXiv 2106.15110).

```json
{
  "id": "ccnews_2018h1_000001", "period": "2018-H1", "source": "cc_news",
  "date": "2018-04-24", "url": "...", "domain": "...", "title": "...",
  "text": "full document text",
  "task": "completion",
  "input": "period: 2018-H1\ncomplete: <first 40-60% of document>",
  "target": "<next 128-256 tokens>",
  "probe_split": "train"
}
```

### Auxiliary — salient-span denoising (recommended, decide by ablation)
Mask entities/dates (NER + regex), not random spans. Evidence that *what* you mask matters more than the
mechanism: fact-targeted masking improves factual recall (+~6 pts LAMA; +5.8 avg on temporal tasks)
(arXiv 2204.07994; 2303.12860; EACL-2023 2023.eacl-main.222). Because BWT measures *knowledge* retention, a
fact-targeted objective is more aligned with the metric than generic completion. Use as a 10–30% mixture.

```json
{
  "task": "salient_span_denoising",
  "input": "period: 2018-H1\ndenoise: The company appointed <extra_id_0> as CEO in <extra_id_1>.",
  "target": "<extra_id_0> Jane Smith <extra_id_1> 2018"
}
```

### Format ablation `E-FORMAT` (add to Paper A §5)
- `F-COMP`  — completion only (default / baseline)
- `F-DENOISE` — salient-span denoising only
- `F-MIX`   — 70% completion + 30% salient-span denoising

Hypothesis: `F-MIX` gives the best knowledge acquisition + a clean BWT signal; `F-COMP` gives the smoothest
saturation signal. Reporting the trade-off pre-empts the reviewer question "why not T5's own objective?"

---

## 3. Evaluation format (probes)

Three probe types, **frozen at period creation, re-evaluated after every later period**:
- **Continuation probes** — held-out prefix→continuation loss by period (sanity/style; not the headline).
- **Entity/date cloze** — mask people/orgs/products/dates/law/drug names (primary factual signal).
- **QA probes** — generated from evidence sentences, lightly validated.
- **MCQ** — optional, easy accuracy, but never the core metric (distractor quality dominates).

```json
{
  "probe_id": "ccnews_2018h1_entity_0001", "origin_period": "2018-H1",
  "source_doc_id": "...", "probe_type": "entity_cloze",
  "input": "period: 2018-H1\nquestion: In 2018-H1, who was appointed CEO of X?",
  "answer": "Jane Smith", "aliases": ["J. Smith"], "answer_type": "person",
  "stability": "updated|stable|deprecated",  /* = CAPSEL's 3 probe types */
  "evidence": "...",
  "eval_after_periods": ["2018-H1","2018-H2","2019-H1","..."]
}
```

**BWT/FWT computation:** build a regret matrix `R[i,j]` = model-after-period-`i` evaluated on period-`j`'s
frozen probes; In-distribution = diagonal, **BWT = lower triangle, FWT = upper triangle** (TiC-LM,
aclanthology 2025.acl-long.1551). Compute over **probe accuracy/EM**, and report a PPL-based version only as a
secondary sanity check (PPL is a style metric — `docs/QnA_Models_Report.docx`).

The `stability` field maps directly to CAPSEL's probe taxonomy: `stable`=forward-transfer check,
`updated`=core forgetting signal, `deprecated`=catastrophic-forgetting check.

---

## 4. Dataset verdict table

| Source | Use? | Why |
|---|---|---|
| **CC-News** (`vblagoje/cc_news`) | ✅ Phase 1 | Dated news, enough volume, already in repo. No native cross-period probes → generate them. |
| **TiC-LM / TiC-CC** (arXiv 2504.02107) | ✅ Paper A | Purpose-built time-continual benchmark: 114 monthly CC dumps, temporal evals, regret-matrix BWT/FWT, code released. Closes gap reqs 1,2,4. |
| **FineWeb** (arXiv 2406.17557) | ✅ Fallback/aux | Streamable by CC `dump`/`date`, has `token_count`; drop-in if TiC-CC reproduction too heavy. Not a benchmark itself. |
| **RedPajama-V2** (arXiv 2411.12372) | ◐ Maybe | 84 snapshots, 30T tokens, quality signals. Useful for raw data eng; heavy for the main path. |
| **Dolma** (arXiv 2402.00159) | ✗ primary | Great open corpus, not period-by-period. |
| **C4** | ✗ primary | Single Apr-2019 CC snapshot — wrong temporal structure. |
| **mC4 / OSCAR / multilingual** | ✗ | Multilingual ≠ temporal; adds confound, no cross-period probes. |
| **TemporalWiki** (2204.14211) | ◐ Eval/aux | TWiki-Diffsets (changed facts) + TWiki-Probes; diffsets can double as training. |
| **CKL** (2110.03215) | ◐ Eval/aux | Invariant/Updated/New LAMA = retention/update/acquisition decomposition. |
| **StreamingQA / RealtimeQA / TRACE** | ◐ Eval only | Retrieval/instruction-oriented temporal checks. |
| **EvolvingQA / TAQA / ChroKnowBench** | ◐ Eval only | Good temporal-knowledge probes; not raw training corpora. |
| **DailyMed / FDA SPL** | ✅ Build (later) | Lowest-effort *versioned* fact updates: `setid`, `spl_version`, `published_date`; diff label sections. |
| **NICE/CDC/WHO/NIH guidelines** | ◐ Build (optional) | Rationale-rich, but scraping/version-alignment/validation costly. |

---

## 5. Three-stage plan with gates

**Stage 1 — CC-News pilot (now).** Train completion (+ optional F-DENOISE ablation); probes = held-out
completion loss + generated entity/date cloze + small validated QA set; periods = 6 half-years 2017–2019.
**Gate:** INCA must beat LLaMA-Pro/fixed-growth on BWT before any TiC-LM run.

**Stage 2 — TiC-LM main run (Paper A).** Train TiC-CC (monthly or coarsened); eval TiC-CC held-out +
TiC-Wiki/StackExchange/CodeDocs; adopt their BWT/FWT defs. **FineWeb fallback** if TiC-CC reproduction blocks.

**Stage 3 — DailyMed/FDA update suite (after Stage 1 passes).** Diff consecutive SPL label versions; extract
changed sections (indications, dosage, contraindications, warnings, adverse reactions, interactions); generate
old-vs-new probes with `stability` labels. This delivers the "real fact update with rationale" story general
web corpora cannot. Optional escalation to NICE/CDC/WHO guidelines for richer rationale.

---

## 6. Build vs reuse + effort

Do **not** build a full dataset before Paper A (that turns the PhD into a dataset project before the model claim
is validated). Instead: reuse CC-News + TiC-LM for training; **build one reusable probe-generation layer**
(`data/probe_gen.py`) that runs on CC-News, FineWeb, and TiC-CC docs; build the DailyMed suite only after the
pilot passes.

| Work item | Effort | Value |
|---|---|---|
| CC-News date slicing + completion format | 2–4 days | immediate pilot (mostly done) |
| Entity/date cloze probe generator | 1–2 wk | **needed for BWT beyond perplexity — do first** |
| QA probe generator + validation | 1–2 wk | stronger factual metric |
| Salient-span denoising + `E-FORMAT` ablation | ~1 wk | method contribution |
| TiC-LM reproduction/adaptation | 2–4 wk | Paper A benchmark |
| FineWeb fallback stream | ~1 wk | insurance |
| DailyMed versioned probe suite | 3–6 wk | strong domain contribution |
| NICE/CDC/WHO rationale corpus | 2–4 mo | possible separate benchmark paper |

---

## 7. Sources
TiC-LM arXiv 2504.02107 / aclanthology 2025.acl-long.1551 / github.com/apple/ml-tic-lm ·
FineWeb arXiv 2406.17557 / HF HuggingFaceFW/fineweb · RedPajama-V2 arXiv 2411.12372 ·
Dolma arXiv 2402.00159 · CC-News HF vblagoje/cc_news · T5 arXiv 1910.10683 ·
Salient-span/temporal masking arXiv 2303.12860, 2204.07994; masking strategies arXiv 2306.07185;
EACL-2023 2023.eacl-main.222 · TempLAMA/temporal-prefix arXiv 2106.15110 ·
CKL arXiv 2110.03215 · TemporalWiki arXiv 2204.14211 · StreamingQA arXiv 2205.11388 ·
EvolvingQA arXiv 2311.08106 · TAQA arXiv 2402.16797 · ChroKnowBench arXiv 2410.09870 ·
knowledge-update objective misalignment arXiv 2411.04448 ·
DailyMed SPL API dailymed.nlm.nih.gov · FDA SPL fda.gov.

*Verification caveat: the deep-research workflow's automated verifier crashed (abstained on all claims); the
evidence above reflects source-grounded human/Claude judgment, corroborated by Codex's independent report.*

# Dataset & Data-Format Research Report for CAPSEL/INCA

**Question:** What dataset(s) and what data *format* best serve a saturation-driven, continually-growing
FLAN-T5 LM whose headline metrics are backward/forward transfer (BWT/FWT) over temporal periods?

**Method note (read this first).** This report was produced by a fan-out web-research workflow (5 angles,
25 sources fetched, 118 candidate claims extracted) followed by my own synthesis. The workflow's automated
*verification* stage hit a tooling failure (every verifier agent failed to emit its structured verdict, so
all claims defaulted to a spurious "refuted / 0-0 abstain"). That is a harness bug, **not** a genuine
refutation — the underlying sources are real and high-quality. I have therefore re-judged each claim against
the primary sources and my own knowledge rather than trusting the crashed verifier. Sources are cited inline.

---

## TL;DR — the four recommendations

1. **Format (training):** Keep **seq2seq completion / prefix-continuation** as the *default* per-period
   training objective, and **add salient-span (fact-targeted) masking as a first-class ablation** (`E-FORMAT`).
   Completion wins on *signal density for the saturation detector*; salient-span masking wins on *knowledge
   acquisition*, which is exactly what BWT measures. This tension is a genuine contribution, not a footnote.
2. **Format (evaluation):** **Decouple probe format from training format.** Measure BWT/FWT with
   **accuracy-based cloze + MCQ probes** (LAMA-style, auto-generated, NER/relation-anchored), never with
   completion/PPL loss. Perplexity is a *style* metric, not a *knowledge* metric.
3. **mC4 / multilingual:** **Wrong axis — skip it.** Multilingual ≠ temporal. It adds a language-distribution
   confound that contaminates "forgetting-over-time" and buys nothing for this thesis. Only relevant if a
   *future* research question is explicitly cross-lingual CL.
4. **Dataset stack:** Phase 1 → **CC-News completion + auto-generated per-period probes** (already wired).
   Paper A main run → **TiC-LM / TiC-CC training stream + TiC-Wiki/StackExchange/CodeDocs probes** using
   TiC-LM's own regret-matrix BWT/FWT definitions. Scale/volume headroom → **FineWeb** or **RedPajama-V2**.
   **Do not** build the medical-guideline corpus as a Phase 1–4 dependency — scope it as an optional
   standalone dataset contribution / future work.

---

## 1. The format question

### 1.1 A constraint the gap analysis missed: the saturation detector needs a *dense, smooth scalar loss*

CAPSEL's growth trigger is a multi-signal consensus over **RIR, grad-norm EMA decay, CKA drift, and loss
plateau slope** (Master Reference §4). Three of those four signals are functions of a *continuous training
loss curve*. This places a hard constraint on the training format that is independent of "which dataset":

- **Completion / prefix-continuation** and **span-corruption denoising** produce a dense, low-variance
  per-step loss — every document contributes a smooth signal. The plateau/RIR/grad-norm signals are
  well-defined and stable. ✅
- **Cloze / MCQ / QA-pair** objectives produce *sparse, high-variance accuracy-like signals*. At FLAN-T5-base
  scale the factual-QA ceiling is only ~30–35% token-F1 (Master Reference §4.1; RealtimeQA), so a loss built
  on them is noisy and would make the saturation detector fire erratically. ❌ as a *training* objective.

**Implication:** the *training* objective must be a dense self-supervised one (completion or span-corruption).
The sparse accuracy objectives belong in *evaluation*, not training. This is the strongest single argument for
the project's current choice — and it is a better justification than the one in the training-data doc.

### 1.2 Is the doc right that "completion beats summarization/QA for pretraining"?

**Partly.** The doc's reasoning (Master Reference §17.1) is correct on two points and incomplete on one:

- ✅ **Correct:** summarization datasets (CNN/DM, XSum) are static (document→summary) pairs with *no temporal
  drift*, so they cannot exercise CAPSEL's core capability. QA datasets are too *sparse* per period.
- ✅ **Correct:** completion is label-free, derived from the document itself, and has genuine year-over-year
  distribution shift.
- ⚠️ **Incomplete:** it conflates "pre-training signal density" with "knowledge injection." For *knowledge
  acquisition specifically* — the thing BWT measures — there is strong evidence that **fact-targeted masking
  beats both generic completion and random span-corruption.**

### 1.3 The salient-span-masking evidence (why completion may be leaving knowledge on the table)

Multiple primary sources converge: *what* you mask matters more than the corruption mechanism.

- **Salient Span Masking (SSM)** masks named entities and dates (NER + regex), oversampling factual content,
  in T5's span-corruption format. Masking knowledge-bearing tokens rather than random tokens improves factual
  recall by **+6.1 pts on LAMA-SQuAD (39.7% vs 33.6%)** and **+5.7–6.7 pts on closed-book QA**
  (arXiv 2204.07994). The standard random-masking objective is *suboptimal for knowledge* because models give
  insufficient attention to knowledge-bearing tokens.
- SSM as intermediate pre-training improves three temporal tasks by **+5.8 pts avg** over base T5; a
  temporal-targeted variant (TSM) adds a further +0.29; the best is a mixture (EACL 2023, 2023.eacl-main.222).
- The **knowledge-update literature** finds a *misalignment between the autoregressive LM objective and the
  parameter updates optimal for acquiring/updating knowledge* (arXiv 2411.04448) — i.e. the default objective
  is not the knowledge-optimal objective. Naive sequential training both under-absorbs new facts and
  catastrophically forgets old ones — which is precisely CAPSEL's motivation.

**Recommendation:** make format an *ablation axis*, `E-FORMAT`:
`completion` (default) vs `span-corruption` (T5-native) vs `salient-span masking` (fact-targeted).
Hypothesis: salient-span masking yields **higher per-period knowledge acquisition and a cleaner BWT signal**,
while completion yields the **smoother saturation signal**. Reporting this trade-off strengthens Paper A's
method section and pre-empts the obvious reviewer question ("why completion and not T5's own objective?").

### 1.4 Temporal-prefix conditioning (a cheap, proven knob)

TempLAMA's "Temporal" model conditions on time simply by **prepending a string timestamp** (`year: 2014`) to
the input — no architecture change — and shows a time-conditioned model can be **"refreshed" on new data with
~30× fewer steps without degrading older-period knowledge** (Dhingra et al., arXiv 2106.15110). For CAPSEL
this is a near-free addition: prefix each period's `complete:` input with its period id. It gives the selector
/ router an explicit temporal handle and is itself an ablation (`with/without temporal prefix`).

### 1.5 Format summary table

| Format | Role | Verdict for CAPSEL |
|---|---|---|
| Seq2seq completion (current) | **Training (default)** | Dense, smooth loss → best for saturation signal; label-free; real drift |
| Salient-span masking (SSM) | **Training (ablation, recommended)** | Best knowledge acquisition; aligns with what BWT measures |
| Span-corruption (T5-native) | Training (ablation/control) | Dense; the "did we need fact-targeting?" control |
| Cloze / fill-in-blank | **Evaluation probe** | Clean cross-period BWT probes; too sparse to *train* on |
| Multiple-choice (MCQ) | **Evaluation probe** | Interpretable accuracy; auto-generatable; not a training objective |
| Instruction / QA pairs | Eval / optional SFT | Sparse; needs labels; not for continual pre-training |
| Plain causal next-token | Training (Track B / Pythia) | Fine for decoder-only track; same density argument |

---

## 2. mC4 / multilingual — clear verdict: **wrong axis**

- **Multilingual is orthogonal to temporal.** CAPSEL's independent variable is *time*; BWT/FWT are measured
  *across periods*. Adding languages introduces a *second* distribution-shift axis (language) that confounds
  the forgetting-over-time signal you are trying to isolate. Keep one axis clean.
- **mC4/C4 lack usable per-document temporal structure.** C4 is essentially a single 2019 Common-Crawl
  snapshot; mC4 is the multilingual extension of the same. Neither exposes the per-document dates or per-dump
  timestamps you need to slice into periods (contrast CC-News's `date` field and FineWeb/RedPajama's CC-dump
  tagging — §3).
- **When multilingual *would* matter:** only if a future paper poses a *cross-lingual continual learning*
  question (e.g. "does growth transfer across languages?"). That is a different thesis. Note it as future work;
  do not put it on the Paper A critical path.

---

## 3. Large temporal corpora (per-period training streams)

| Corpus | Temporal slicing | Size | Per-period volume | License / access | Fit |
|---|---|---|---|---|---|
| **TiC-LM / TiC-CC** (apple/ml-tic-lm; ACL 2025, arXiv 2504.02107) | **114 monthly CC dumps, May 2013–Jul 2024** | ~2.9T tokens | Very high | Apple pipeline (reproduce from CC) | ★ **Best fit** — see §3.1 |
| **FineWeb** (HuggingFaceFW/fineweb; arXiv 2406.17557) | **96+ CC dumps 2013–2024**, each a name-selectable subset (`CC-MAIN-2024-10`) | ~15T tokens | Effectively unlimited | ODC-By, trivial HF streaming | ★ Best *raw-volume* slicer |
| **RedPajama-Data-V2** (together.ai) | **84 CC snapshots**, `snapshots=[...]` param | 30T filtered tokens | Effectively unlimited | permissive, HF streaming | ★ Big + has quality signals |
| **CC-News** (vblagoje/cc_news) | per-doc `date` field, 2017–2019 | 708K articles | ~55–72K/period (half-year) | HF, easy | ✅ Phase 1 default (already wired) |
| Dolma / Pile / Pile-CC | weak/none per-doc dates | large | n/a | open | ✗ no clean temporal axis |

### 3.1 Why TiC-LM is the answer for Paper A (it already solves 3 of the 4 "gaps")

The gap analysis concluded "no existing dataset provides all four requirements." That is true for the
*medical/rationale* angle, but **TiC-LM substantially closes the gap for general knowledge** and — critically —
**ships the exact BWT/FWT machinery CAPSEL needs**:

- **(1) per-period training docs:** TiC-CC = 114 monthly Common Crawl dumps, directly sliceable by month.
- **(2) cross-period probes:** time-stratified eval suites **TiC-Wikipedia, TiC-StackExchange, TiC-CodeDocs**,
  paired with the training corpus.
- **(4) volume per period:** web-scale; far above the ~200–2000 docs/period CAPSEL needs.
- **BWT/FWT framework, for free:** TiC-LM defines In-distribution (diagonal), **Backward Transfer (lower
  triangle), Forward Transfer (upper triangle)** over a regret matrix `R[i,j]` = model-after-month-i evaluated
  on month-j (aclanthology 2025.acl-long.1551). This is *identical* to CAPSEL's BWT matrix — adopt their
  definitions so your numbers are directly comparable to a 2025 ACL benchmark.
- **Bonus CL finding to cite:** in TiC-LM, replay of older data is essential to avoid forgetting on *generic
  web* data but matters much less for *specific domains* — directly relevant to CAPSEL's per-block replay story.

The only requirement TiC-LM does *not* give you is **(3) fact updates with explicit rationale** — and that is a
*nice-to-have for the CLS/analysis chapter*, not a prerequisite for the core result. See §5.

**Practical note:** TiC-CC must be reproduced via Apple's CC pipeline (heavier than a plain HF download). If
that is a bottleneck, **FineWeb is the drop-in substitute** for the raw per-period training stream
(`name="CC-MAIN-YYYY-WW"`), paired with TiC-LM's eval suites or your own auto-generated probes.

---

## 4. Cross-period evaluation benchmarks (the probe side)

Pair a raw training corpus with these to manufacture honest BWT matrices:

| Benchmark | What it gives | Maps to |
|---|---|---|
| **TiC-LM eval suite** (TiC-Wiki/StackExchange/CodeDocs) | time-stratified probes paired with TiC-CC | **Primary** BWT/FWT for Paper A |
| **CKL** (Jang et al. 2022, arXiv 2110.03215) | InvariantLAMA / UpdatedLAMA / NewLAMA | retention vs **update** vs acquisition — clean FWT/BWT decomposition |
| **TemporalWiki** (arXiv 2204.14211) | **TWiki-Diffsets** (changed facts = *training*) + **TWiki-Probes** | both a training signal *and* probes for changed knowledge |
| **TempLAMA** (arXiv 2106.15110) | 50,310 cloze probes, same template → different answer per year | textbook cross-period probe; template for auto-gen |
| **GrowOVER** (ACL 2024, arXiv 2406.05606) | continuously-updated QA + Dialogue | newer dynamic probe set |
| **EvolvingQA** (NAACL 2024, arXiv 2311.08106) | Wikipedia-based knowledge-update QA | update-focused probes |
| **ChroKnowledge / ChroKnowBench** (dmis-lab) | chronological knowledge across domains incl. biomedical; time-variant vs invariant split | domain-stratified temporal probes |
| StreamingQA, RealtimeQA, TRACE, TAQA | existing secondary evals | sanity / breadth |

**Key pattern:** CKL and TemporalWiki are the cleanest because they *separate invariant / updated / new*
knowledge — which is exactly the FWT (new), BWT (retained), and update axes. TemporalWiki's Diffsets even
double as a *training* stream of changed facts.

---

## 5. Build vs reuse — the cross-period-probe problem

**Verdict: reuse for Paper A; auto-generate to fill gaps; treat the medical corpus as optional future work.**

The four requirements decompose cleanly by effort:

| Requirement | Lowest-effort path | Effort |
|---|---|---|
| (1) per-period training docs | **Reuse** TiC-CC / FineWeb / CC-News | days |
| (2) cross-period probes | **Reuse** TiC-LM/CKL/TemporalWiki **+ auto-generate** NER-anchored cloze/MCQ on your training corpus | 1–2 weeks for the generator |
| (4) volume per period | **Reuse** (web-scale corpora have headroom) | none |
| (3) updates *with rationale* | **Build** (no existing resource gives rationale + supersession at scale) | 2–4 months — optional |

### 5.1 Auto-generate probes (the cheap win)

The QnA report (`docs/QnA_Models_Report.docx`) already prescribes this: **NER-anchored question/cloze
generation** — extract entities/dates → fill templates → add MCQ distractors → confidence-filter. TempLAMA and
CKL prove KB/relation-anchored cloze works. Build one small pipeline (`data/probe_gen.py`) and you can mint
cross-period probes for *any* temporal corpus (CC-News and TiC-CC alike), giving you requirement (2) without
hand annotation. This is the single highest-leverage data task after Phase 0.

### 5.2 On the medical-guideline corpus (gap-analysis proposal)

It is the *only* path to requirement (3) — genuine fact updates with documented rationale and supersession
(WHO/CDC/NIH/NICE/FDA versioned docs). But:

- It is a **multi-month build** (scrape versioned docs → diff consecutive versions → candidate probes → human
  validation), i.e. a dataset-contribution project in its own right.
- It is **not required** for Paper A's core claim (INCA beats LLaMA-Pro on BWT). It would strengthen the
  *analysis/CLS* story and could be a **third contribution, a thesis chapter, or a standalone dataset paper.**
- **Lower-effort proxies for "fact update" structure** exist: **Wikipedia revision histories** and
  **TemporalWiki Diffsets** already capture old→new fact changes (without curated clinical rationale). Use
  these first; only build the medical corpus if you specifically want the *rationale/supersession* angle as a
  headline contribution.

**Recommendation:** do **not** gate Phases 1–4 on building a new corpus. Reuse TiC-LM + auto-generated probes.
Park the medical corpus as an explicit, scoped *optional contribution* with a go/no-go after Paper A.

---

## 6. Concrete dataset stack (drop-in for the roadmap)

**Phase 1 (pilot, CC-News scale):**
- Train: CC-News completion, 6 half-year periods, ~20K docs/period (already wired). Add temporal prefix.
- Eval: auto-generated NER-anchored cloze + MCQ probes per period → BWT matrix. (Build `data/probe_gen.py`.)
- Optional ablation: `E-FORMAT` (completion vs span-corruption vs salient-span) on this small, fast setup.

**Paper A main run (TiC-LM scale):**
- Train: **TiC-CC** monthly slices (or **FineWeb** `CC-MAIN-*` subsets if the Apple pipeline is too heavy).
- Eval: **TiC-Wiki / TiC-StackExchange / TiC-CodeDocs** + **CKL (Invariant/Updated/New)** for the
  retention/update/acquisition decomposition. Adopt TiC-LM's regret-matrix BWT/FWT definitions verbatim.
- Scale test (E-SCALE): FineWeb or RedPajama-V2 slices (unlimited volume).

**Skip:** mC4 / C4 / OSCAR / multilingual (wrong axis).

---

## 7. Sources

**Format / objective:** SSM & knowledge masking — arXiv 2204.07994; EACL 2023 (aclanthology 2023.eacl-main.222);
knowledge-update objective misalignment — arXiv 2411.04448; TempLAMA + temporal prefix + cheap refresh —
Dhingra et al. arXiv 2106.15110.
**Corpora:** TiC-LM — arXiv 2504.02107 / aclanthology 2025.acl-long.1551 / github.com/apple/ml-tic-lm;
RedPajama-V2 — together.ai/blog/redpajama-data-v2; FineWeb — huggingface.co/datasets/HuggingFaceFW/fineweb
(arXiv 2406.17557); CC-News — huggingface.co/datasets/vblagoje/cc_news.
**Eval benchmarks:** CKL — arXiv 2110.03215 / github.com/joeljang/continual-knowledge-learning;
TemporalWiki — arXiv 2204.14211 / github.com/joeljang/temporalwiki; GrowOVER — aclanthology 2024.acl-long.181 /
arXiv 2406.05606; EvolvingQA — arXiv 2311.08106; ChroKnowledge/ChroKnowBench — github.com/dmis-lab/ChroKnowledge;
StreamingQA — Liška et al. 2022.

*(Verifier-stage caveat repeated: the automated 3-vote verification crashed and abstained on all claims; the
above reflects my own source-grounded judgment, not the workflow's verdict.)*

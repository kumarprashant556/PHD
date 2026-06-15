# CAPSEL/INCA Dataset Strategy Report

Date: 2026-06-15

## Executive Decision

The project should **not** use mC4/C4 as the main dataset. mC4 is large and multilingual, but multilingual coverage is not the same thing as temporal drift. CAPSEL/INCA needs period-indexed learning streams and cross-period probes, not just more languages.

The recommended stack is:

1. **Phase 1 pilot:** CC-News, sliced by publication date, with generated continuation + entity/date cloze probes.
2. **Paper A main run:** TiC-LM / TiC-CC, because it is explicitly a time-continual LM benchmark built from 114 Common Crawl dumps and already includes temporal evaluation construction code.
3. **Fallback or auxiliary corpus:** FineWeb, because it exposes Common Crawl `dump`, crawl `date`, URL, and token count fields and can be streamed by individual dump.
4. **Scientific extension:** build a small versioned medical/regulatory corpus from DailyMed/FDA labels and/or clinical guideline updates. This should not block Paper A; it should become a high-precision “fact revision with rationale” probe suite after the architecture shows signal on CC-News/TiC-LM.

The recommended **training format** is:

> **Temporal text-to-text completion as the primary training objective, plus optional salient-span denoising over entities/dates as an auxiliary objective.**

QA, cloze, and multiple-choice should be treated primarily as **evaluation/probe formats**, not the main training stream.

## Why Dataset Format Matters For INCA

INCA is not just measuring final task accuracy. It needs learning curves, saturation signals, freezing/growth decisions, and BWT/FWT matrices. Therefore the training data must create enough gradient signal per period for the saturation detector to mean something.

A good format must satisfy five constraints:

1. **Dense learning signal:** many target tokens per example, not one short answer.
2. **Stable loss curves:** enough examples per period to observe plateau/gradient/representation signals.
3. **Temporal locality:** every example must belong to a period.
4. **Probe derivability:** the same raw document stream should generate evaluation probes tied to that period.
5. **Cross-period re-evaluation:** period `t` probes must be stored and tested again after training on `t+1...T`.

## Format Recommendation

### Primary format: prefix-to-continuation completion

Use this for the main continual-pretraining stream:

```json
{
  "id": "ccnews_2018h1_000001",
  "period": "2018-H1",
  "source": "cc_news",
  "date": "2018-04-24",
  "url": "...",
  "domain": "...",
  "title": "...",
  "text": "full article or document text",
  "task": "completion",
  "input": "period: 2018-H1\ncomplete: <first 40-60% of document>",
  "target": "<next 128-256 tokens>",
  "probe_split": "train"
}
```

Why this works:

- It gives many supervised target tokens per example.
- It is compatible with FLAN-T5 as text-to-text learning.
- It makes loss, gradient norm, and plateau signals meaningful.
- It avoids the brittleness of generated QA as a training objective.
- It can be mirrored for decoder-only Track B as causal next-token learning.

Completion is better than summarization/QA for the **training** stage because summarization and QA compress the document into a small target. That small target is attractive for evaluation but weak for saturation detection: the model may memorize a few labels or answer spans without learning the period distribution. Completion forces the model to model the local language and facts of the period.

### Auxiliary format: salient span denoising

Use this as an optional 10-30% mixture, especially for entity/date-heavy documents:

```json
{
  "task": "salient_span_denoising",
  "input": "period: 2018-H1\ndenoise: The company appointed <extra_id_0> as CEO in <extra_id_1>.",
  "target": "<extra_id_0> Jane Smith <extra_id_1> 2018"
}
```

This is useful because salient span masking explicitly oversamples entities and dates. The temporal-span masking work reports that salient span masking improves temporal downstream tasks, and masking-strategy work finds that entity/correlated-span masking retains more factual knowledge than random token masking. For INCA, that means better pressure on the facts likely to change over time.

Do **not** replace completion with denoising immediately. Treat it as an ablation:

- `F-COMP`: completion only
- `F-DENOISE`: salient span denoising only
- `F-MIX`: 70% completion + 30% salient span denoising

### Evaluation formats

Use three probe types, stored separately from training:

```json
{
  "probe_id": "ccnews_2018h1_entity_0001",
  "origin_period": "2018-H1",
  "source_doc_id": "...",
  "probe_type": "entity_cloze",
  "question": "In 2018-H1, who was appointed CEO of X?",
  "input": "period: 2018-H1\nquestion: ...",
  "answer": "Jane Smith",
  "aliases": ["J. Smith"],
  "answer_type": "person",
  "stability": "updated|stable|deprecated",
  "evidence": "...",
  "eval_after_periods": ["2018-H1", "2018-H2", "2019-H1", "..."]
}
```

Recommended probes:

- **Continuation probes:** held-out prefix-to-continuation loss by period.
- **Entity/date cloze probes:** masks people, organizations, products, dates, law/drug names.
- **QA probes:** generated from evidence sentences and lightly validated.

Multiple-choice can be included for easy accuracy scoring, but it should not be the core metric because distractor quality can dominate the result.

## Verdict On Dataset Options

| Dataset/source | Use? | Why |
|---|---:|---|
| **CC-News** | Yes, Phase 1 | Manageable, dated news documents, enough volume, already in repo. Weakness: no native cross-period probes. |
| **TiC-LM / TiC-CC** | Yes, Paper A | Best match to time-continual pretraining: 114 Common Crawl months, temporal held-out evals, code released. |
| **FineWeb** | Yes, fallback/auxiliary | Large, clean, streamable by Common Crawl dump, has `date` and `dump`. Not a continual benchmark by itself. |
| **RedPajama-Data-V2** | Maybe | Huge and snapshot-indexed with quality signals. Useful for raw data engineering, too heavy for the main PhD path. |
| **Dolma** | No as primary | Excellent open pretraining corpus, but not designed as period-by-period continual stream. |
| **C4** | No as primary | Based on a single April 2019 Common Crawl snapshot; wrong temporal structure. |
| **mC4 / multilingual CC / OSCAR** | No unless multilingual drift becomes a claim | Multilingual is a different axis. It adds complexity without solving cross-period probes. |
| **TemporalWiki** | Eval/auxiliary | Good for changed Wikipedia facts; too narrow as sole training corpus. |
| **CKL** | Eval/auxiliary | Directly studies invariant/updated/new knowledge, supports the thesis argument, but not enough as raw pretraining stream. |
| **StreamingQA / RealtimeQA** | Eval only | Useful temporal QA checks, but retrieval/QA oriented. |
| **TRACE** | Eval only | Continual instruction/task benchmark, not temporal document learning. |
| **EvolvingQA / TAQA / ChroKnowBench** | Eval only | Good temporal-knowledge probes; still not raw period training corpora. |
| **DailyMed/FDA labels** | Build small custom suite | Best low-effort path to versioned, structured fact updates. Rationale is weaker than guidelines, but version history is strong. |
| **NICE/CDC/WHO/NIH guidelines** | Build later | Best for rationale-rich medical changes, but scraping/version alignment and validation cost more. |

## Recommended Dataset Stack

### Phase 1: CC-News Pilot

Goal: prove INCA beats fixed-growth / LLaMA-Pro style baselines before scaling.

Use:

- Training: `vblagoje/cc_news`
- Periods: six half-year periods from 2017-2019
- Format: completion first, optional salient span denoising ablation
- Probes:
  - held-out completion loss by period
  - generated entity/date cloze probes
  - small QA probe set generated from high-confidence evidence sentences

Gate:

> INCA must beat LLaMA-Pro/fixed-growth on BWT before any TiC-LM main run.

### Paper A: TiC-LM Main Run

Goal: publish the architecture on a serious temporal continual-pretraining benchmark.

Use:

- Training: TiC-CC from TiC-LM
- Periods: monthly or coarsened monthly groups, depending on compute
- Evaluation:
  - TiC-CC held-out pages
  - TiC-CC-News / TiC-CC-Wiki if available in the scripts
  - TiC-Wiki changed/unchanged facts
  - TiC-StackExchange answer perplexity
  - TiC-CodeDocs for versioned technical documentation

This is the strongest fit because TiC-LM was built exactly for time-continual LLM pretraining and reports long-horizon continual experiments over Common Crawl.

### Fallback: FineWeb Time-Sliced Stream

If TiC-LM data reproduction is too heavy, use FineWeb:

- Slice by `dump` or `date`
- Use `token_count` to balance periods
- Use URL/domain filters to create news/wiki/code substreams
- Generate probes using the same pipeline as CC-News

FineWeb is not a benchmark by itself, but it is a very practical corpus for controlled experiments.

### Later Extension: Versioned Medical/Regulatory Update Suite

Build this after the architecture works.

Lowest-effort strong option:

- Start with DailyMed/FDA SPL records.
- Use `setid`, `spl_version`, and `published_date`.
- Diff consecutive label versions.
- Extract changed sections: indications, dosage, contraindications, warnings, adverse reactions, interactions.
- Generate old-vs-new probes from changed sentences.

Higher-value but harder option:

- Add NICE/CDC/WHO/NIH clinical guidelines.
- Capture rationale text where available.
- Human-validate ambiguous changes.

This gives the “real fact update with rationale” story that general web corpora cannot provide.

## Build-vs-Reuse Verdict

Do **not** build a full new dataset before Paper A. That would turn the PhD into a dataset project before the model claim is validated.

Instead:

1. **Reuse CC-News and TiC-LM for training streams.**
2. **Build a reusable probe-generation layer** that can operate on CC-News, FineWeb, and TiC-LM-style documents.
3. **Build a small high-precision medical/regulatory update suite** only after the CC-News pilot passes.

Estimated effort:

| Work item | Effort | Value |
|---|---:|---|
| CC-News date slicing + completion format | 2-4 days | Immediate pilot |
| Entity/date cloze probe generator | 1-2 weeks | Needed for BWT beyond perplexity |
| QA probe generator + validation UI/file | 1-2 weeks | Stronger factual metric |
| TiC-LM reproduction/adaptation | 2-4 weeks | Paper A main benchmark |
| FineWeb fallback stream | 1 week | Practical insurance |
| DailyMed versioned probe suite | 3-6 weeks | Strong domain contribution |
| NICE/CDC/WHO rationale-rich corpus | 2-4 months | Possible separate benchmark contribution |

## Final Recommendation

Use a **two-layer format**:

1. **Training layer:** raw temporal documents converted to completion, plus optional salient-span denoising.
2. **Evaluation layer:** frozen period-indexed probes: continuation loss, cloze, QA, and limited MCQ.

Use a **three-stage dataset plan**:

1. **CC-News now** for fast architecture validation.
2. **TiC-LM for Paper A** once the CC-News gate passes.
3. **DailyMed/FDA + selected guidelines later** for genuine update/rationale probes.

The important move is to stop thinking of “the dataset” as one artifact. For CAPSEL/INCA, the correct unit is a **temporal data system**:

```text
period documents -> training tasks
period documents -> probe generation
probes from period t -> re-evaluate after every future period
versioned updates -> high-precision update/rationale tests
```

That system is what lets INCA make a credible claim about learning, forgetting, and capacity growth.

## Sources

- CAPSEL local gap analysis: `docs/dataset_gap_analysis.md`
- TiC-LM paper: https://arxiv.org/abs/2504.02107
- TiC-LM code repository: https://github.com/apple/ml-tic-lm
- CC-News dataset card: https://huggingface.co/datasets/vblagoje/cc_news
- FineWeb dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- FineWeb paper: https://arxiv.org/abs/2406.17557
- RedPajama-Data-V2 dataset card: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
- RedPajama paper: https://arxiv.org/abs/2411.12372
- Dolma dataset card: https://huggingface.co/datasets/allenai/dolma
- Dolma paper: https://arxiv.org/abs/2402.00159
- T5 paper: https://arxiv.org/abs/1910.10683
- Salient Span Masking for Temporal Understanding: https://arxiv.org/abs/2303.12860
- Effect of Masking Strategies on Knowledge Retention: https://arxiv.org/abs/2306.07185
- CKL: https://arxiv.org/abs/2110.03215
- TemporalWiki: https://arxiv.org/abs/2204.14211
- StreamingQA: https://arxiv.org/abs/2205.11388
- EvolvingQA: https://arxiv.org/abs/2311.08106
- TAQA / Set the Clock: https://arxiv.org/abs/2402.16797
- ChroKnowledge / ChroKnowBench: https://arxiv.org/abs/2410.09870
- DailyMed overview: https://dailymed.nlm.nih.gov/dailymed/about-dailymed.cfm
- DailyMed SPL API: https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm
- FDA SPL resources: https://www.fda.gov/industry/fda-data-standards-advisory-board/structured-product-labeling-resources

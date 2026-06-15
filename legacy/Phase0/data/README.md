# Phase 0 — Dataset Pipeline

This directory contains all dataset download scripts, raw data archives, and
processed period-sliced JSONL files used by the CAPSEL/INCA Phase 0 baseline
evaluation harness.

---

## Directory Layout

```
Phase0/data/
├── README.md                    ← this file
├── prepare_all.py               ← orchestrator: runs all downloaders in sequence
├── download_cc_news.py
├── download_temporalwiki.py
├── download_trace.py
├── download_ckl.py
├── download_realtimeqa.py
├── download_medmcqa.py
├── download_tic_lm.py
├── _utils.py                    ← shared helpers (make_doc, probes, write_jsonl…)
│
├── raw/                         ← original dataset files (Parquet / JSONL)
│   ├── cc_news/
│   ├── temporalwiki/
│   ├── trace/
│   ├── ckl/
│   ├── realtimeqa/
│   ├── medmcqa/
│   └── tic_lm/
│
└── processed/                   ← period-sliced JSONL consumed by the harness
    └── <dataset>/
        ├── stream/<period>.jsonl    ← training documents for that period
        ├── probes/<period>.jsonl    ← evaluation probes for that period
        ├── timeline.json            ← ordered list of period ids + metadata
        └── metadata.json
```

HuggingFace Arrow cache is stored in `~/.cache/huggingface` (not in `raw/`).
`raw/` holds the actual original dataset files (Parquet splits or JSONL) so
you can re-process without re-downloading.

---

## Dataset Summary

| # | Dataset | Role in CAPSEL | HF Source | Period Scheme | Periods | Stream Docs | Probes | Raw Size |
|---|---------|---------------|-----------|---------------|:-------:|------------:|-------:|---------:|
| 1 | **CC-News** | Primary temporal benchmark | `vblagoje/cc_news` | Monthly (YYYY-MM) | 26 | 630,351 | 616,550 | 1,632 MB |
| 2 | **TemporalWiki** | Gradual-drift secondary | `seonghyeonye/TemporalWiki` + `wikimedia/wikipedia` | Snapshot | 2 | 1,000 | 80 | — ¹ |
| 3 | **TRACE** | Multi-task CL capability | 5 public HF datasets (see below) | Task name | 5 | 510,796 | 179,552 | 884 MB |
| 4 | **CKL** | Knowledge-retention proxy | `web_questions` + `trivia_qa` | Snapshot | 2 | 416,599 | 553,710 | 6,950 MB |
| 5 | **RealtimeQA** | High-difficulty weekly QA | `prajaktakini/realtime_qa` | Weekly (YYYY-Www) | 69 | 0 ² | 1,619 | 0.2 MB |
| 6 | **MedMCQA** | Medical domain benchmark | `openlifescienceai/medmcqa` | Subject group | 4 | 182,822 | 43,395 | 78 MB |
| 7 | **TiC-LM** | Temporal LM drift | `allenai/c4` (realnewslike) | Daily (YYYY-MM-DD) | 9 | 27,000 | 26,810 | 71 MB |
| | **Total** | | | | **117** | **1,768,568** | **1,421,716** | **~9.6 GB** |

¹ TemporalWiki raw JSONL not saved (downloader ran before raw-save refactor); processed data is intact.
² RealtimeQA carries no article text — it is a probes-only dataset by design.

---

## Per-Dataset Details

---

### 1. CC-News

**Purpose:** Primary temporal benchmark. Tests month-over-month factual
knowledge drift as the news cycle evolves.

**Source:** [`vblagoje/cc_news`](https://huggingface.co/datasets/vblagoje/cc_news)
— English news articles, January 2017 – August 2019.

**Download script:** `download_cc_news.py`

**Processing steps:**
1. Streamed without caching.
2. Documents bucketed by publication month from the `date` field (`YYYY-MM`).
3. Deduplication by first-50-character prefix within each month.
4. Raw articles saved as `raw/cc_news/raw.jsonl` (one JSON object per line: date, title, text, url).
5. Stream docs: full article text, minimum 200 characters.
6. Probes: auto-cloze — a sentence from the second half of each article is masked; four distractor sentences from the same period form the choices.

**Period range:** 2017-01 → 2019-08 (26 months; some months in 2019 are absent from the HF dataset)

| Metric | Value |
|--------|-------|
| Stream periods | 26 |
| Total stream docs | 630,351 |
| Total probes | 616,550 |
| Avg docs / period | ~24,244 |
| Raw files | 1 JSONL |
| Raw size | 1,632 MB |
| Probe formats | `mc4` (cloze), `completion` |

---

### 2. TemporalWiki

**Purpose:** Gradual-drift secondary benchmark (Dhingra et al., EMNLP 2022).
Tests factual update tracking across a multi-year Wikipedia gap.

**Sources:**
- `period_2022` → [`seonghyeonye/TemporalWiki`](https://huggingface.co/datasets/seonghyeonye/TemporalWiki) — 2022-era Wikipedia
- `period_2023` → [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) config `20231101.en` — November 2023 snapshot

The original gated `temporalwiki/twiki-probes` requires authentication;
these are equivalent public mirrors.

**Download script:** `download_temporalwiki.py`

**Processing steps:**
1. Both sources streamed with a `max_docs_per_period` cap (default 500).
2. Deduplication by first-50-character prefix.
3. Raw rows saved as `raw/temporalwiki/<period>.jsonl` (title, text, period).
4. Probes split 50 / 50: second half → UpdatedLAMA-style cloze (`mc4`); first half → completion probes.

| Metric | Value |
|--------|-------|
| Stream periods | 2 |
| Total stream docs | 1,000 |
| Total probes | 80 |
| Avg docs / period | 500 |
| Raw files | missing ¹ |
| Raw size | — |
| Probe formats | `mc4` (cloze), `completion` |

¹ Re-run `python download_temporalwiki.py --force` to regenerate raw JSONL.

---

### 3. TRACE

**Purpose:** Multi-task continual learning capability benchmark (Wang et al.,
arXiv 2310.06762). Tests instruction-following retention across five task
domains presented sequentially.

**Sources (one HF dataset per period):**

| Period | HF Dataset | Config | Description |
|--------|-----------|--------|-------------|
| `general` | [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | — | Open-domain instruction following |
| `math` | [`gsm8k`](https://huggingface.co/datasets/gsm8k) | `main` | Grade-school math word problems |
| `summarization` | [`cnn_dailymail`](https://huggingface.co/datasets/cnn_dailymail) | `3.0.0` | News article summarisation |
| `coding` | [`iamtarun/python_code_instructions_18k_alpaca`](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) | — | Python code generation |
| `medical` | [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa) | — | Medical QA |

**Download script:** `download_trace.py`

**Processing steps:**
1. Each task loaded separately; all available splits saved as `raw/trace/<task>_<split>.parquet`.
2. Stream docs formatted as Alpaca-style instruction–response pairs:
   `### Instruction:\n…\n\n### Response:\n…`
3. Probes: `instruction` probes (full prompt → reference response) + `mc4` (correct response vs. 3 distractors from the same task) + `completion` probes from stream docs.

| Metric | Value |
|--------|-------|
| Stream periods | 5 |
| Total stream docs | 510,796 |
| Total probes | 179,552 |
| Avg docs / period | ~102,159 |
| Raw files | 10 Parquet splits |
| Raw size | 884 MB |
| Probe formats | `instruction`, `mc4`, `completion` |

**Per-period breakdown:**

| Period | Stream Docs | Probes |
|--------|------------:|-------:|
| general | ~15,000 | varies |
| math | ~7,473 | varies |
| summarization | ~287,113 | varies |
| coding | ~18,000 | varies |
| medical | ~182,822 | varies |

---

### 4. CKL

**Purpose:** Knowledge-retention proxy for CKL (Jang et al., ICLR 2022).
Tests whether models retain stable facts (InvariantLAMA) and acquire new ones
(NewLAMA) across two QA knowledge snapshots.

**Sources:**
- `period_A` → [`web_questions`](https://huggingface.co/datasets/web_questions) (Berant et al., 2013) — older entity facts via Freebase
- `period_B` → [`trivia_qa`](https://huggingface.co/datasets/trivia_qa) rc config (Joshi et al., 2017) — broader fact coverage with evidence passages

The original `joeyoonjeong/CKL` requires authentication; this two-period
layout is an equivalent fully public proxy.

**Download script:** `download_ckl.py`

**Processing steps:**
1. `web_questions`: loaded non-streaming; saved as `raw/ckl/web_questions_train.parquet`. Stream docs: `Q: …\nA: …`.
2. `trivia_qa`: streamed (first 2,000+ rows); saved as `raw/ckl/trivia_qa_train.parquet` via pandas. Stream docs from `search_results.search_context` passages where available, fallback to `Q: …\nA: …`.
3. `mc4` probes: correct answer as key `A`, plus 3 random distractors drawn from all answers in the same period.
4. Completion probes fill remaining probe budget from stream docs.

| Metric | Value |
|--------|-------|
| Stream periods | 2 |
| Total stream docs | 416,599 |
| Total probes | 553,710 |
| Avg docs / period | ~208,300 |
| Raw files | 2 Parquet files |
| Raw size | 6,950 MB |
| Probe formats | `mc4`, `completion` |

---

### 5. RealtimeQA

**Purpose:** High-difficulty secondary benchmark (Kasai et al., 2022). Tests
real-time factual knowledge — questions whose answers require knowing facts
from a specific week's news cycle. Used as evaluation probes only.

**Source:** [`prajaktakini/realtime_qa`](https://huggingface.co/datasets/prajaktakini/realtime_qa)
— weekly batches of multiple-choice questions from CNN and USA Today,
covering ISO weeks 2023-W01 through 2025-W49.

**Download script:** `download_realtimeqa.py`

**Processing steps:**
1. Loaded non-streaming; raw split saved as `raw/realtimeqa/train.parquet` (200 KB).
2. Questions grouped by ISO week (`YYYY-Www`) from the question date field.
3. Stream docs: built from `evidence` / `context` field (minimum 80 characters). **Field is empty for all rows** — stream files are intentionally 0 bytes.
4. Each question becomes a native `mc4` probe with 4 answer choices.

| Metric | Value |
|--------|-------|
| Stream periods | 69 |
| Total stream docs | 0 (probes only) |
| Total probes | 1,619 |
| Avg probes / week | ~23 |
| Coverage | 2023-W01 → 2025-W49 |
| Raw files | 1 Parquet |
| Raw size | 0.2 MB |
| Probe formats | `mc4` |

---

### 6. MedMCQA

**Purpose:** Medical domain benchmark. Replaces the gated TiC-LM domain test.
Provides a multi-period CL stream by grouping medical sub-domains as sequential
subject-group periods.

**Source:** [`openlifescienceai/medmcqa`](https://huggingface.co/datasets/openlifescienceai/medmcqa)
— ~194k Indian medical entrance exam questions (AIIMS + USMLE) with 4-way
multiple choice, explanations, and subject / topic metadata (Pal et al., AAAI 2022).

**Download script:** `download_medmcqa.py`

**Processing steps:**
1. All three splits loaded non-streaming; saved as Parquet to `raw/medmcqa/` (train 182k, validation 4.2k, test 6.1k labels unavailable).
2. Questions bucketed into 4 subject-group periods by `subject_name`:

| Period | Subjects Included |
|--------|------------------|
| `basic_sciences` | Anatomy, Physiology, Biochemistry, Microbiology |
| `clinical_basics` | Pharmacology, Pathology |
| `clinical_applied` | Medicine, Surgery, Gynaecology & Obstetrics, Paediatrics |
| `specialties` | Radiology, Anaesthesia, Forensic Medicine, Ophthalmology, ENT, Psychiatry, Dermatology, Orthopaedics, Social & Preventive Medicine |

3. Stream docs: `Question: …\n  A. …\n  B. …\nAnswer: X\nExplanation: …`
4. `mc4` probes built from validation split (fallback to train if empty). Correct answer rotated to key `A` for consistency across all probes.
5. Completion probes fill remaining budget from stream docs.

| Metric | Value |
|--------|-------|
| Stream periods | 4 |
| Total stream docs | 182,822 |
| Total probes | 43,395 |
| Avg docs / period | ~45,706 |
| Raw files | 3 Parquet splits |
| Raw size | 78 MB |
| Probe formats | `mc4`, `completion` |

---

### 7. TiC-LM

**Purpose:** Temporal language model drift benchmark (Li et al., ACL 2025).
Tests whether model perplexity and probe accuracy degrade as news text shifts
day-by-day across a crawl window.

**Source:** [`allenai/c4`](https://huggingface.co/datasets/allenai/c4) config `realnewslike`
— ~36 GB news-filtered subset of C4 (Common Crawl, April 2019 crawl). The
original `apple/TiC-LM` requires authentication. C4 realnewslike uses the same
CC lineage with a `timestamp` field that reflects per-article publication dates
at day-level precision.

**Download script:** `download_tic_lm.py`

**Processing steps:**
1. Streamed with a **scan-budget strategy**: scans `min(n_periods × n_docs × 20, 2,000,000)` docs, buckets every document by full publication date (`timestamp[:10]` → `YYYY-MM-DD`), then selects the top N days by doc count.
2. Raw docs saved as `raw/tic_lm/<YYYY-MM-DD>.jsonl` (one file per day period).
3. `mc4` cloze probes from second half of docs; completion probes from first half.

**Current download:** 9 daily periods (2019-04-18 → 2019-04-26), 3,000 docs/day.

| Metric | Value |
|--------|-------|
| Stream periods | 9 |
| Total stream docs | 27,000 |
| Total probes | 26,810 |
| Avg docs / period | 3,000 |
| Date range | 2019-04-18 → 2019-04-26 |
| Raw files | 9 JSONL files |
| Raw size | 71 MB |
| Probe formats | `mc4` (cloze), `completion` |

To extend the date range, increase `--max_periods` (up to ~30 distinct days are available in the April 2019 crawl window):

```bash
python Phase0/data/prepare_all.py --only tic_lm --max_periods 20 --max_docs 3000 --force
```

---

## Processed Data Format

All stream and probe files are newline-delimited JSON (JSONL — one JSON object per line).

### Stream document

```json
{
  "text":     "Full article / QA / instruction-response text",
  "doc_id":   "cc_news_2017-01_000042",
  "period":   "2017-01",
  "source":   "cc_news",
  "char_len": 1830
}
```

### Probe — mc4 (4-way multiple choice)

```json
{
  "format":     "mc4",
  "question":   "Which protein is responsible for…",
  "evidence":   "Optional supporting sentence (≤300 chars)",
  "choices":    {"A": "correct answer", "B": "distractor", "C": "distractor", "D": "distractor"},
  "answer_key": "A",
  "period":     "basic_sciences",
  "source":     "medmcqa"
}
```

The correct answer is always rotated to key `A` across all datasets for consistent evaluation.

### Probe — completion

```json
{
  "format":  "completion",
  "prompt":  "The first portion of a sentence or paragraph…",
  "target":  "…the masked continuation",
  "period":  "2017-01",
  "source":  "cc_news"
}
```

### Probe — instruction (TRACE only)

```json
{
  "format":    "instruction",
  "prompt":    "### Instruction:\nWrite a Python function that…",
  "reference": "def fibonacci(n):\n    …",
  "period":    "coding",
  "source":    "trace"
}
```

### timeline.json

```json
["2017-01", "2017-02", "2017-03", "…", "2019-08"]
```

or the dict form (where `counts` is a list of `[period_id, n_docs, n_probes]`):

```json
{
  "source":       "vblagoje/cc_news",
  "period_scheme":"month",
  "probe_formats":["mc4", "completion"],
  "counts":       [["2017-01", 24311, 23890], ["2017-02", 21500, 20900], "…"]
}
```

---

## How to Reproduce

```bash
# Install dependencies
pip install datasets pandas pyarrow

# Full download — skips datasets already present in raw/
python Phase0/data/prepare_all.py

# Quick sanity run — 4 periods × 500 docs, skips heavy datasets (cc_news, trace, tic_lm)
python Phase0/data/prepare_all.py --quick

# Only specific datasets
python Phase0/data/prepare_all.py --only medmcqa realtimeqa

# Skip specific datasets
python Phase0/data/prepare_all.py --skip ckl

# Force re-download everything
python Phase0/data/prepare_all.py --force

# Custom caps
python Phase0/data/prepare_all.py --max_periods 12 --max_docs 2000

# TiC-LM with more data (up to ~30 days available)
python Phase0/data/prepare_all.py --only tic_lm --max_periods 20 --max_docs 3000 --force
```

`prepare_all.py` skips any dataset whose `raw/` directory already contains at
least one non-empty `.parquet` or `.jsonl` file. Use `--force` to override.

**Heavy datasets** (skipped by `--quick`): `cc_news`, `trace`, `tic_lm`.

---

## Known Issues

| Dataset | Issue | Impact | Fix |
|---------|-------|--------|-----|
| TemporalWiki | `raw/temporalwiki/` contains only lock files, no JSONL | Processed data intact; raw backup missing | `python download_temporalwiki.py --force` |
| RealtimeQA | All stream files are 0 bytes | Dataset has no article text; probes-only use is correct and expected | None needed |
| TiC-LM | All 9 periods fall within April 2019 | Temporal variation is daily (within one month) not monthly/yearly — inherent to C4's single-crawl design | Increase `--max_periods` for more days; accept the daily granularity |

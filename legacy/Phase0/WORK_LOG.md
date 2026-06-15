# Phase 0 — Work Log

This log tracks the state of the Phase 0 baselines package. Update it
whenever a baseline lands, a script is added, or a refactor changes the
public API. Newest entries at the top.

## Scope

Phase 0 implements the eight baselines (B1–B8) plus a self-contained eval
harness against which every Paper A variant is measured. All metrics follow
the CAPSEL Research Memorandum (FINAL, Part XIII). Phase 0 has **no
dependency on any prior research codebase** — it is fully self-contained.

## Current state

### Baselines

| ID | Name                | Status         | File                                    |
|----|---------------------|----------------|-----------------------------------------|
| B1 | Naive fine-tune     | code complete  | `Phase0/baselines/b1_finetune.py`       |
| B2 | Replay-only         | code complete  | `Phase0/baselines/b2_replay.py`         |
| B3 | EWC                 | code complete  | `Phase0/baselines/b3_ewc.py`            |
| B4 | L2P                 | code complete  | `Phase0/baselines/b4_l2p.py`            |
| B5 | LoRA-MoE            | code complete  | `Phase0/baselines/b5_lora_moe.py`       |
| B6 | LLaMA-Pro vertical  | code complete  | `Phase0/baselines/b6_llama_pro.py`      |
| B7 | Progressive NN      | code complete  | `Phase0/baselines/b7_pnn.py`            |
| B8 | BlockStack variant  | code complete  | `Phase0/baselines/b8_block_stack.py`    |

### Common modules

| Module                          | Purpose                                                    |
|---------------------------------|------------------------------------------------------------|
| `common/config.py`              | `Phase0Config` dataclass + YAML/JSON loader                |
| `common/datasets.py`            | JSONL period reader (no external deps)                     |
| `common/harness.py`             | `TextDataset`, `Period`, `RunLogger`, `evaluate_past_periods` |
| `common/metrics.py`             | CAPSEL XIII metrics: PPL, probe acc, combined, BWT, ACC, FWT, RIR, CKA, Fisher convergence |
| `common/runner.py`              | `BaselineRunner` — drives the period loop, writes summary.json |
| `common/variant.py`             | `BlockStack` — Paper A variant (frozen stack + current block + selector) |

### Datasets

`Phase0/data/` contains downloaders that materialise the six CAPSEL
benchmarks into `Phase0/data/processed/<dataset>/{stream,probes}/<period>.jsonl`
plus a `timeline.json`. See "Dataset downloaders" below.

## Log entries

### 2026-04-28 — Refactor: drop external dependency, add dataset downloaders

- All Phase 0 code is now fully self-contained. No imports from any prior
  research codebase. The metrics module implements every quantity from
  CAPSEL XIII directly: PPL, probe MC accuracy, combined score, BWT/ACC/FWT
  via `StreamAccuracyMatrix`, RIR (XIII.4), CKA (XIII.2), diagonal-Fisher
  convergence rate (XIII.3).
- Replaced the previous "audited v1" baseline with `B8 BlockStack` — a
  self-contained variant living at `common/variant.py`. It composes a
  frozen stack of pretrained GPT-2 blocks, a single trainable current
  block (zero-init residual to start at identity), and a pluggable
  selector (`weighted_sum`, `gated`, or `cross_attn`).
- Added `max_docs_per_period` to `Phase0Config` (the runner references it).
- Renamed harness `_TextDataset` → `TextDataset`; updated B2 to match.
- Wrote download scripts for the six CAPSEL benchmarks under `Phase0/data/`
  plus a `prepare_all.py` orchestrator. Each downloader writes the
  standard period-sliced JSONL layout the loader expects.

## Dataset downloaders

| Dataset      | Script                                | HF source / origin                          | Stream period | Probes |
|--------------|---------------------------------------|---------------------------------------------|---------------|--------|
| CC-News      | `data/download_cc_news.py`            | `cc_news` (HF `vblagoje/cc_news`)           | year-month    | auto-cloze |
| TemporalWiki | `data/download_temporalwiki.py`       | Dhingra et al, EMNLP 2022; HF mirrors       | year-half     | UpdatedLAMA-style cloze |
| TiC-LM       | `data/download_tic_lm.py`             | `apple/TiC-LM` monthly subsets              | year-month    | TiC-LM eval slice |
| TRACE        | `data/download_trace.py`              | TRACE 8-task release                        | task name     | per-task   |
| CKL          | `data/download_ckl.py`                | Jang et al ICLR 2022 (HF: `joeyoon/ckl`)    | period        | InvariantLAMA / UpdatedLAMA / NewLAMA |
| RealtimeQA   | `data/download_realtimeqa.py`         | `realtimeqa/realtimeqa_public`              | year-week     | per-question MC |

`prepare_all.py` runs every downloader. Each script supports
`--max_periods` and `--max_docs_per_period` for fast iteration.

## Pending

- Run the eight baselines on a training machine; populate `docs/baselines_report.md`.
- Wire `python -m Phase0.common.report` for automatic regeneration of the
  comparison table from `Phase0/results/*/summary.json`.

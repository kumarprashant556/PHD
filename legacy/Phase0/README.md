# Phase 0 — Baselines and Audit

This folder implements Phase 0 of the CAPSEL programme: the eight baselines
that every Paper A variant will be measured against. No variant is trusted
until the baselines are in place and a shared eval harness exists.

Phase 0 is **self-contained** — no dependency on any external research
codebase. Metrics follow the CAPSEL definitions (Part XIII of the memo);
datasets are produced by the `Phase0/data/download_*.py` scripts into
`Phase0/data/processed/<dataset>/`.

## Quick start

1. Prepare a dataset (run on a machine with network access):
   ```bash
   python Phase0/data/download_cc_news.py
   # or: download_temporalwiki.py, download_tic_lm.py, download_trace.py,
   #     download_ckl.py, download_realtimeqa.py
   # or: python Phase0/data/prepare_all.py  (runs every downloader)
   ```
2. Run a baseline:
   ```bash
   python -m Phase0.baselines.b1_finetune --config Phase0/configs/base.yaml
   ```
3. Results land in `Phase0/results/<baseline_id>/{metrics.json,summary.json,training.log}`.
4. `python -m Phase0.common.report` regenerates `docs/baselines_report.md`.

## Design

Each baseline is a thin orchestrator on top of `Phase0/common/`:

- `common/datasets.py` — JSONL period loader (no external dependencies).
- `common/harness.py` — shared tokenisation + period dataclass + logger.
- `common/metrics.py` — CAPSEL metrics: perplexity, probe accuracy,
  combined score, BWT/ACC/FWT (Lopez-Paz & Ranzato), RIR (XIII.4), CKA
  (XIII.2), diagonal-Fisher convergence rate (XIII.3).
- `common/runner.py` — the shared period-loop skeleton. Every baseline
  implements `build_model`, `scoring_model`, `on_period_start`,
  `train_period`, `on_period_end`.
- `common/variant.py` — the Paper A variant: a growable block-stack
  architecture with a pluggable selector, used by B8.
- `common/config.py` — shared config dataclass loaded from YAML.

What makes a baseline "fair":
1. Same backbone (`distilgpt2` by default; the memo does not forbid `gpt2`).
2. Same dataset partition and same probe set.
3. Same eval harness (BWT from a `StreamAccuracyMatrix`, same PPL helper).
4. Same optimiser family (AdamW) and same tokenisation / max_seq_len.
5. Parameter-count budget is reported per baseline but not matched —
   matching is impossible across architectures and hides the real question
   (does the architecture earn its parameters?).

## Baselines

| ID | Name                | What it isolates                                             |
|----|---------------------|--------------------------------------------------------------|
| B1 | Naive fine-tune     | Catastrophic-forgetting floor. No CL mechanism.              |
| B2 | Replay-only         | Replay contribution independent of architecture.             |
| B3 | EWC                 | Regularisation-based CL without replay.                      |
| B4 | L2P                 | Prompt-based CL with frozen backbone.                        |
| B5 | LoRA-MoE            | Modular CL with routing but no shared representation growth. |
| B6 | LLaMA-Pro           | Fixed-schedule vertical growth (strong growth-but-no-saturation baseline). |
| B7 | PNN                 | Column-per-period with lateral connections.                  |
| B8 | BlockStack variant  | Growable block-stack with pluggable selector (Paper A reference). |

## Outputs every baseline must produce

1. `results/<id>/metrics.json` — per-period record: `pre_ppl`, `post_ppl`,
   `pre_probe_acc`, `post_probe_acc`, `pre_combined`, `post_combined`,
   `rir`, `bwt_row`, `final_loss`, `params_total`, `params_trainable`.
2. `results/<id>/summary.json` — final CAPSEL metrics: `ACC`, `BWT`, `FWT`,
   `final_combined_last_period`, params.
3. `results/<id>/training.log` — line-per-event human-readable log.
4. `results/<id>/config.snapshot.json` — exact config this run used.
5. `results/<id>/probes_period*.csv` — per-probe predictions for error
   analysis.

## Status

See `WORK_LOG.md` for the running status. Short answer: scaffold is in, the
shared harness + CAPSEL metrics + all eight baselines compile clean; dataset
downloaders are next, then runs on user's training machine.

# CAPSEL / INCA — PhD Task List (sequenced)

Single ordered checklist to take the project from today's smoke-tested code to a defended thesis.
Tasks are in **execution order**. Each phase ends with a **GATE** that must pass before the next phase starts.
Paths use the current function-based repo layout (`data/ models/ training/ evaluation/ scripts/ configs/ tests/`).

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done
Source of truth: `docs/CAPSEL_INCA_Master_Reference.pdf`, `docs/CAPSEL_Implementation_Guide.pdf`, `docs/CAPSEL_PhD_Roadmap.pdf`.

**Three standing rules:** (1) never run a TiC-LM main experiment before the CC-News pilot confirms the method; (2) freeze data splits before Phase 1 full run — any drift invalidates all comparisons; (3) if INCA underperforms a metric, report and explain it.

---

## Phase 0 — Baselines  (foundation; sets the comparison floor for Paper A)

- [ ] **P0.1** Verify the 8 baseline trainers run end-to-end on CC-News (`models/baselines/{finetune,replay,ewc,l2p,lora_moe,llama_pro,pnn,block_stack}.py`)
- [ ] **P0.2** Freeze CC-News splits and save manifest → `results/phase0/data_split_manifest.json` (BLOCKS everything downstream)
- [ ] **P0.3** Confirm EWC Fisher uses Soen-2025 correction (labelled targets, batch-size-normalised, unit-tested vs closed form)
- [ ] **P0.4** Run B1–B8 on CC-News, 6 periods × 3 seeds → `results/phase0/`
- [ ] **P0.5** Compute BWT / FWT / Acc (+ WER/PPL on held-out) via `evaluation/metrics.py`; export CSV
- [ ] **P0.6** Fill `results/phase0/baselines_report.md` with the Table-1 numbers
- [ ] **P0.7** 1-page internal note: which baseline is hardest to beat and why (expect B6 LLaMA-Pro)
- [ ] **P0.8** Git tag `phase0-results`
- [ ] **GATE 0:** all 8 baselines reported on frozen splits; LLaMA-Pro (B6) BWT/FWT is the number to beat

### Data system (cross-cutting — see `docs/dataset_strategy_FINAL.md`)
- [ ] **D.1** `data/probe_gen.py` — reusable NER-anchored probe generator (entity/date cloze + QA + optional MCQ), `stability` ∈ {stable,updated,deprecated}, frozen per period, `eval_after_periods` list — **highest-leverage data task; needed for BWT beyond perplexity**
- [ ] **D.2** Adopt TiC-LM regret-matrix BWT/FWT definitions in `evaluation/metrics.py` (R[i,j] = model-after-i on probes-of-j); BWT=lower triangle, FWT=upper; computed over probe accuracy, PPL only as sanity check
- [ ] **D.3** Period-prefix all training inputs (`period: <id>\ncomplete: ...`)
- [ ] **D.4** `data/salient_span.py` — entity/date salient-span denoising objective (for E-FORMAT)
- [ ] **D.5** (Paper A) TiC-CC loader (`data/tic_lm.py`) + TiC-Wiki/StackExchange/CodeDocs evals; FineWeb fallback loader (`data/fineweb.py`, CC-dump sliced)
- [ ] **D.6** (after GATE 1) DailyMed/FDA SPL versioned-label update suite — diff consecutive `spl_version`s → old-vs-new probes with rationale. *Optional standalone contribution; do NOT gate Paper A on it.*

## Phase 1 — INCA Core  (the core contribution; smoke run already passing)

- [ ] **P1.1** Full CC-News INCA run: 6 periods × 3 seeds, collect BWT/FWT/Acc → `results/phase1/cc_news/` *(roadmap T1.7)*
- [ ] **P1.2** Checkpoint save/load tested across grow events (resume mid-training) *(T1.10)*
- [ ] **P1.3** SPRT-based saturation detector replacing fixed patience in `models/inca/plateau.py` (α=0.05, β=0.10) *(T1.8)*
- [ ] **P1.4** Fisher-based structured pruning before freeze → `models/inca/` (heads + MLP neurons) *(T1.9)*
- [ ] **P1.5** **E-CCNEWS pilot:** INCA-v3 vs B6 LLaMA-Pro head-to-head → `results/phase1/comparison.md` *(T1.12)* — **this is the go/no-go for TiC-LM**
- [ ] **P1.6** Git tag `phase1-results`
- [ ] **GATE 1:** INCA beats LLaMA-Pro on CC-News BWT (pilot); checkpointing robust. If it loses, invoke risk plan (reframe to "saturation-driven growth converges faster") before scaling.

## Phase 2 — Growth Architecture  (makes INCA adaptive; fills Paper A §5 ablations)

- [ ] **P2.1** `models/inca/lateral.py` — rank-r lateral adapters, α_k=0 init (function-preserving); unit-test identity at init *(T2.1)*
- [ ] **P2.2** `models/inca/width.py` — Net2Net width expansion (G-HORIZ), function-preserving split *(T2.2)*
- [ ] **P2.3** `models/inca/growth_chooser.py` — G-VERT / G-HORIZ / G-LAT decision from saturation profile *(T2.3)*
- [ ] **P2.4** Wire chooser into `training/inca_trainer.py` (`growth_mode: auto|vert|horiz|lat`) *(T2.4)*
- [ ] **P2.5** **E-SCOPE:** lateral rank r ∈ {4,8,16} BWT/param tradeoff → `results/phase2/e_scope/`
- [ ] **P2.6** **E-GROW:** G-VERT/HORIZ/LAT/auto, 3 seeds → `results/phase2/e_grow/`
- [ ] **P2.7** **E-PRUNE:** Fisher sweep p ∈ {0,10,15,20}% (Fisher vs magnitude) → `results/phase2/e_prune/`
- [ ] **P2.7b** **E-FORMAT:** F-COMP (completion) vs F-DENOISE (salient-span) vs F-MIX (70/30) → knowledge acquisition vs saturation-signal smoothness → `results/phase2/e_format/`
- [ ] **P2.8** **Decision:** pick growth primitive for Paper A (auto-chooser only if it beats best fixed by >2% BWT, else G-VERT for a simpler story)
- [ ] **P2.9** Git tag `phase2-ablations`
- [ ] **GATE 2:** growth primitive selected; E-GROW/E-SCOPE/E-PRUNE results in hand

## Phase 3 — CLS Experiments  (mechanistic analysis; fills Paper A §6)

- [ ] **P3.1** `evaluation/probes.py` — linear probes on each frozen block's mean-pooled reps
- [ ] **P3.2** **E-CLS1:** frozen block 0 retains ≥90% period-0 probe accuracy at end → `results/phase3/e_cls1/`
- [ ] **P3.3** **E-CLS2:** CKA(block0, blockN) divergence increases with chain depth → `results/phase3/e_cls2/`
- [ ] **P3.4** **E-CLS3:** hard-replay (p_hard=0.70) vs uniform → BWT reduction → `results/phase3/e_cls3/`
- [ ] **P3.5** **E-CLS4:** grokking guard prevents premature freeze (MI progress floor) → `results/phase3/e_cls4/`
- [ ] **P3.6** **E-CLS5:** SPRT sensitivity — vary α/β, measure false-grow / missed-saturation → `results/phase3/e_cls5/`
- [ ] **P3.7** Write ~2-page analysis narrative → `notes/cls_analysis_narrative.md` (neocortex/hippocampus framing)
- [ ] **P3.8** Git tag `phase3-cls`
- [ ] **GATE 3:** CLS story validated and written; ready to assemble Paper A §6

## Phase 4 — Routing + Paper A  (highest-stakes; ~40–60 GPU-h)

- [ ] **P4.1** `models/inca/uclbr.py` — Read-ME pre-gate + DeepSeek aux-loss-free load balance + uncertainty-calibrated confidence
- [ ] **P4.2** Wire `--selector uclbr`; validate on CC-News first
- [ ] **P4.3** **E-ROUTE:** S-WS / S-FULL / S-QKV / UCLBR on CC-News → pick best (S-QKV default unless UCLBR wins ≥1%) → `results/phase4/e_route/`
- [ ] **P4.4** **E-SAT:** any-1/any-2/any-3/any-4 consensus → false-grow rate → `results/phase4/e_sat/`
- [ ] **P4.5** **Request ≥200 GPU-h** allocation before the main run
- [ ] **P4.6** **E-TIC-A (main run):** INCA + B1–B8, 3 seeds, flan-t5-base, 12-mo TiC-LM (or compressed 6-mo) → `results/phase4/tic_lm_track_a/`
- [ ] **P4.7** **E-SCALE:** flan-t5-large single seed to confirm scaling → `results/phase4/e_scale/`
- [ ] **P4.8** Draft Paper A §1–§7 → `paper_a/` (Method, TiC-LM Table 1, ablations, CLS analysis)
- [ ] **P4.9** Supervisor review ×2 + revisions → `paper_a/revision_v1/`
- [ ] **P4.10** Submit Paper A (NeurIPS → ICLR → ICML fallback) + git tag `paper-a-submission`
- [ ] **GATE 4 / Paper A done:** min bar = INCA > LLaMA-Pro BWT by ≥1% absolute, 3 seeds, TiC-LM Track A

## Phase 5 — Paper B — Bounded INCA via Block Merging

- [ ] **P5.1** `models/inca/block_merge.py` — TIES merge (trim/elect-sign/disjoint) + distillation refinement + router recalibration
- [ ] **P5.2** Derive & prove post-merge router error bound → `paper_b/proof/`
- [ ] **P5.3** **E-MERGE:** merge cadence K ∈ {2,3,5} → accuracy drop vs chain length → `results/phase5/e_merge/`
- [ ] **P5.4** **E-TAU:** τ_merge ∈ {0.85,0.90,0.95} → compression vs BWT → `results/phase5/e_tau/`
- [ ] **P5.5** **E-TIC-B (main run):** bounded-INCA vs unbounded vs SEEKR vs LoRA-MoE, Pythia-160M, 24 periods, 3 seeds → `results/phase5/tic_lm_track_b/`
- [ ] **P5.6** Draft Paper B §1–§6 → `paper_b/`; supervisor review ×2 → `paper_b/revision_v1/`
- [ ] **P5.7** Submit Paper B (NeurIPS Efficient ML / ICML) + git tag `paper-b-submission`
- [ ] **GATE 5 / Paper B done:** bounded-INCA matches unbounded within 1% BWT at K=3, chain ≤4 blocks

## Thesis Write-up  (overlaps Phase 5)

- [ ] **TH.1** Ch 1 Introduction (motivation, research questions, contributions)
- [ ] **TH.2** Ch 2 Literature Review (CL taxonomy, LLMs for CL, grow-then-prune)
- [ ] **TH.3** Ch 3 INCA Architecture & Method (expand Paper A §3–§4)
- [ ] **TH.4** Ch 4 Experimental Evaluation (Paper A results + ablations + CLS)
- [ ] **TH.5** Ch 5 Bounded INCA (Paper B: merge algorithm, error bound, Track B)
- [ ] **TH.6** Ch 6 Discussion & Future Work (UCLBR, task-free NN-CUSUM, arch (c)) + Appendices A (proofs) / B (hyperparams) / C (compute)
- [ ] **TH.7** Full draft → supervisor; 2 rounds of revisions
- [ ] **TH.8** Submit to examiners
- [ ] **TH.9** Viva prep (30-min talk + anticipated Q&A) → defend → corrections → **PhD awarded**

---

### Immediate next actions (start here)
1. **P0.2** freeze + manifest the CC-News splits.
2. **P0.4–P0.6** run B1–B8 (3 seeds) and fill `baselines_report.md`.
3. In parallel, **P1.1** the full 6-period INCA run, then **P1.5** the LLaMA-Pro pilot — that pilot is the project's first real go/no-go.

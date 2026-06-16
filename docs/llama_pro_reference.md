# LLaMA-Pro Reference — Extracted for Paper B Alignment

> **Purpose**: Structured extraction of LLaMA-Pro's datasets, methodology, training details,
> and results to serve as (a) a factual reference for writing Paper B and (b) a blueprint
> for structuring the INCA results section in the same format.
>
> **Source**: Wu et al. (2024) "LLaMA Pro: Progressive LLaMA with Block Expansion"
> arXiv:2401.02415v2, 30 May 2024
>
> **Related**: `paper_b_implementation_brief.md`, `paper_main_outline.md`

---

## 1. Core Idea (One Paragraph)

LLaMA-Pro proposes **block expansion post-pretraining**: take a pretrained LLM (LLaMA2-7B),
interleave additional Transformer blocks at fixed positions, zero-initialise their output
projections (W_O and W3) so the model is identity-preserving at init, then **freeze the
original blocks** and train only the new blocks on domain-specific data.  Result: a 8.3B
model (7B → 8.3B) that retains general ability while gaining code/math capability.

The key limitation (INCA's opportunity): the expansion schedule is **fixed** — exactly 1 new
block per domain boundary regardless of whether the existing blocks are actually saturated.

---

## 2. Datasets

### 2.1 Pre-training corpus

| Source | Tokens | Weight | Domain |
|---|---|---|---|
| The Stack-Dedup (Python subset) | 22B | 1.50 | Code |
| Proof-Pile-2 total | 55B | 1.00 | Math |
| — AlgebraicStack | 11B | — | Math |
| — OpenWebMath | 15B | — | Math |
| — ArXiv | 29B | — | Math |
| **Total** | **~80B** | — | Code + Math |

### 2.2 Instruction fine-tuning (SFT) corpus

| Dataset | # Samples | Avg rounds | Avg prompt len | Avg completion len |
|---|---|---|---|---|
| ShareGPT | 63,817 | 2.9 | 293 | 1,157 |
| WizardLM_evol_instruct_V2 | 143,000 | 1.0 | 603 | 1,705 |
| SlimOrca | 517,982 | 1.0 | 574 | 599 |
| MetaMath | 395,000 | 1.0 | 209 | 498 |
| Evol-CodeAlpaca | 111,272 | 1.0 | 653 | 1,552 |
| **Total (approx)** | **~1M** | — | — | — |

### 2.3 Law domain (ablation only)

- **FreeLaw subset of Pile** (Gao et al. 2020): 51.2 GiB raw, 16.7B tokens, 3.6M documents

### 2.4 Perplexity evaluation sets

| Domain | Dataset |
|---|---|
| General | LAMBADA (OpenAI + Standard) |
| Code | `bigcode/the-stack-smol-xs` (Python split) |

---

## 3. Methodology

### 3.1 Block expansion procedure

1. Partition original L=32 blocks into **N=8 groups**, each containing M=4 blocks.
2. For each group, create **P=1 identity copy** of the top block.
3. Interleave original + identity blocks → 32 + 8 = **40 blocks** (8.3B parameters).
4. **Zero-init** output projections (W_O in MHSA, W3 in FFN) in new blocks → identity at init.
5. **Freeze** all original 32 blocks; train only the 8 new blocks.
6. Train on code + math corpus (80B tokens).

### 3.2 Why interleaved (not top or bottom)

- Bottom stacking: disrupts model foundation → error propagation.
- Top stacking: preserves general but weaker domain specialisation.
- **Interleaved** (their default): respects the prior that deeper layers encode more abstract features.

### 3.3 SFT follows standard pipeline

- After block expansion pretraining, full SFT (all blocks trained) with ~1M instruction samples.
- No special modification needed for SFT compatibility.

---

## 4. Training Hyperparameters

### 4.1 Pre-training (code + math)

| Hyperparameter | Value |
|---|---|
| Base model | LLaMA2-7B |
| Blocks after expansion | 40 (from 32) |
| Batch size | 1,024 |
| Sequence length | 4,096 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 6% |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Precision | bf16 |
| Attention | Flash-Attention |
| Total steps | 15,900 |
| Hardware | 16× NVIDIA H800 |
| Total compute | ~2,830 GPU-hours |
| Tokens trained | ~80B |

### 4.2 SFT

| Hyperparameter | Value |
|---|---|
| Batch size | 128 |
| Sequence length | 4,096 |
| Learning rate | 2e-5 |
| LR scheduler | Cosine |
| Warmup ratio | 3% |
| Precision | bf16 |

---

## 5. Evaluation Benchmarks

### 5.1 General language tasks (using EleutherAI LM Eval Harness)

| Benchmark | Shots | Task type |
|---|---|---|
| ARC (AI2 Reasoning Challenge) | 25-shot | Grade-school science MCQ |
| HellaSwag | 10-shot | Commonsense inference |
| MMLU | 5-shot | 57-task multitask accuracy |
| TruthfulQA | 0-shot | Truthfulness / factuality |
| Winogrande | 5-shot | Adversarial commonsense |

### 5.2 Math tasks

| Benchmark | Shots | Task type |
|---|---|---|
| GSM8K | 5-shot | Grade-school math word problems |
| GSM8K-PoT | 5-shot | Program of Thought (Python code) |
| MATH | — | Competition mathematics (SFT eval) |

### 5.3 Code tasks (BigCode Evaluation Harness, pass@1 greedy)

| Benchmark | Shots | Task type |
|---|---|---|
| HumanEval | 0-shot | 164 handwritten Python problems |
| MBPP | 3-shot | Crowd-sourced Python problems |

### 5.4 Chat / agent tasks

| Benchmark | Notes |
|---|---|
| MT-Bench | GPT-4 auto-scoring of multi-turn chat quality (1–10 scale) |
| MINT-Bench | 586 multi-turn tool-use instances across 8 datasets (success rate %) |

### 5.5 Law domain (ablation)

| Benchmark | Setting | Task type |
|---|---|---|
| UNFAIR-ToS (LexGLUE) | 4-shot | Multilabel sentence classification (8 labels) |

---

## 6. Key Results

### 6.1 Pretrain comparison (Table 1 in paper)

| Model | ARC | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K | GSM8K-PoT | HumanEval | MBPP | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| **LLaMA-Pro (8.3B)** | **54.10** | 77.94 | 47.88 | 39.04 | 73.95 | 17.89 | 25.42 | 28.66 | 33.20 | **44.23** |
| LLaMA2-7B (base) | 53.07 | 78.59 | 46.87 | 38.76 | 74.03 | 14.48 | 17.68 | 13.05 | 20.09 | 39.62 |
| CodeLLaMA-7B | 39.93 | 60.80 | 31.12 | 37.82 | 64.01 | 5.16 | 25.20 | 33.50 | 41.40 | 37.66 |
| StarCoder-15B | 30.38 | 47.93 | 29.96 | 41.28 | 56.12 | 9.48 | 25.09 | 33.63 | 43.28 | 35.24 |
| LLaMA-7B | 50.94 | 77.81 | 35.69 | 34.33 | 71.43 | 8.04 | 10.46 | 10.61 | 17.04 | 35.15 |
| CrystalCoder-7B | 47.01 | 71.97 | 48.78 | 35.91 | 67.17 | 10.77 | 24.96 | 28.38 | 36.38 | 41.26 |

**Key observation**: LLaMA-Pro maintains general language scores (close to LLaMA2-7B on ARC/HellaSwag)
while significantly improving math (14.48 → 17.89 on GSM8K) and code (13.05 → 28.66 on HumanEval).
CodeLLaMA sacrifices general ability; LLaMA-Pro doesn't.

### 6.2 SFT comparison (Table 1, lower half)

| Model | ARC | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K | GSM8K-PoT | HumanEval | MBPP | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| **LLaMA-Pro-Instruct** | 52.30 | 76.88 | 52.57 | 48.80 | 72.53 | 43.59 | 55.61 | 44.51 | 37.88 | **53.85** |
| LLaMA2-7B-Chat | 52.90 | 78.55 | 48.32 | 45.57 | 71.74 | 7.35 | 19.73 | 14.63 | 21.60 | 40.04 |
| WizardCoder-Python-7B | 41.81 | 65.06 | 32.29 | 36.32 | 61.72 | 4.70 | 17.60 | 42.07 | 47.20 | 38.75 |
| WizardMath-7B | 54.10 | 79.55 | 45.97 | 43.65 | 72.69 | 2.73 | 25.57 | 12.20 | 18.00 | 39.38 |

### 6.3 MT-Bench (GPT-4 scoring, Table 2)

| Model | MT-Bench |
|---|---|
| **LLaMA-Pro-Instruct** | **6.32** |
| LLaMA2-7B-Chat | 6.27 |
| Vicuna-7B | 6.17 |
| CodeLLaMA-7B-Instruct | 5.71 |
| Alpaca-13B | 4.53 |

### 6.4 Perplexity (Table 11)

| Model | LAMBADA (OpenAI) | LAMBADA (Std) | Stack (Code) |
|---|---|---|---|
| LLaMA-2-7B | 3.39 | 4.13 | 9.46 |
| LLaMA-Pro | 3.46 | 4.30 | **5.25** |

Key: code perplexity drops 44.5% (9.46 → 5.25) while general perplexity barely changes (3.39 → 3.46).
This is their empirical evidence for "general ability preservation + domain specialisation."

### 6.5 Ablation — Number of blocks and strategy (Table 5, law domain)

| Method | ARC | HellaSwag | MMLU | TruthfulQA | Winogrande | **Lang Avg** | Unfair-ToS | **Overall Avg** |
|---|---|---|---|---|---|---|---|---|
| Add 1 Block | 52.30 | 77.92 | 38.62 | 37.80 | 73.16 | 55.96 | 67.45 | 61.71 |
| Add 2 Block | 53.16 | 77.91 | 39.62 | 38.92 | 73.01 | 56.52 | 69.57 | 63.05 |
| Add 4 Block | 52.39 | 76.92 | 37.30 | 40.53 | 72.22 | 55.87 | 71.31 | 63.59 |
| **Add 8 Block** | 52.90 | 76.63 | 41.74 | 39.83 | 72.38 | **56.70** | **75.11** | **65.91** |
| Add 16 Block | 51.88 | 76.59 | 41.35 | 40.13 | 71.82 | 56.35 | 75.17 | 65.76 |
| Add 32 Block | 50.77 | 76.72 | 40.68 | 41.66 | 72.77 | 56.52 | 73.93 | 65.23 |
| MoE | 51.45 | 76.51 | 42.47 | 40.13 | 72.23 | 56.56 | 67.27 | 61.92 |
| Full Fine-tuning | 48.81 | 74.49 | 41.13 | 41.49 | 69.14 | 55.01 | 70.63 | 62.82 |
| LoRA (rank=1024) | 53.50 | 78.12 | 44.30 | 40.96 | 73.88 | 58.15 | 65.34 | 61.75 |
| Prefix Stacking (8 blk) | 27.82 | 26.12 | 23.12 | 22.52 | 47.20 | 29.36 | 0.81 | 15.08 |
| Suffix Stacking (8 blk) | 52.56 | 77.89 | 39.10 | 39.03 | 72.38 | 56.19 | 60.98 | 58.59 |

**Finding**: Adding 8 blocks is the sweet spot. Interleaved > Suffix > Prefix. More is not always better
(16 blocks ≈ 8 blocks). MoE ≈ 4 blocks but with more parameters. LoRA preserves general but fails
at domain distribution shift.

### 6.6 Mistral-Pro (Table 4)

| Model | ARC | HellaSwag | MMLU | TruthfulQA | Winogrande | GSM8K | HumanEval |
|---|---|---|---|---|---|---|---|
| Mistral-7B | 60.8 | 83.3 | 62.7 | 42.6 | 78.0 | 39.2 | 28.7 |
| Gemma-7B | 61.9 | 82.2 | 64.6 | 44.8 | 79.0 | 50.9 | 32.3 |
| **Mistral-Pro (ours)** | **63.2** | 82.6 | 60.6 | **48.3** | 78.9 | **50.6** | **32.9** |

---

## 7. What LLaMA-Pro Does NOT Report (INCA's Opportunity)

| Gap in LLaMA-Pro | INCA Paper B can fill |
|---|---|
| No BWT/FWT table — forgetting measured implicitly via perplexity | **Explicit BWT matrix** across 3 domains × all methods |
| Block schedule is fixed (1/period) — no analysis of "when is the right time?" | **E-TIMING ablation** (early / saturation / late / never) |
| Efficiency measured anecdotally ("fewer GPU hours than CodeLLaMA") | **MEM_TRAIN, PAR, ACC/MB table** — rigorous efficiency comparison |
| Saturation never measured — blocks added regardless | **4-signal saturation detector** — EXP_N shows when growth actually fires |
| Single domain stream (code+math together, not sequentially) | **3-domain sequential curriculum** with clear period boundaries |
| No continual learning analysis (BWT, FWT, plasticity, stability) | **Full CL metrics suite** as in §4 of paper outline |
| Law ablation is a side experiment, not the headline | **All domains are the headline** (math → code → science sequential transfer) |

---

## 8. INCA Paper B — Results Section Blueprint

### 8.1 Structural mapping to LLaMA-Pro's paper

| LLaMA-Pro section | Paper B equivalent | Format |
|---|---|---|
| §4.1 Experimental Settings | §4.1 Experimental Settings | Prose + 2 tables |
| §4.2 Pretrain Results (Table 1) | §4.2 Main Results (Table 1) — ACC per domain per method | Table 1: 3 domain × 8 method matrix |
| §4.5 Ablation — blocks (Table 5) | §4.4 Ablations (Tables 3-6) | E-SAT, E-ROUTE, E-TIMING, E-CLS3 |
| Figure 4 (efficiency scatter) | Figure 2 (ACC/MB scatter) | Efficiency bubble chart |
| Table 11 (perplexity — general preservation) | BWT heatmap + bar chart | Figure 3: BWT per domain |
| — (not present) | Table 2: Memory metrics | MEM_TRAIN, PAR, ACC/MB per method |

---

### 8.2 Proposed Table 1 (Main Results — mirrors their Table 1 in spirit)

INCA Paper B Table 1 — Accuracy and Forgetting across 3-Domain Curriculum

```
Method          | P1-Math | P2-Code | P3-Sci | Avg ACC | BWT↑  | FWT↑ | PAR(M)↑ | ACC/MB↑
----------------|---------|---------|--------|---------|-------|------|---------|--------
B1 (FT-all)     |         |         |        |         |       |      |         |
B2 (FT-frozen)  |         |         |        |         |       |      |         |
B3 (EWC)        |         |         |        |         |       |      |         |
B4 (ER)         |         |         |        |         |       |      |         |
B5 (ProgressNet)|         |         |        |         |       |      |         |
B6 (LLaMA-Pro)  |         |         |        |         |       |      | +1/period|
B7 (PackNet)    |         |         |        |         |       |      |         |
**INCA (ours)** |         |         |        |         |       |      | adaptive|
```

- **P1-Math / P2-Code / P3-Sci**: accuracy measured *after all 3 periods* (tests forgetting for P1,P2)
- **Avg ACC**: average across the 3 domains at the end of training
- **BWT**: backward transfer (negative = forgetting; 0 = no forgetting; formula: §3 of paper)
- **FWT**: forward transfer (positive = positive knowledge transfer)
- **PAR(M)**: total parameters added across all periods (millions)
- **ACC/MB**: mean accuracy gain per MB of peak training memory (headline efficiency metric)

> Note: LLaMA-Pro (B6) will always show PAR = fixed +1 block per period.
> INCA PAR is adaptive: if P1 doesn't saturate, no block is added → fewer params, same/better ACC.

---

### 8.3 Proposed Table 2 (Memory Efficiency — new, not in LLaMA-Pro)

INCA Paper B Table 2 — Memory and Efficiency Comparison (B6 vs INCA)

```
Method        | Period | MEM_TRAIN (MB) | MEM_INFER (MB) | PAR_delta (M) | Wall time (s) | ACC_delta | ACC/MB
--------------|--------|----------------|----------------|---------------|---------------|-----------|-------
B6 (LLaMA-Pro)| P1     |                |                | +X (fixed)    |               |           |
              | P2     |                |                | +X (fixed)    |               |           |
              | P3     |                |                | +X (fixed)    |               |           |
INCA          | P1     |                |                | 0 or +Y       |               |           |
              | P2     |                |                | 0 or +Y       |               |           |
              | P3     |                |                | 0 or +Y       |               |           |
```

> This table has **no analog in LLaMA-Pro**. It is INCA Paper B's unique contribution to the
> CL efficiency literature. LLaMA-Pro only says "fewer GPU hours than CodeLLaMA" — no numbers.

---

### 8.4 Proposed Table 3 (E-TIMING ablation — mirrors their Figure 5 + Table 5 spirit)

INCA Paper B Table 3 — When to Expand: Effect of Expansion Timing

```
expand_at    | P1-Math | P2-Code | P3-Sci | Avg ACC | BWT↑ | PAR(M) | Grow events
-------------|---------|---------|--------|---------|------|--------|------------
early        |         |         |        |         |      |        |
saturation   |         |         |        |         |      |        |  (INCA default)
late         |         |         |        |         |      |        |
never        |         |         |        |         |      | 0      | 0
```

Expected finding: `saturation >= late > never >> early` (concave in timing offset).
This is the headline figure distinguishing INCA from LLaMA-Pro: we grow *when data says so*,
not on a fixed schedule.

---

### 8.5 Proposed Figure 1 (Efficiency Scatter — mirrors their Figure 4)

**LLaMA-Pro Figure 4**: X-axis = code task avg, Y-axis = language task avg, bubble size = tokens trained.
Shows LLaMA-Pro on the Pareto frontier.

**INCA Paper B Figure 2**: X-axis = peak training memory (MB), Y-axis = Avg ACC across domains.
Methods as labeled points. INCA should appear in the upper-left (high ACC, low memory).

```
High ACC  |
          |   ★ INCA
          |        ● B6 (LLaMA-Pro)
          |     ● B4 (ER)
          |  ● B3 (EWC)
          | ● B1 (FT-all)
          |______________
         Low             High
                  MEM_TRAIN (MB)
```

---

### 8.6 Proposed Figure 2 (BWT Heatmap — not in LLaMA-Pro, INCA original)

A 3×8 heatmap: rows = test domain (P1-Math, P2-Code, P3-Sci), columns = method.
Cell color = BWT (red = forgetting, green = positive transfer).

This replaces LLaMA-Pro's implicit "general perplexity barely changes" claim with a rigorous
per-domain measurement. Compelling for a CL venue (ACL/EMNLP systems track).

---

### 8.7 Ablation section structure (mirrors LLaMA-Pro §4.5)

| LLaMA-Pro ablation | INCA equivalent | Table # |
|---|---|---|
| # of blocks added (1/2/4/8/16/32) | E-TIMING: expand_at mode | Table 3 |
| LoRA vs MoE vs full fine-tune vs block-exp | E-ROUTE: selector type | Table 4 |
| Law domain transfer | Curriculum order (math→code→sci) | Main Table 1 |
| Token distribution shift | — (not planned for Paper B) | — |
| Block placement (bottom/top/interleaved) | E-SAT: saturation threshold sweep | Table 5 |
| SFT compatibility | — (our model is generative, no SFT stage) | — |

---

## 9. Writing Guidance — Matching LLaMA-Pro's Framing

### 9.1 Phrases to mirror

| LLaMA-Pro phrase | INCA Paper B adaptation |
|---|---|
| "...effectively balances natural language processing and coding capabilities" | "...effectively balances plasticity across three sequential domains while maintaining prior domain accuracy" |
| "...excels in general tasks, programming, and mathematics" | "...excels in cumulative accuracy while dynamically managing block capacity" |
| "...without compromising the old" | (identical — use same framing, it's the CL paper's thesis) |
| "The newly added blocks... are further tuned with only domain-specific corpus" | "The newly added block... is allocated only when the saturation detector confirms that the current block chain cannot absorb further domain signal" |
| "we only fine-tune the newly added blocks while freezing the original blocks" | "INCA freezes previously grown blocks and routes new domain data exclusively through the newly instantiated block" |

### 9.2 Scale argument (pre-empting reviewer concern)

LLaMA-Pro: LLaMA2-7B + 8 blocks (ratio: 8/32 = **25% depth increase**)
INCA: FLAN-T5-base + max 2 blocks (ratio: 2/12 = **17% depth increase per grow event**)

Use "block expansion ratio" to make the comparison fair. State explicitly in §4.1:
"While LLaMA-Pro operates at 7B scale, INCA's architectural contribution is architecture-
agnostic; we validate at 250M (FLAN-T5-base) to enable reproducible experiments on
single-GPU hardware within the resource budget of academic labs."

### 9.3 What to NOT claim

- Do not claim INCA beats LLaMA-Pro on raw accuracy at scale (we use 250M vs 8.3B).
- Claim: INCA achieves **comparable relative improvement** over its frozen baseline (B2) as
  LLaMA-Pro achieves over LLaMA2-7B, while using **fewer parameters on average** and
  requiring **no fixed schedule**.
- The headline claim is **efficiency + adaptivity**, not raw accuracy.

---

## 10. Checklist Before Results Section Draft

- [ ] All 57 sweep jobs in `run_paper_b.py` complete (or sufficient for table population)
- [ ] `memory_log.json` files present for both INCA and B6 runs
- [ ] `metrics_summary.json` (or `regret_matrix.csv`) available per run for BWT computation
- [ ] E-TIMING (expand_at) ablation runs complete (12 jobs: 4 modes × 3 seeds)
- [ ] Mean ± std computed across 3 seeds for all cells
- [ ] At least one row of Table 1 filled per method (partial tables acceptable for draft)

---

## Document Metadata

| Field | Value |
|---|---|
| Author | Nishant Kumar (with Claude assistance) |
| Created | 2026-06-16 |
| Source paper | Wu et al. 2024, arXiv:2401.02415v2 |
| Purpose | LLaMA-Pro extraction + INCA Paper B result alignment blueprint |
| Rule | All code items require Nishant's explicit approval before execution |

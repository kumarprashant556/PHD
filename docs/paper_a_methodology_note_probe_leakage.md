# Paper A — Methodology Note: Probe-Answer Leakage on CC-News v2

> **Status.** Draft v1 — written 2026-06-16, immediately after running
> `scripts/analyze_probe_leakage.py` against the CC-News v2 stream + probe set used by the B1-B7
> sweep. Intended destination: Paper A §3 (Datasets) + Appendix A, and Thesis Chapter 4 §4.3.

> **Companion document.** [`literature_survey_t5_base_em_scores.md`](literature_survey_t5_base_em_scores.md) —
> the absolute-EM-band defence whose conclusions are revised in light of this finding.

---

## 1. Why this note exists

The B1 (naive fine-tune) sweep on CC-News v2 produced a regret matrix in which the
*off-diagonal* values (model after training on period t, evaluated on period j < t probes) were
**equal to or higher than** the diagonal — i.e., the model appeared to exhibit **zero
catastrophic forgetting** on a temporally-organised news stream.

A skeptical reading is: continual fine-tuning on a temporally adjacent slice did *not* erase the
representations needed to answer earlier probes. A more skeptical reading is: the model never had
to "remember" anything, because the answers re-appeared.

This note documents the empirical test of the second reading and concludes that the second
reading is correct for CC-News v2.

---

## 2. Setup

- **Stream**: `datasets/cc_news/processed/stream_v2/` — 4 periods (2017_H1, 2017_H2, 2018_H1,
  2018_H2), each containing a mix of `completion`, `SSD`, `entity_cloze`, `date_cloze` training
  items.
- **Probes**: `datasets/cc_news/processed/probes_v2/` — held-out cloze probes per period;
  ~190 scored items (entity_cloze + date_cloze) per period.
- **Probe targets per period**: 192 / 190 / 186 / 188.
- **Cloze-style training targets per period (unique)**: 42,336 / 42,234 / 49,375 / 12,115.

## 3. Metrics

Two leakage matrices, both computed on **normalised** strings (lowercase, strip punctuation,
collapse whitespace — same normalisation as the runner's evaluation):

- **Target leakage** `L_target[t, j]` — fraction of period-j probe answers that appear as a
  cloze training *target* of some item in periods 0..t. This is the strict measure of "the model
  has been trained to *output* this exact answer."
- **Context leakage** `L_context[t, j]` — fraction of period-j probe answers that appear as a
  substring of the training *input/evidence* text of some item in periods 0..t. This is the
  looser measure of "the model has been exposed to this answer in context, possibly via a
  completion or SSD item."

The script (`scripts/analyze_probe_leakage.py`) is read-only, GPU-free, and was run while the
B1-B7 sweep was in progress.

## 4. Result

### 4.1 Target-leakage matrix (re-exposure as a *training target*)

```
                 probes_2017_H1   probes_2017_H2   probes_2018_H1   probes_2018_H2
after 2017_H1         1.000            0.826            0.796            0.755
after 2017_H2         1.000            1.000            0.844            0.798
after 2018_H1         1.000            1.000            1.000            0.824
after 2018_H2         1.000            1.000            1.000            1.000
```

### 4.2 Context-leakage matrix (re-exposure in *training text*)

```
                 probes_2017_H1   probes_2017_H2   probes_2018_H1   probes_2018_H2
after 2017_H1         1.000            0.879            0.946            0.883
after 2017_H2         1.000            1.000            0.962            0.910
after 2018_H1         1.000            1.000            1.000            0.926
after 2018_H2         1.000            1.000            1.000            1.000
```

### 4.3 What this means in one sentence

After training on period 0 *alone*, the model has been exposed to between **75% and 83%** of
every future period's probe answers as direct training targets, and to between **88% and 95%**
of them in some surrounding text.

### 4.4 Diagonal is trivially 1.000

The diagonal is 1.000 by construction: v2 probes are sampled from the same per-period source
articles as the training stream, so every period-j probe answer is present in period-j training
data. The diagonal is not the surprising part.

### 4.5 The off-diagonal is the surprising part

The off-diagonal cells L[t, j] with t < j (i.e., "before period j has been trained on, how much
of its probe answers has the model seen?") are *all* ≥ 0.75 for target leakage and ≥ 0.88 for
context leakage. This is the result that explains the flat-BWT pattern.

---

## 5. Interpretation — what we can and cannot claim

### 5.1 What the CC-News v2 numbers measure

They measure how well the model **fits a temporally-organised news memorisation task** in which
~80% of all "test" answers re-appear in any training slice. They do **not** measure long-horizon
factual recall under genuine distribution shift.

### 5.2 The flat BWT row is fully explained by leakage

For the B1 regret matrix, the observation R[1,0] = 0.219 > R[0,0] = 0.208 means: after training
on period 1, period-0 probe accuracy did not drop. The leakage analysis explains why directly:
**82.6%** of period-0 probe answers re-appear as period-1 training targets, and **87.9%** appear
as substrings of period-1 training text. The model is being **re-taught** the period-0 answers
during period-1 training. There is no opportunity for catastrophic forgetting to manifest.

### 5.3 What we *cannot* claim from CC-News v2 alone

- **No claim of "INCA reduces BWT vs B1"** can be made on CC-News v2, because the BWT signal is
  saturated at the leakage floor for every method that fine-tunes on the stream.
- **No claim of "this is a temporal-drift benchmark"** can be made, because >75% of the "drift"
  is illusory.
- **No claim of "X% accuracy is good for closed-book QA"** can be made by direct comparison to
  Roberts 2020 / FLAN-T5 numbers, because those numbers are computed on tasks with no overlap
  between train and test answers.

### 5.4 What we *can* claim from CC-News v2

- **Convergence-speed / data-efficiency** comparisons (how fast does method X reach a given
  accuracy on the in-period probes?). Leakage does not confound this.
- **Stability of representations** (do non-fine-tuned layers drift?). Leakage does not confound
  this.
- **The methodological finding itself** (news-stream temporal benchmarks contain massive
  probe-answer leakage that needs to be measured and corrected). This is a publishable
  contribution.

---

## 6. Implication for Paper A architecture

The CAPSEL PhD Roadmap allocates CC-News as the Phase 1 pilot benchmark and TiC-LM as the
Phase 4 headline benchmark, with a Plan B contingency that flips CC-News into the headline if
TiC-LM slips. The leakage finding **closes off the Plan B path** for any claim that rests on
BWT, and recommends one of three architectures for the final paper.

### Architecture 1 — *Recommended*: TiC-LM is the only BWT venue

- CC-News appears as Section 3 (Pilot benchmark) with the **leakage matrix in Appendix A** as
  the explicit reason BWT-on-CC-News is not the headline.
- TiC-LM Track A is Section 5 (Main results) — all BWT / forgetting / regret claims live here.
- A leakage analysis identical to the one in this note is run on TiC-LM **as a precondition**
  for using it as the headline; if TiC-LM shows similar leakage, the entire field's framing
  needs revision and the paper pivots to a methodology contribution.

### Architecture 2 — Drift-only subset of CC-News

- Filter the v2 probes down to those whose normalised answer does **not** appear in any other
  period's training stream. This will leave roughly 30-60 probes per period (rough estimate from
  the 0.17-0.25 of probes that are "unique to their period" in the target-leakage matrix).
- The leakage-filtered probe set can support a CC-News BWT claim — small N, but methodologically
  defensible.
- This requires a new script (`scripts/build_drift_only_probes.py`) and a re-run of the sweep on
  the filtered probe set, but **not** a re-run of training. Eval-only.

### Architecture 3 — Methodology paper first, baselines paper later

- A short paper *("Why news-stream temporal benchmarks under-report forgetting")* using
  StreamingQA, TemporalWiki and CC-News v2 leakage matrices as the empirical evidence, with the
  drift-only sub-evaluation as the proposed fix.
- The CAPSEL/INCA paper then cites this methodology paper rather than including the leakage
  matrix as an appendix.

---

## 7. Recommended action items

Listed in order of cost vs. value:

1. **Write `scripts/build_drift_only_probes.py`** (≈30 lines). Re-run eval on the leakage-free
   probe subset against the existing checkpoints — no training cost. This unlocks Architecture 2.
2. **Add the leakage matrices to Paper A Appendix A** as Figure A1 (target) and Figure A2
   (context), with the same captions used here.
3. **Run `analyze_probe_leakage.py` on TiC-LM Track A** before committing to it as headline.
   This is a precondition test; if TiC-LM is similarly leaky, the whole paper plan changes.
4. **Update §3 (Datasets) of the Paper A draft** with the paragraph in §6 of this note.
5. **Update Thesis Chapter 4 §4.3** with the same paragraph plus the leakage matrices.
6. **Re-run B1 on the drift-only probe subset and report the corrected regret matrix.** This is
   the cleanest experimental answer to the reviewer question "isn't this just memorisation?"

The first six can be done without re-training; only items 3 and 6 require GPU time (eval-only).

---

## 8. Proposed text for Paper A §3.2 (Datasets — CC-News v2)

> We use a temporally-stratified slice of CC-News (2017_H1, 2017_H2, 2018_H1, 2018_H2) as a
> pilot benchmark to validate our continual learning pipeline. CC-News was chosen for its
> coverage of named entities and dates across a two-year span; however, a leakage analysis of
> the entity-cloze and date-cloze probes against the training stream (Appendix A) shows that
> 75-83% of any later period's probe answers appear as cloze training targets in the *earliest*
> period, and 88-95% appear as substrings of the earliest period's training text. The CC-News
> diagonal accuracies therefore characterise the model's data-fit on a high-overlap news stream
> rather than its long-horizon recall under distribution shift, and we use TiC-LM Track A
> (Section 5) as the primary venue for forgetting and backward-transfer claims. CC-News
> nonetheless remains useful for convergence-speed and representation-stability comparisons,
> which we report in Section 4.

This paragraph (a) acknowledges the limitation up-front, (b) defines the metric the CC-News
numbers actually measure, and (c) tells the reviewer where the actual BWT result lives. It
removes the leakage objection before the reviewer can raise it.

---

## 9. Proposed Appendix A caption

> **Figure A1 — Target leakage matrix for CC-News v2.** Each cell `L[t, j]` reports the fraction
> of period-j probe answers that appear as a cloze training target in periods 0..t (normalised
> string match). The off-diagonal cells with t < j characterise the *forward leakage* of probe
> answers: how much of the "future" test set the model has already been trained to output by
> the time it gets to period j. Values in the 0.75-0.84 range across all off-diagonal cells
> imply that the CC-News diagonal accuracies are dominated by re-exposure, not pure recall.
>
> **Figure A2 — Context leakage matrix for CC-News v2.** As Figure A1, but counting any
> substring match in the training *input/evidence* text rather than only cloze training targets.
> Off-diagonal values in the 0.88-0.95 range confirm that any T5-class model trained on this
> stream has been exposed to the textual context of any later period's probe answers in some
> form, regardless of whether the answer was a designated training target.

---

## 10. References

- `scripts/analyze_probe_leakage.py` — the analyser (read-only, GPU-free, safe to run alongside
  training).
- `results/leakage/leakage_target.csv` — the target-leakage matrix.
- `results/leakage/leakage_context.csv` — the context-leakage matrix.
- `results/leakage/leakage_summary.json` — full result with per-period stats.
- `docs/literature_survey_t5_base_em_scores.md` §3.6 — the EM-band caveat that depends on this
  finding.
- `docs/CAPSEL_PhD_Roadmap.pdf` — the roadmap whose Plan-B contingency this note resolves.

---

## 11. Document metadata

| Field        | Value                                                                          |
|--------------|--------------------------------------------------------------------------------|
| Author       | Nishant Kumar                                                                  |
| Date         | 2026-06-16                                                                     |
| Purpose      | Document the CC-News v2 leakage finding for Paper A methodology + Thesis Ch. 4 |
| Used in      | Paper A §3.2 + Appendix A; Thesis Ch. 4 §4.3                                   |
| Status       | Draft v1 — needs the same analysis re-run on TiC-LM Track A as a precondition  |
| Open actions | (1) drift-only probe builder; (2) TiC-LM leakage check; (3) B1 re-eval on drift-only subset |

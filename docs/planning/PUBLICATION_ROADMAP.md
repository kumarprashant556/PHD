# Publication Roadmap — CAPSEL / INCA (Paper B)
> Last updated: 2026-06-17
> Status: Experiments running on MPS (5-domain sweep pending). ICLR 2027 is primary target.

---

## Quick Reference Table

| # | Venue | Type | Deadline (est.) | Pages | Archival? | Priority |
|---|-------|------|-----------------|-------|-----------|----------|
| 1 | **arXiv preprint** | Preprint | Aug 2026 (self-set) | 6–8 | No | ⭐ Do first |
| 2 | **NeurIPS 2026 CL Workshop** | Workshop | ~Sep 15, 2026 | 4 | No | ⭐ Early feedback |
| 3 | **ICLR 2027** | Conference | ~Oct 7, 2026 | 9+appendix | Yes | ⭐ Primary target |
| 4 | **CoLLAs 2027** | Conference | ~Mar 2027 | 8 | Yes | Backup / follow-on |
| 5 | **ICML 2027** | Conference | ~Jan 30, 2027 | 9+appendix | Yes | Backup if ICLR rejected |
| 6 | **TMLR** | Journal | Rolling | No limit | Yes | Last resort / extended version |

> **Archival** = counts as a published paper; double-submission with ICLR is prohibited.
> Non-archival workshops can be submitted simultaneously with ICLR — always verify at submission time.

---

## Timeline View

```
2026
────────────────────────────────────────────────────────────────────────────
Jun 17  ← TODAY
        ▶ Kill old 3-domain run, restart with 5-domain + fp16 + signal logging
        ▶ Begin smoke tests → full sweep

Jul     ▶ 5-domain sweep running (all 81 jobs)
        ▶ Begin drafting arXiv preprint (skeleton, figures)

Aug     ▶ Sweep results in → populate Tables 1–6 in PAPER_INCA.md
        ▶ Generate EXP_T histogram, CKA/RIR trajectories, AGPM bar chart
        ▶ SUBMIT ARXIV PREPRINT  ← timestamp established

Sep     ▶ NeurIPS 2026 workshop deadline (~Sep 15) — submit 4-page abstract
        ▶ Begin ICLR 2027 full paper (expand arXiv → 9 pages + appendix)

Oct 1–7 ▶ ICLR 2027 abstract deadline (estimated ~Oct 1)
Oct 7–10▶ ICLR 2027 full paper deadline (estimated ~Oct 7–10)
        ▶ SUBMIT TO ICLR 2027  ← primary target

Nov     ▶ ICLR reviews arrive (~Nov 2026)
        ▶ Prepare rebuttal (2-week window, typically)

Dec     ▶ NeurIPS 2026 conference (Vancouver) — workshop poster
        ▶ Collect community feedback, note repeated criticisms

────────────────────────────────────────────────────────────────────────────
2027

Jan     ▶ ICLR 2027 decisions
        ▶ IF REJECTED: submit revised version to ICML 2027 (~Jan 30 deadline)
        ▶ IF ACCEPTED: celebrate, prepare 15-min talk

Feb     ▶ ICML 2027 deadline (if ICLR rejected)

Mar–Apr ▶ CoLLAs 2027 deadline (whether or not ICLR accepted)
        ▶ Submit companion/follow-on paper OR extended version

May     ▶ ICLR 2027 conference (if accepted)

Jul     ▶ ICML 2027 conference / CoLLAs 2027 conference

Ongoing ▶ TMLR submission if needed (rolling, no deadline, highest bar)
```

---

## Venue 1 — arXiv Preprint

**Target date:** August 2026 (before NeurIPS workshop deadline)
**Format:** 6–8 pages, NeurIPS / ICLR LaTeX style
**Cost:** Zero. No reviewers.

### Why do this first
The arXiv timestamp is your legal priority claim. In ML, this matters more than any
other IP protection. Once it's posted, no one can publish the same idea and claim
they invented it first. This is especially important for a PhD student — your advisor,
competitors, and future employers will look at the timestamp.

### What goes on the plate

| Section | Include? | Notes |
|---------|----------|-------|
| Abstract | ✅ Full | Lead with the saturation trigger insight, not the architecture |
| Introduction | ✅ Full | Motivate catastrophic forgetting + architecture stasis as dual problem |
| Related work | ✅ ~1 page | Progressive Nets, PackNet, PNN, DEN, LLaMA-Pro, EWC |
| Architecture (§3) | ✅ Full | Block-chain growth, S-QKV selector, UCLBR — all of it |
| 4-Signal consensus (§3.2) | ✅ Full | RIR, CKA drift, grad-norm EMA decay, loss plateau — all described |
| CLS theory grounding | ✅ Yes | This elevates from "engineering trick" to theoretically motivated |
| Domain curriculum (§4.1) | ✅ 5-domain table | Show the Q/A framing |
| Preliminary results | ✅ 1–2 domains | Even partial results — enough to show learning happens |
| EXP_T histogram | ✅ Yes | Your most unique figure. No other CL paper has this. |
| Full BWT/FWT ablation table | ❌ No | "Full results in submission" |
| E-TIMING vs E-UNIFORM vs E-RANDOM | ❌ No | Core ICLR contribution, save it |
| All 6 baselines | ❌ No | Show vs LLaMA-Pro only |

### Tone
Frame it as: *"We present INCA, a growing transformer trained with a saturation-driven
curriculum. We describe the mechanism and report early experimental results. Full
ablations are in preparation for a forthcoming conference submission."*
This is completely normal on arXiv — it signals "ongoing work" and discourages scooping
without overpromising.

---

## Venue 2 — NeurIPS 2026 Workshop on Continual Learning

**Deadline:** ~September 15, 2026 *(exact date announced when workshop list is released ~Aug 2026)*
**Format:** 4 pages (extended abstract), NeurIPS style, non-archival
**Notification:** ~October 2026
**Presentation:** December 2026 (Vancouver, BC) — poster, possibly 5-min spotlight

### Why this venue
The NeurIPS CL workshop is the single best place to get your idea stress-tested by
the 50 researchers who know this area best. They will immediately tell you:
- Whether your saturation trigger is distinguishable from existing CL criteria
- Whether the growing architecture is truly novel vs prior continual architecture work
- Whether the CLS framing is seen as strong or hand-wavy
This feedback arrives 6–8 weeks *before* ICLR reviews, giving you time to sharpen.

> **Check:** NeurIPS 2026 workshop list at https://neurips.cc/virtual/2026/workshops
> (released ~Aug 2026). Also look for: "Workshop on Distribution Shifts," "Efficient
> Foundation Models," "Foundation Model Fine-tuning" as backup targets if no dedicated CL workshop.

### What goes on the plate (4 pages only — be brutal about cutting)

| Section | Words (est.) | Priority |
|---------|-------------|----------|
| Intro + problem framing | ~300 | Must have |
| Architecture overview (diagram) | ~250 + 1 fig | Must have |
| 4-signal consensus mechanism | ~250 | Must have |
| Preliminary result vs LLaMA-Pro (1 table or 1 figure) | ~150 | Must have |
| EXP_T histogram | 1 fig, ~100 caption | Include — unique, memorable |
| Related work (2 paragraphs) | ~200 | Should have |
| "Ongoing + future work" | ~100 | 1 short paragraph |

**Total: ~4 pages.** No appendix. No full tables.

### What to withhold
- E-TIMING ablation (your primary ICLR contribution)
- Exact hyperparameter values (reviewers don't need them at this stage)
- Full 5-domain forgetting matrix
- Any comparison to baselines other than LLaMA-Pro

### Key questions to plant in the paper
Put these explicitly in "Future Work" or the poster — you want people to react:
1. *"Is the asymmetric 4-signal consensus trigger novel enough vs. prior CL stopping criteria?"*
2. *"Does variability of EXP_T across domains carry information (or is it noise)?"*
3. *"Does the CLS hippocampus/neocortex framing apply here, or is it a stretch?"*

---

## Venue 3 — ICLR 2027 (Primary Target)

**Abstract deadline:** ~October 1, 2026 (estimated; ICLR 2026 was Oct 1)
**Full paper deadline:** ~October 7–10, 2026 (estimated)
**Reviews arrive:** ~November 2026
**Rebuttal period:** ~2 weeks, late November 2026
**Decisions:** January 2027
**Conference:** May 2027

> ICLR is the right venue because: (1) architecture-level innovation is valued there;
> (2) CLS/neuroscience-grounded ML is a growing track; (3) reproducibility culture means
> your signal logs and figures will be appreciated, not penalized.

### What goes on the plate (9 pages + unlimited appendix)

**Main paper (9 pages):**

| Section | Target length | Key content |
|---------|--------------|-------------|
| Abstract | 250 words | INCA, 5-domain, headline number (BWT improvement vs LLaMA-Pro) |
| §1 Introduction | 1 page | Dual problem: forgetting + architecture stasis. Saturation insight. |
| §2 Related Work | 1 page | Progressive Nets, PackNet, LLaMA-Pro, EWC, GEM, CLS theory |
| §3 Architecture | 1.5 pages | Block-chain, S-QKV (+ D=1024 param count), UCLBR |
| §3.2 Saturation Detector | 1 page | All 4 signals, asymmetric conjunction rule, state_dict exposure |
| §4 Experimental Setup | 0.75 page | 5-domain table, FLAN-T5-large, hardware, hyperparams |
| §5 Results — Main table | 0.75 page | BWT, FWT, AGPM, Params — INCA vs all 6 baselines |
| §5.4 E-TIMING (Headline) | 0.75 page | Figure 1 (timing conditions), Figure 2 (EXP_T histogram) |
| §6 Discussion + Limitations | 0.5 page | Scale (780M only), MPS thermal throttling, future work |
| References | ~0.5 page | 25–35 citations |

**Appendix (no page limit):**
- Full hyperparameter table
- All 5-domain individual BWT/FWT scores
- Full forgetting matrix (5×5)
- CKA/RIR trajectory plots (Figure from signals.csv)
- Ablation: S-QKV vs S-FULL vs embedding_query
- Proof / derivation of saturation criteria if formalized
- Dataset statistics (n_per_period, class balance, token length distribution)

### ICLR-specific advice
- **ICLR reviewers penalize missing ablations.** E-TIMING vs E-UNIFORM vs E-RANDOM
  is non-negotiable — it must be there.
- **Reproducibility.** Link the GitHub repo in the paper. ICLR rewards this.
- **Framing matters.** Do not frame as "we built a system." Frame as:
  *"We identify saturation as the missing inductive bias for continual architecture growth"*
  — a scientific claim with evidence.
- **Score distribution reality:** Most accepted papers at ICLR score 6/6/8 or 6/8/8.
  A 5/6/8 with a strong rebuttal can still make it. Don't panic at first reviews.

---

## Venue 4 — CoLLAs 2027 (Conference on Lifelong Learning Agents)

**Deadline:** ~March–April 2027 *(CoLLAs 2026 was July 2026; 2027 edition TBD)*
**Format:** 8 pages, PMLR style, fully archival
**Conference:** ~July 2027

### When to use this venue
- **If ICLR 2027 is rejected:** Revise based on reviews, submit to CoLLAs 2027.
  CoLLAs is specifically the continual learning community — arguably a better fit
  than ICLR for a CL-heavy paper. Acceptance rate is higher (~30% vs ~25% ICLR).
- **Even if ICLR is accepted:** Submit a focused companion paper on the
  *saturation detector alone* (without the full architecture paper) to CoLLAs.
  This gives the detector mechanism its own venue and citation.

### What goes on the plate (CoLLAs companion paper)
If submitting standalone saturation-detector paper after ICLR:
- Focus entirely on the 4-signal consensus mechanism
- Compare to other CL stopping criteria: task-boundary detection, gradient episodic memory triggers
- Include the CKA/RIR trajectory figures in detail
- Position as: *"A general-purpose saturation detector for continual learning"*
  (applicable beyond INCA)

---

## Venue 5 — ICML 2027 (Backup if ICLR rejected)

**Deadline:** ~January 30, 2027 *(estimated; ICML 2027 not yet announced)*
**Format:** 9 pages + appendix, PMLR style
**Conference:** ~July 2027

### When to use
ICLR decisions come in January 2027 — the same month as the ICML deadline.
If ICLR rejects (likely with reviews in hand), you have ~3 weeks to revise and
resubmit to ICML. This is a tight but viable path.

### Differences from ICLR submission
- ICML reviewers tend to weight empirical rigor slightly more than theoretical framing
- Reduce the CLS framing emphasis; increase the empirical ablation depth
- Add any experiments ICLR reviewers asked for in rebuttal but you couldn't run in time

---

## Venue 6 — TMLR (Transactions on Machine Learning Research)

**Deadline:** Rolling (submit any time)
**Format:** No page limit, journal style
**Review turnaround:** ~3–6 months
**Acceptance bar:** High, but "reject without resubmission" is rare — usually "revise and resubmit"

### When to use
- If ICLR + ICML both reject (unlikely but possible)
- If you want a definitive, extended version of the work post-conference
- TMLR is gaining prestige fast (JMLR equivalent for DL papers)
- The extended version can include: Phase 5 XL-scale results, lateral adapter ablation,
  full cross-domain generalization study

---

## What NOT to Submit / Avoid

| Venue | Why to skip |
|-------|-------------|
| EMNLP / ACL / NAACL | NLP venues — reviewers evaluate language generation quality, not CL architecture. Wrong crowd. |
| AAAI 2027 | Deadline Oct/Nov 2026 — conflicts with ICLR prep. Lower prestige for ML architecture work. |
| IEEE TNNLS / TCYB | Too slow (18-month review). Only if you need journal credit specifically. |
| NeurIPS 2026 main track | Deadline already passed (~May 2026). |
| Any "predatory" open-access journal | Never. |

---

## How Much to Reveal — Decision Matrix

```
STAGE           MECHANISM    RESULTS      ABLATIONS    CODE
─────────────────────────────────────────────────────────────
arXiv           FULL         PARTIAL      NONE         Link repo
NeurIPS WS      FULL         1-2 domains  NONE         Mention arXiv
ICLR 2027       FULL         FULL (5-dom) FULL         Reproduce pkg
CoLLAs 2027     FULL         FULL         Extended     Same repo
TMLR            FULL         FULL+XL      Full+extra   Full release
```

**The rule:** You reveal the *mechanism* fully at every stage once the arXiv is up —
because the arXiv timestamp protects you. What you withhold at early stages is
*experimental depth* (ablations, full baselines), not the core idea.

---

## Scooping Risk Assessment

| Risk factor | Level | Mitigation |
|-------------|-------|------------|
| Someone independently builds growing transformer + saturation trigger | LOW | Your combination is specific; arXiv timestamp protects |
| Reviewer leaks ICLR idea to their group | LOW-MED | Standard ICLR double-blind; anonymous submission |
| A workshop reviewer publishes first | VERY LOW | Workshop papers are non-archival; you have arXiv priority |
| Large lab (Google/Meta) publishes similar at scale | MED | They work at 7B+; your 780M contribution is still valid at that scale |
| Forgetting the CL-specific novelty in framing | MED | Always lead with saturation insight, not architecture diagram |

---

## Per-Venue Submission Checklist

### arXiv (August 2026)
- [ ] Results from at least 2-3 domains (1 complete INCA run)
- [ ] EXP_T histogram generated from signals.csv
- [ ] At least 1 CKA/RIR trajectory figure
- [ ] 1 comparison table (INCA vs LLaMA-Pro, 2-3 domains)
- [ ] GitHub repo public (or at minimum linked with note "code available on acceptance")
- [ ] NeurIPS/ICLR LaTeX template used

### NeurIPS 2026 Workshop (September 2026)
- [ ] arXiv preprint already posted (required before this)
- [ ] 4-page hard limit respected (use \vspace tricks judiciously)
- [ ] EXP_T histogram included (your distinctive figure)
- [ ] "Ongoing work" framing in abstract and conclusion
- [ ] 3 open questions planted for poster discussion
- [ ] Verified workshop's double-submission policy for simultaneous ICLR submission

### ICLR 2027 (October 2026)
- [ ] Full 5-domain sweep complete (81 jobs)
- [ ] Tables 1–6 populated in PAPER_INCA.md
- [ ] E-TIMING ablation complete (4 conditions × 5 seeds)
- [ ] All 6 baselines run and compared
- [ ] Appendix with full forgetting matrix and hyperparams
- [ ] GitHub repo with reproducibility package (requirements.txt, configs, launch script)
- [ ] Abstract registered ~Oct 1 (even if paper not fully written)
- [ ] Double-checked: no self-identifying info in blind submission

---

## Notes on Advisor Strategy

- Share the arXiv draft with your advisor **before** posting — they may want co-authorship
  or may know of directly competing work you should cite
- For workshop submissions, advisors often don't need to review (low stakes)
- For ICLR, get advisor sign-off at least 1 week before deadline — do not surprise them
- If your advisor is not active in CL/NLP, consider finding a senior PhD student or
  postdoc in the area to give the ICLR draft a pre-review ("mock review") 2 weeks before submission

---

*Document generated: 2026-06-17 | Maintained alongside PAPER_INCA.md and TASKS.md*

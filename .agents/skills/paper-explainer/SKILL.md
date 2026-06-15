---
name: paper-explainer
description: Read an arXiv paper (URL, ID, or uploaded PDF) and teach it back to the user as a polished study PDF. The output has three required sections — Intuition, Math (with full step-by-step derivations of the central results), Continual-Learning Connection (broad CL framing only, no project-specific tie-ins) — plus an optional Code section that the skill explicitly asks the user about at invocation time. Use this skill whenever the user pastes an arXiv link or ID, uploads a paper PDF, or asks Codex to "explain", "walk me through", "teach me", "read", "summarize", "break down", or "help me understand" a paper, even if they don't say the word "skill" or "PDF". Strongly prefer this skill over a quick chat answer whenever the user is in research / study mode and the input is a research paper, because the deliverable is a saved study artifact rather than ephemeral conversation.
---

# Paper Explainer

## What this skill does

Take a research paper as input — an arXiv URL, an arXiv ID like `2204.04411`,
or an uploaded PDF — and produce a study-quality PDF that teaches the paper to
the reader. The PDF has a fixed shape: **Intuition**, **Math (with full
derivations)**, **CL Connection**, and an optional **Code** section.

The audience is a graduate-level continual-learning researcher who already
knows linear algebra, probability, and basic deep learning, but who hasn't yet
read this specific paper. Calibrate to that.

## Workflow

1. **Resolve and read the paper.** Accept whichever form the user supplied.
   See [references/fetching.md](references/fetching.md) for arXiv URL/ID
   handling and PDF parsing — read it the first time you do this; the rules
   for arXiv canonicalisation and the PDF text-extraction tradeoffs aren't
   obvious.

2. **Decide on the Code section.** Code is optional, and the skill is supposed
   to ask. After you've read the abstract and introduction (so you know what's
   in the paper), ask the user a single, concrete question — see
   [Asking about code](#asking-about-code-once-per-invocation) below. Don't
   ask before reading the paper, because the question is more useful when you
   can describe what the central method actually is.

3. **Identify the spine.** Before drafting, write a short scratch note (in
   your head, not in the PDF) that names: (a) the **central claim** in one
   sentence, (b) the **central equation or algorithm** that earns the claim,
   and (c) the **assumption that, if dropped, would break the result**. The
   four sections all flow from this spine — if you can't name these three
   things, re-read the paper before drafting.

4. **Draft the four sections.** Use the structure and word-counts in
   [Output structure](#output-structure) below. For the Math section,
   [references/derivations.md](references/derivations.md) explains how to do
   full step-by-step derivations well — read it before drafting the Math
   section if it's been a while.

5. **Write the CL Connection.** Even if the paper isn't about continual
   learning, find the legitimate connection (and if there really is none, say
   so explicitly rather than forcing it).
   [references/cl_taxonomy.md](references/cl_taxonomy.md) is a one-page map of
   the CL field — read it once for context the first time, then refer back as
   needed.

6. **Render to PDF.** Author in markdown with `$...$` and `$$...$$` math, then
   call `scripts/build_pdf.sh <input.md> <output.pdf>`. The script wraps
   pandoc + xelatex with a study-document template that gets math rendering,
   numbered sections, and a TOC right. Don't reinvent the toolchain — see
   [references/rendering.md](references/rendering.md) only if the script
   complains or you need to extend it.

7. **Save and link.** The output PDF goes to
   `/sessions/quirky-confident-mccarthy/mnt/WorkingDir/explained_papers/<arxiv-id-or-slug>.pdf`.
   Reply to the user with a 5-line tl;dr of the paper plus a `computer://` link
   to the PDF. Five lines, not five paragraphs — the PDF is the artifact, the
   chat reply is just the breadcrumb.

## Output structure

Use this exact section ordering in the rendered PDF. Word counts are guides,
not laws — a 30-page theory paper warrants more Math; a 4-page workshop paper
warrants less of everything.

### Title block
- Paper title, authors, venue/year, arXiv ID.
- A two-sentence "what this paper is about" line directly under the title.
  This sets calibration before the reader hits Intuition.

### 1. Intuition (≈ 400–700 words)
The single most important section. The reader should leave with a working
mental model **before** they touch any equations. Aim for:

- **The problem in plain language**: what was broken or unknown before this
  paper? Which prior approach was failing, and why?
- **The key idea in one sentence**, then unpacked over a paragraph. If you
  can't compress it to one sentence, you don't yet understand it — re-read.
- **An analogy or thought experiment**, where one helps. Don't force it.
- **What's surprising**: papers exist because something non-obvious is true.
  Name what that is.

Avoid restating the abstract verbatim. The abstract is calibrated to fellow
specialists; Intuition is calibrated to a smart reader who isn't yet inside
the problem.

### 2. Math (≈ 800–1500 words, with full derivations)

This is the section the user explicitly asked for "heavy — full derivations".
That means: when the paper says "by the chain rule" or "see Appendix B", you
fill in the missing steps. See [references/derivations.md](references/derivations.md)
for the craft of doing this well — the short version is:

- State assumptions and notation **before** the first equation. If the paper
  uses notation that conflicts with common convention, flag it.
- Present the **central result first**, then derive it. Don't make the reader
  read 12 pages of setup hoping it pays off.
- Show every algebraic step that a careful graduate student would reproduce
  with paper and pen — but annotate each step with a one-line justification
  (chain rule / Bayes / Jensen / etc.) so the reader is never guessing.
- After the derivation, write a **one-paragraph "what this means"** that
  re-grounds the equation in the paper's claim.

If the paper has multiple results, derive **the most central one** in full,
and outline the others briefly. A great Math section that derives one result
deeply beats a mediocre one that surveys five.

### 3. CL Connection (≈ 300–600 words, broad-CL framing only)

The user wants this section anchored in the **broad continual-learning
literature**, not in any specific project. So:

- Name where this paper sits taxonomically. The taxonomy is in
  [references/cl_taxonomy.md](references/cl_taxonomy.md): regularization,
  replay, parameter-isolation, prompt-based, distillation, modular/expansion,
  test-time adaptation. Pick one and justify it.
- Cite 3–5 closely related CL papers and contrast trade-offs. Be honest about
  what this paper does *worse* than competitors, not just better.
- If the paper isn't about CL at all but has CL implications (e.g. an
  optimisation paper that bears on plasticity, a representation-learning paper
  that informs feature drift), state the legitimate connection and skip the
  forced cross-walk.
- **Don't reference specific projects, baselines, codenames, or the user's
  own work.** This is broad-field framing, not a project memo.

### 4. Code (optional — only if the user said yes when asked)

If the user opted in, follow the format they chose:

- *Pseudocode + Python sketch*: a pseudocode box first (write it as a fenced
  ````pseudo```` markdown block — the template renders these with line
  numbers and a frame), then a ≤30-line PyTorch or NumPy sketch (fenced
  ````python```` block) that compiles and runs in isolation.
- *Runnable Python sketch*: skip the pseudocode block, jump to the sketch.
- *Pseudocode only*: ````pseudo```` block, no Python.
- *Author's repo + pseudocode*: link to the official implementation if one
  exists (search GitHub for the paper's title or first-author repo), plus a
  short ````pseudo```` summary of the central method.

If the user said skip, omit this section entirely — don't leave a placeholder.

See [references/rendering.md](references/rendering.md) for the exact markdown
syntax for math, pseudocode boxes, and tables that the template expects.

## Asking about code (once per invocation)

After fetching and reading the paper but before drafting, ask the user one
question to confirm the code section. Phrase the question with the paper's
actual method named — generic prompts get generic answers. For example, after
reading EWC:

> "EWC is short — its core method is one Fisher-weighted regularisation term.
> Want me to include a Code section with that term implemented? Options:
> (1) pseudocode + ≤30-line PyTorch sketch, (2) Python sketch only,
> (3) pseudocode only, (4) link to author's repo + pseudocode, (5) skip."

Default to skip if the user is silent, but ask once. Don't repeat the
question if they've already answered it for this invocation.

## Length calibration

Short workshop papers (4–8 pages, 1 method): ≈ 1500–2500 words total in the
PDF. The reader can re-read the paper itself in 30 minutes — don't make the
explainer longer than the original.

Standard ML papers (10–20 pages, 1–2 methods + experiments): ≈ 2500–4500 words
total.

Theory-heavy or survey papers (20+ pages, multiple results): ≈ 4500–7000
words. At this length, longer Math is justified — but the reader should still
be able to read the explainer in one sitting.

## What success looks like

A reader who has *not* read the paper should, after reading the explainer:

1. Be able to state the central claim in one sentence.
2. Reproduce the central derivation on a whiteboard with prompts (not from
   memory — but they shouldn't be lost).
3. Place the paper in the CL taxonomy and name two related works.
4. Decide whether they need to read the paper themselves, or whether the
   explainer was enough.

If the explainer doesn't get a reader to those four outcomes, it's too long
or too shallow. Cut or deepen.

## Anti-patterns

- **Restating the abstract.** The abstract is the paper's calibration to
  specialists. Your job is to teach, not to summarise.
- **Hand-waving derivations.** "It can be shown that..." is the failure mode
  this skill exists to prevent. If you find yourself writing it, you owe the
  reader the next two lines of algebra.
- **Forced CL connections.** If the paper is about a vision transformer with
  no CL angle, write that honestly: "This paper has no direct CL contribution,
  but its [feature X] would be a useful building block for [class of CL
  method] because [reason]." Don't invent a connection that isn't there.
- **Code dumps.** A 200-line training loop teaches nothing. The Code section
  exists to crystallise the central method, not to be a working trainer.
- **Markdown-as-PDF without rendering.** The user asked for a PDF. Don't ship
  a `.md` file and call it done — run the build script.

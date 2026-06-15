# Rendering markdown to PDF

The build script `scripts/build_pdf.sh` handles the common path. Read this
file only if the script complains, you need to extend the template, or
something about the math rendering looks wrong.

## What the script does

`scripts/build_pdf.sh INPUT.md OUTPUT.pdf` runs:

```bash
pandoc INPUT.md \
  --from markdown+tex_math_dollars+raw_tex \
  --to pdf \
  --pdf-engine=xelatex \
  --template=assets/study.tex \
  --toc --toc-depth=2 \
  --variable=geometry:margin=1in \
  --variable=fontsize=11pt \
  --variable=mainfont:"Latin Modern Roman" \
  --variable=mathfont:"Latin Modern Math" \
  --variable=monofont:"Latin Modern Mono" \
  -o OUTPUT.pdf
```

## Authoring conventions

Write the explainer in markdown, with these expectations:

- **Inline math:** `$...$`. Standard.
- **Display math:** `$$...$$` on its own line, blank lines above and below.
- **Numbered display math:** wrap in raw LaTeX, since pandoc-markdown alone
  doesn't number equations:
  ```
  \begin{equation}
  \mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_{\text{new}}(\theta)
    + \tfrac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2
  \label{eq:ewc}
  \end{equation}
  ```
- **Algorithm boxes** (for the optional Code section, pseudocode form): use a
  fenced markdown block tagged `pseudo`. The template defines a `pseudo`
  language for `listings` so these get a numbered, framed box without
  pulling in `algorithm.sty` (which isn't installed in many sandboxes).
  ````
  ```pseudo
  Require: previous-task params θ*, Fisher F, learning rate η, weight λ
  for batch B in new task:
      g ← ∇_θ L_new(θ; B) + λ F ⊙ (θ - θ*)
      θ ← θ - η g
  ```
  ````
  Use Unicode arrows (←, →) and dot operators (⊙, ⊗) directly — the template
  uses xelatex with `unicode-math`, so they render natively.
- **Code blocks** (for the runnable Python sketch): triple backticks with a
  language tag. The template's default `listings` style is configured for
  Python with line numbers and a frame.
- **Citations** — keep them light. Inline parenthetical refs like
  "(Kirkpatrick et al., PNAS 2017)" are enough. No bibtex setup needed.

## When pandoc fails

Most failures are one of:

1. **Missing LaTeX package.** `xelatex` errors mention the missing `.sty`.
   Install with `tlmgr install <name>` or just rewrite the offending macro
   in plain LaTeX.
2. **Stray smart-quote in math.** Pandoc converts `'` and `"` aggressively;
   inside `$...$` this kills the render. Fix: keep math single-line so the
   conversion doesn't fire mid-expression, or use raw LaTeX `\(...\)`.
3. **Unbalanced `$`.** A single `$` somewhere in prose (e.g. "$5") will
   kick off math mode. Escape as `\$`.
4. **Wide equation overflow.** Use `\begin{aligned}` inside `$$...$$` to
   break long derivations across lines, or `\begin{multline}`.

## Extending the template

`assets/study.tex` is a single-file pandoc template. To add things (a header
image, page numbering style, custom theorem environments), edit it directly.
Keep the existing variable interpolation (`$title$`, `$body$`, etc.) intact —
those are pandoc placeholders, not LaTeX errors.

## Output location

Save explained papers to:
`/sessions/quirky-confident-mccarthy/mnt/WorkingDir/explained_papers/<arxiv-id-or-slug>.pdf`

Use the arXiv ID when there is one (e.g. `2204.04411.pdf`); fall back to a
short kebab-case slug from the paper title (e.g. `attention-is-all-you-need.pdf`).
Create the `explained_papers/` directory if it doesn't exist — the build
script does this for you when you pass an output path inside it.

## Sanity check before linking

After the script writes the PDF:

1. `ls -la <output>.pdf` — confirm the file exists and is non-trivially
   sized (a real explainer is 100 KB minimum; 8 KB usually means LaTeX bailed
   silently).
2. `pdftotext <output>.pdf - | head -40` — confirm the front matter actually
   reads as the paper you explained, not garbage.

Only then return the `computer://` link to the user.

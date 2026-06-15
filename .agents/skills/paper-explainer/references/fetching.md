# Fetching and parsing the paper

Read this the first time you need to resolve an arXiv input or extract text
from an uploaded PDF. The rules aren't obvious and several patterns silently
fail.

## arXiv canonicalisation

Users paste arXiv links in many shapes. Canonicalise to the abstract page
first, then derive the PDF URL — that way you can also pull the title,
authors, and abstract metadata without re-parsing the PDF.

| Input the user pastes | Canonical abstract URL | Canonical PDF URL |
|---|---|---|
| `2204.04411` | `https://arxiv.org/abs/2204.04411` | `https://arxiv.org/pdf/2204.04411.pdf` |
| `arXiv:2204.04411v2` | `https://arxiv.org/abs/2204.04411v2` | `https://arxiv.org/pdf/2204.04411v2.pdf` |
| `https://arxiv.org/abs/2204.04411` | (already canonical) | swap `/abs/` → `/pdf/` and append `.pdf` |
| `https://arxiv.org/pdf/2204.04411.pdf` | swap `/pdf/` → `/abs/`, drop `.pdf` | (already canonical) |
| `https://arxiv.org/pdf/2204.04411v2` | as above | (already canonical, no `.pdf` needed) |
| Old-style `cs/0701001` | `https://arxiv.org/abs/cs/0701001` | `https://arxiv.org/pdf/cs/0701001.pdf` |

Use a regex like `r"(?:arXiv:)?(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})(v\d+)?"` to
pull the ID out of free text — users often paste the ID inside a sentence.

## Fetching the PDF

Prefer `WebFetch` for the abstract page (clean HTML, easy to extract title
and abstract) and a direct download for the PDF itself. The PDF is usually
~1–10 MB and arXiv mirrors handle it fine.

```bash
# Inside the sandbox shell:
curl -sSL -o /tmp/paper.pdf "https://arxiv.org/pdf/2204.04411.pdf"
```

If the download is blocked, ask the user to upload the PDF directly. Don't
silently skip the paper.

## Reading the PDF

Two layers of fallback, because no single extractor works on every paper.

**Layer 1 — `pdftotext` (poppler).** Fast, preserves reading order on
single-column papers, struggles on two-column or heavily-equation-laden
layouts.

```bash
pdftotext -layout /tmp/paper.pdf /tmp/paper.txt
```

**Layer 2 — `pdfplumber` (Python).** When `pdftotext` produces garbage —
typically interleaved columns or scrambled equations — fall back to
`pdfplumber`, which respects bounding boxes:

```python
import pdfplumber
with pdfplumber.open("/tmp/paper.pdf") as doc:
    pages = [page.extract_text(layout=True) for page in doc.pages]
text = "\n\n".join(p for p in pages if p)
```

For two-column papers, also try `--layout` plus a column-split heuristic, or
just read the paper's source `.tex` from arXiv (`https://arxiv.org/e-print/<id>`
returns a tar of the LaTeX source; that's the gold standard for math-heavy
papers).

**Layer 3 — LaTeX source.** When math fidelity matters most:

```bash
curl -sSL -o /tmp/source.tar "https://arxiv.org/e-print/2204.04411"
mkdir -p /tmp/src && tar -xf /tmp/source.tar -C /tmp/src
# Find the main .tex file (usually the one with \documentclass)
grep -l "documentclass" /tmp/src/*.tex
```

This is invaluable for the Math section — you can copy the paper's actual
equations rather than transcribing them from a PDF render.

## What to extract

Before drafting, pull these into a scratch note:

1. **Title, authors, venue, year, arXiv ID** — for the title block.
2. **Abstract** — to anchor your "what this paper is about" two-liner.
3. **The central equation(s)** — copy them verbatim from the source if you
   have it. Get the LaTeX exact; transcription errors poison the Math section.
4. **The algorithm box(es), if any** — they map directly into the optional
   Code section.
5. **Author's GitHub link, if mentioned** — usually in the abstract or
   footer. Useful for the "Author's repo" Code option.

## When the paper has supplementary material

Skim the appendix for:
- Full proofs of theorems whose statement appears in the body.
- Hyperparameter tables (rarely worth reproducing, but flag if surprising).
- Additional ablations that change how you'd describe the central claim.

Don't drag every appendix proof into the explainer. Pick the one that's most
load-bearing for the central claim.

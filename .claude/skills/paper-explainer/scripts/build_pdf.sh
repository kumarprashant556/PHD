#!/usr/bin/env bash
#
# build_pdf.sh — render a paper-explainer markdown file to a study-quality PDF.
#
# Usage:
#   scripts/build_pdf.sh INPUT.md OUTPUT.pdf
#
# The output directory is created if it doesn't exist. xelatex is used so
# that math fonts and unicode behave; pandoc handles the markdown → LaTeX
# conversion. The template at assets/study.tex sets up sectioning, math,
# algorithm boxes, and listings.

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 INPUT.md OUTPUT.pdf" >&2
  exit 64
fi

INPUT="$1"
OUTPUT="$2"

if [ ! -f "$INPUT" ]; then
  echo "Input markdown not found: $INPUT" >&2
  exit 66
fi

# Resolve script directory so the template path works regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/../assets/study.tex"

if [ ! -f "$TEMPLATE" ]; then
  echo "Template not found: $TEMPLATE" >&2
  exit 66
fi

# Make sure the output directory exists.
mkdir -p "$(dirname "$OUTPUT")"

# Run pandoc. We accept both `$...$` and `\(...\)` for inline math, and `$$...$$`
# for display math. raw_tex lets us drop into LaTeX for algorithm boxes and
# numbered equations.
pandoc "$INPUT" \
  --from "markdown+tex_math_dollars+tex_math_single_backslash+raw_tex+pipe_tables" \
  --to pdf \
  --pdf-engine=xelatex \
  --template="$TEMPLATE" \
  --toc --toc-depth=2 \
  --number-sections \
  --variable=geometry:"margin=1in" \
  --variable=fontsize:11pt \
  --variable=linkcolor:"NavyBlue" \
  --variable=urlcolor:"NavyBlue" \
  -o "$OUTPUT"

# Sanity check: refuse to claim success if the file is suspiciously small.
size_bytes="$(stat -c%s "$OUTPUT" 2>/dev/null || stat -f%z "$OUTPUT")"
if [ "$size_bytes" -lt 8000 ]; then
  echo "Warning: $OUTPUT is only ${size_bytes} bytes — pandoc likely produced an empty doc." >&2
  exit 70
fi

echo "Wrote $OUTPUT (${size_bytes} bytes)."

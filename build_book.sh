#!/usr/bin/env bash
#
# build_book.sh
#
# Builds Machine_Learning.pdf (the colored-box book) from the top-level
# Markdown lecture notes.
#
# 1. Runs generate_figures.py to execute code blocks and render their
#    figures into fig/generated/ (left in place afterwards -- book_convert.py
#    references them from there). By default this reuses whatever is
#    already in fig/generated/ instead of re-running every code block (so a
#    normal build does not retrain the MNIST example, etc.); pass --regen
#    to force every figure to be regenerated from scratch.
# 2. Runs book_convert.py to turn each chapter into a LaTeX subfile inside a
#    scratch build directory (alongside copies of the book's static LaTeX
#    files: main.tex, amd.sty, theorems.tex, syntax_highlighting.tex,
#    title.tex, preface.tex, contents.tex, and the standalone TikZ diagrams
#    under fig/*.tex that some chapters \input). book_convert.py also
#    regenerates References.bib (at the repo root and in the build dir)
#    from the "<!-- cite: ... -->" comments in the chapters.
# 3. Restores the original Markdown files (generate_figures.py rewrites
#    their image links in place).
# 4. Compiles main.tex with xelatex, re-running (up to 4 passes total) until
#    it stops asking for a rerun, so cross-references, the ToC/LOF, and
#    citation numbering settle, then copies the resulting PDF to the repo
#    root and deletes the scratch build directory -- only the final PDF and
#    fig/generated/ survive the build.
#
# Usage:
#   ./build_book.sh            # reuse existing figures in fig/generated/
#   ./build_book.sh --regen    # re-run every code block and regenerate all figures

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

FORCE_FIGURES=()
if [ "${1:-}" = "--regen" ]; then
  FORCE_FIGURES=(--force)
fi

# Chapter sources are the numbered top-level notes (00 Notation, 01 ..., 10 ...).
# Anything else at the root (todo.md, external-review.md, ...) is internal and
# must not enter the book. Sorted with a leading zero so 05b follows 05a/05.
CHAPTERS=()
for f in [0-9]*.md; do
  [ -e "$f" ] || continue
  CHAPTERS+=("$f")
done

OUT_PDF="$SCRIPT_DIR/Machine_Learning.pdf"
STATIC_FILES=(main.tex amd.sty theorems.tex syntax_highlighting.tex title.tex preface.tex contents.tex)

command -v pandoc  >/dev/null 2>&1 || { echo "pandoc is required" >&2; exit 1; }
command -v xelatex >/dev/null 2>&1 || { echo "xelatex (texlive-xetex) is required" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }

BUILD_DIR="$(mktemp -d)"
MD_BACKUP_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR" "$MD_BACKUP_DIR"' EXIT

if [ "${#CHAPTERS[@]}" -eq 0 ]; then
  echo "Error: no chapter notes ([0-9]*.md) found in $SCRIPT_DIR" >&2
  exit 1
fi

for chapter in "${CHAPTERS[@]}"; do
  cp "$chapter" "$MD_BACKUP_DIR/$chapter"
done

echo "=== Step 1: Generating Figures from Code Blocks ==="
if [ -f "generate_figures.py" ]; then
  if [ "${#FORCE_FIGURES[@]}" -gt 0 ]; then
    python3 generate_figures.py --force
  else
    python3 generate_figures.py
  fi
else
  echo "WARNING: generate_figures.py not found! Skipping figure generation." >&2
fi
echo

echo "=== Step 2: Converting Chapters to LaTeX ==="
for f in "${STATIC_FILES[@]}"; do
  cp "$f" "$BUILD_DIR/"
done
if compgen -G "fig/*.tex" > /dev/null; then
  mkdir -p "$BUILD_DIR/fig"
  cp fig/*.tex "$BUILD_DIR/fig/"
fi
python3 book_convert.py --out-dir "$BUILD_DIR"
echo

echo "=== Step 3: Restoring Original Markdown ==="
for chapter in "${CHAPTERS[@]}"; do
  cp "$MD_BACKUP_DIR/$chapter" "$chapter"
done
echo "  Restored original Markdown files."
echo

echo "=== Step 4: Compiling the Book (xelatex) ==="
cd "$BUILD_DIR"
xelatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/book_build_pass1.log 2>&1 || {
  echo "First xelatex pass FAILED — see /tmp/book_build_pass1.log" >&2
  tail -n 40 /tmp/book_build_pass1.log >&2
  exit 1
}

if command -v biber >/dev/null 2>&1 && [ -f main.bcf ]; then
  biber main > /tmp/book_build_biber.log 2>&1 || echo "biber reported warnings — see /tmp/book_build_biber.log" >&2
fi

# Keep re-running until cross-references/page numbers (TOC, LOF, \pageref,
# \citep) settle -- adding/moving content can take more than 2 passes to
# converge, and xelatex says so via this exact warning.
LAST_LOG="/tmp/book_build_pass1.log"
for pass in 2 3 4; do
  LAST_LOG="/tmp/book_build_pass${pass}.log"
  xelatex -interaction=nonstopmode -halt-on-error main.tex > "$LAST_LOG" 2>&1 || {
    echo "xelatex pass ${pass} FAILED — see $LAST_LOG" >&2
    tail -n 40 "$LAST_LOG" >&2
    exit 1
  }
  grep -q "Rerun to get cross-references right" "$LAST_LOG" || break
done

cp main.pdf "$OUT_PDF"

echo
echo "Done."
echo "  Book PDF: $OUT_PDF"

n_overfull=$(grep -c "Overfull \\\\hbox" "$LAST_LOG" || true)
if [ "${n_overfull:-0}" -gt 0 ]; then
  echo "  Note: $n_overfull overfull-hbox warning(s) remain (see $LAST_LOG) — check for margin overflow."
fi

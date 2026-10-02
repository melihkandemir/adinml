#!/usr/bin/env bash
#
# build_pdf.sh
#
# 1. Automatically discovers all Markdown files in version sort order.
# 2. Runs generate_figures.py to execute code blocks and render plots to disk.
# 3. Converts each chapter into a temporary PDF via Pandoc/XeLaTeX.
# 4. Concatenates all individual PDFs into pdf/Lecture_Notes_Full.pdf.
# 5. Restores original Markdown files, and deletes intermediate chapter PDFs 
#    and the fig/generated directory.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="$SCRIPT_DIR/pdf"
COMBINED_NAME="Lecture_Notes_Full.pdf"
COMBINED_PATH="$OUT_DIR/$COMBINED_NAME"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Dynamic Chapter Discovery (sort -V ensures 05a follows 05)
mapfile -t CHAPTERS < <(find . -maxdepth 1 -name "*.md" ! -name "todo.md" ! -name ".*" -printf "%f\n" | sort -V)

if [ "${#CHAPTERS[@]}" -eq 0 ]; then
  echo "Error: No Markdown files found in $SCRIPT_DIR" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Backup original Markdown files before generate_figures.py modifies them
# ---------------------------------------------------------------------------
MD_BACKUP_DIR="$TMP_DIR/md_backups"
mkdir -p "$MD_BACKUP_DIR"
for chapter in "${CHAPTERS[@]}"; do
  cp "$chapter" "$MD_BACKUP_DIR/$chapter"
done

# ---------------------------------------------------------------------------
# 1. Run Figure Generation
# ---------------------------------------------------------------------------
echo "=== Step 1: Generating Figures from Code Blocks ==="
if [ -f "generate_figures.py" ]; then
  python3 generate_figures.py --force
else
  echo "WARNING: generate_figures.py not found! Skipping figure generation." >&2
fi
echo

# ---------------------------------------------------------------------------
# 2. Dependency Checks
# ---------------------------------------------------------------------------
missing=()
command -v pandoc  >/dev/null 2>&1 || missing+=("pandoc")
command -v xelatex >/dev/null 2>&1 || missing+=("xelatex (texlive-xetex)")
command -v python3 >/dev/null 2>&1 || missing+=("python3")

if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing required tool(s): ${missing[*]}" >&2
  exit 1
fi

CONCAT_TOOL=""
if command -v pdfunite >/dev/null 2>&1; then
  CONCAT_TOOL="pdfunite"
elif command -v gs >/dev/null 2>&1; then
  CONCAT_TOOL="gs"
else
  echo "Missing PDF concatenation tool (poppler-utils or ghostscript)." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

FONT_ARGS=()
if fc-list 2>/dev/null | grep -qi "DejaVu Serif:"; then
  FONT_ARGS+=(-V "mainfont=DejaVu Serif")
fi
if fc-list 2>/dev/null | grep -qi "DejaVu Sans Mono:"; then
  FONT_ARGS+=(-V "monofont=DejaVu Sans Mono")
fi

# ---------------------------------------------------------------------------
# 3. Preprocessing for PDF
# ---------------------------------------------------------------------------
preprocess_for_pdf() {
  python3 - "$1" "$2" <<'PYEOF'
import re, sys, os

src, dst = sys.argv[1], sys.argv[2]
base_dir = os.path.dirname(os.path.abspath(src))
text = open(src, encoding="utf-8").read()

img_pattern = re.compile(r'!\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')

def fix_image(m):
    path = m.group(1)
    resolved = path if os.path.isabs(path) else os.path.join(base_dir, path)
    if os.path.isfile(resolved):
        return m.group(0)
    return f"*[figure omitted: `{path}` not found in source notes]*"

align_envs = r'align\*?|gather\*?|multline\*?|eqnarray\*?|alignat\*?'
unwrap_pattern = re.compile(
    r'\$\$[ \t]*\n([ \t]*(?:\\begin\{(?:' + align_envs + r')\}).*?(?:\\end\{(?:' + align_envs + r')\}))[ \t]*\n\$\$',
    re.S,
)

hr_pattern = re.compile(r'^[ \t]*-{3,}[ \t]*$')

lines = text.split("\n")
segments = []
in_code = False
current = []
for line in lines:
    if line.lstrip().startswith("```"):
        current.append(line)
        segments.append((in_code, current))
        current = []
        in_code = not in_code
        continue
    current.append(line)
segments.append((in_code, current))

out_parts = []
for is_code, seg_lines in segments:
    chunk = "\n".join(seg_lines)
    if not is_code:
        chunk_lines = [
            "***" if hr_pattern.match(l) else img_pattern.sub(fix_image, l)
            for l in chunk.split("\n")
        ]
        chunk = "\n".join(chunk_lines)
        chunk = unwrap_pattern.sub(lambda m: m.group(1), chunk)
    out_parts.append(chunk)

open(dst, "w", encoding="utf-8").write("\n".join(out_parts))
PYEOF
}

# ---------------------------------------------------------------------------
# 4. Convert Chapters to Individual PDFs
# ---------------------------------------------------------------------------
convert_chapter() {
  local src="$1"
  local base="${src%.md}"
  local cleaned="$TMP_DIR/${base}.md"
  local out_pdf="$OUT_DIR/${base}.pdf"
  local log="$TMP_DIR/${base}.log"
  local title="${base//_/ }"

  preprocess_for_pdf "$src" "$cleaned"

  echo "-> $src"
  if ! pandoc "$cleaned" \
      --pdf-engine=xelatex \
      --highlight-style=tango \
      -V geometry:margin=1in \
      -V fontsize=11pt \
      -V colorlinks=true \
      -V linkcolor=blue \
      -V urlcolor=blue \
      --metadata title="$title" \
      "${FONT_ARGS[@]}" \
      -o "$out_pdf" 2> "$log"; then
    echo "    FAILED — see $log" >&2
    tail -n 20 "$log" >&2 || true
    return 1
  fi
}

echo "=== Step 2: Converting ${#CHAPTERS[@]} Chapters to PDF ==="
ok=()
failed=()

for chapter in "${CHAPTERS[@]}"; do
  if convert_chapter "$chapter"; then
    ok+=("$OUT_DIR/${chapter%.md}.pdf")
  else
    failed+=("$chapter")
  fi
done

if [ "${#ok[@]}" -eq 0 ]; then
  # Restore Markdown files before exiting on error
  for chapter in "${CHAPTERS[@]}"; do
    cp "$MD_BACKUP_DIR/$chapter" "$chapter"
  done
  echo "No chapters converted successfully; aborting." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 5. Concatenate All PDFs
# ---------------------------------------------------------------------------
echo
echo "=== Step 3: Concatenating ${#ok[@]} Chapters ==="
case "$CONCAT_TOOL" in
  pdfunite)
    pdfunite "${ok[@]}" "$COMBINED_PATH"
    ;;
  gs)
    gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite \
       -sOutputFile="$COMBINED_PATH" "${ok[@]}"
    ;;
esac

# ---------------------------------------------------------------------------
# 6. Full Cleanup (Restores original .md files & deletes generated assets)
# ---------------------------------------------------------------------------
echo
echo "=== Step 4: Full Cleanup ==="

# 1. Restore original .md notes (clears modified links pointing to fig/generated)
for chapter in "${CHAPTERS[@]}"; do
  cp "$MD_BACKUP_DIR/$chapter" "$chapter"
done
echo "  Restored original Markdown files."

# 2. Remove intermediate chapter PDFs
for pdf_file in "${ok[@]}"; do
  rm -f "$pdf_file"
done
echo "  Removed individual chapter PDFs."

# 3. Force-remove generated figures folder and top-level fig directory if empty
rm -rf "$SCRIPT_DIR/fig/generated"
rmdir "$SCRIPT_DIR/fig" 2>/dev/null || true
echo "  Removed generated figures (fig/generated)."

echo
echo "Done."
echo "  Final Combined PDF: $COMBINED_PATH"

if [ "${#failed[@]}" -gt 0 ]; then
  echo
  echo "The following chapter(s) failed to convert and were excluded:" >&2
  printf '  - %s\n' "${failed[@]}" >&2
  exit 1
fi

#!/usr/bin/env python3
"""
generate_figures.py

Executes the Python code blocks embedded in the lecture-note chapters and
saves the matplotlib figures they produce, then rewrites each chapter's
"![png](output_x_y.png)"-style image links -- leftovers from an old
Jupyter-notebook export whose actual output images were never included in
the repository -- to point at the freshly generated files.

This fixes figure display both in Obsidian (the links currently point
nowhere) and in PDFs built by build_pdf.sh (which, until now, could only
strip these dangling links and print a placeholder note instead of the
actual plot).

Usage:
    python3 generate_figures.py            # process every chapter
    python3 generate_figures.py FILE.md    # process just one chapter
    python3 generate_figures.py --force    # regenerate even if already resolved

Design notes:
  - Each chapter's Python code blocks are executed in ONE shared namespace,
    in document order, exactly as a notebook's cells would be -- later
    blocks routinely depend on variables (models, data splits) defined in
    earlier ones.
  - Figures are captured right after each block finishes, by saving every
    currently open matplotlib figure and then closing it -- this mirrors
    what a notebook's own "plt.show()" display step would have produced,
    without needing to special-case each block's plotting calls. Each saved
    figure is tagged with the index of the block that produced it, so
    figures can be matched back to the "![png](...)" placeholder that
    followed that same block in the original document, in order.
  - A block whose only meaningful content is IPython's
    "display.Image('path/to/existing.png')" is treated as producing that
    existing file directly, rather than something to execute and capture.
  - A chapter is skipped entirely (fast) once none of its image links are
    still dangling, so re-running this script after a first successful
    pass is a no-op.
  - Execution happens in a subprocess with a generous timeout, so a slow or
    network-dependent block later in a chapter (e.g. downloading a dataset)
    cannot hang the whole run -- and cannot stop figures already produced
    by earlier blocks in the same chapter from being saved, since each
    block's figures are flushed to disk immediately after that block runs.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "fig", "generated")
TIMEOUT_SECONDS = 480

IMG_LINK_RE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
CODE_BLOCK_RE = re.compile(r'```python\n(.*?)```', re.S)
DISPLAY_IMAGE_RE = re.compile(r'display\.Image\(\s*["\']([^"\']+)["\']')

RUNNER_TEMPLATE = '''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json as _json

_outdir = {outdir!r}
_prefix = {prefix!r}
_manifest_path = {manifest!r}
_saved = []  # list of {{"block": i, "file": path}}
_counter = [0]

def _flush_figures(block_index):
    import os as _os
    for num in plt.get_fignums():
        fig = plt.figure(num)
        _counter[0] += 1
        fname = _os.path.join(_outdir, f"{{_prefix}}_{{_counter[0]}}.png")
        try:
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            _saved.append({{"block": block_index, "file": fname}})
        except Exception as e:
            print(f"[generate_figures] WARNING: could not save figure {{num}}: {{e}}",
                  file=__import__("sys").stderr)
    plt.close("all")
    with open(_manifest_path, "w") as f:
        _json.dump(_saved, f)
'''


def slugify(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return re.sub(r'[^A-Za-z0-9]+', '_', base).strip('_')


def find_unresolved_image_links(text: str, base_dir: str):
    """Ordered list of (full_match_text, alt_text, referenced_path) for
    image links that don't resolve to a real file, skipping fenced code
    blocks."""
    out = []
    in_code = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for m in IMG_LINK_RE.finditer(line):
            alt, path = m.group(1), m.group(2)
            resolved = path if os.path.isabs(path) else os.path.join(base_dir, path)
            if not os.path.isfile(resolved):
                out.append((m.group(0), alt, path))
    return out


def extract_code_blocks(text: str):
    return [m.group(1) for m in CODE_BLOCK_RE.finditer(text)]


def static_display_target(block: str, base_dir: str):
    """If this block's only relevant action is IPython.display.Image(...)
    on a file that already exists, return its resolved path."""
    m = DISPLAY_IMAGE_RE.search(block)
    if not m:
        return None
    candidate = m.group(1)
    resolved = candidate if os.path.isabs(candidate) else os.path.join(base_dir, candidate)
    return resolved if os.path.isfile(resolved) else None


def build_runner_script(blocks, outdir, prefix, manifest_path):
    parts = [RUNNER_TEMPLATE.format(outdir=outdir, prefix=prefix, manifest=manifest_path)]
    for i, block in enumerate(blocks):
        parts.append(block)
        parts.append(f"\n_flush_figures({i})\n")
    return "\n".join(parts)


def process_chapter(path, force=False):
    base_dir = os.path.dirname(os.path.abspath(path))
    text = open(path, encoding="utf-8").read()

    unresolved = find_unresolved_image_links(text, base_dir)
    if not unresolved and not force:
        return "skipped (nothing to fix)"

    blocks = extract_code_blocks(text)
    if not blocks:
        return "skipped (no python code blocks)"

    slug = slugify(os.path.basename(path))
    os.makedirs(FIG_DIR, exist_ok=True)

    # Blocks that are pure "display an existing file" calls produce that
    # file directly, without needing to be executed for figure purposes;
    # they are still included in the executed script for state continuity.
    static_by_block = {}
    for i, block in enumerate(blocks):
        target = static_display_target(block, base_dir)
        if target is not None:
            static_by_block[i] = target

    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = os.path.join(tmp, "manifest.json")
        runner_path = os.path.join(tmp, "runner.py")
        with open(runner_path, "w", encoding="utf-8") as f:
            f.write(build_runner_script(blocks, FIG_DIR, slug, manifest_path))

        try:
            result = subprocess.run(
                [sys.executable, runner_path],
                cwd=base_dir,
                timeout=TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            result = None
            timed_out = True

        produced = []
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                produced = json.load(f)

        if not produced and not static_by_block:
            if timed_out:
                return f"FAILED (timed out after {TIMEOUT_SECONDS}s, no figures captured)"
            stderr_tail = "\n".join((result.stderr or "").strip().splitlines()[-15:])
            return f"FAILED (exit {result.returncode}, no figures captured):\n{stderr_tail}"

    # Merge figures produced during execution with static display targets,
    # both keyed by the block that produced them, then walk blocks in
    # document order to get one flat, correctly ordered list of images.
    by_block = {}
    for item in produced:
        by_block.setdefault(item["block"], []).append(item["file"])
    for i, target in static_by_block.items():
        by_block.setdefault(i, []).append(target)

    slots = []
    for i in range(len(blocks)):
        slots.extend(by_block.get(i, []))

    if len(slots) != len(unresolved):
        return (f"WARNING: produced {len(slots)} image(s) but found "
                f"{len(unresolved)} unresolved link(s) in the text -- left "
                f"unchanged to avoid a wrong mapping. Generated files (if "
                f"any) remain under {FIG_DIR}.")

    new_text = text
    for (match_text, alt, old_path), new_file in zip(unresolved, slots):
        rel = os.path.relpath(new_file, base_dir)
        # Keep whatever caption the note already gives the figure; only a
        # placeholder alt text (empty, or literally the old filename) is
        # replaced with the newly generated file's name.
        new_alt = alt if alt and alt != os.path.basename(old_path) else os.path.basename(new_file)
        new_text = new_text.replace(match_text, f"![{new_alt}]({rel})", 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)

    return f"OK ({len(slots)} figure(s) generated and linked)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="specific chapter(s) to process; default: all")
    ap.add_argument("--force", action="store_true", help="regenerate even if already resolved")
    args = ap.parse_args()

    # Numbered chapter notes only (00, 01, ..., 10); internal working
    # documents (todo.md, external-review.md, ...) are not part of the book.
    targets = args.files or sorted(f for f in os.listdir(ROOT) if re.match(r"[0-9]", f) and f.endswith(".md"))

    for fn in targets:
        path = fn if os.path.isabs(fn) else os.path.join(ROOT, fn)
        if not os.path.isfile(path):
            print(f"-> {fn}: not found, skipping")
            continue
        print(f"-> {fn} ...", end=" ", flush=True)
        print(process_chapter(path, force=args.force))


if __name__ == "__main__":
    main()

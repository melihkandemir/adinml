#!/usr/bin/env python3
r"""
book_convert.py

Converts the top-level Markdown lecture-note chapters into LaTeX subfiles
for the book template (amd.sty / theorems.tex / main.tex at repo root), and
writes chapters.tex, the driver file \input by main.tex.

The Markdown sources are assumed to already be in the format this expects
(aligned instead of align* inside $$ blocks, no stray whitespace touching a
'$', every chapter starting at heading level 1, an explicit "---" marking
where a Definition/Theorem/etc. box should end) -- see book_convert.py's
sibling instructions for that one-time source cleanup. This script does not
re-check or fix formatting; it assumes the .md files are already correct
and just transcribes them.

Pipeline per chapter:
  1. Resolve image links (in fig/, written there by generate_figures.py)
     to absolute paths, so they render regardless of the working directory
     used to compile the book; drop dangling links with a note.
  2. Split the document into blank-line-delimited chunks, keeping fenced
     code blocks and $$ math blocks atomic.
  3. Recognize the notes' own "**Theorem 9.1 (Mercer).** ..." convention
     (Definition/Theorem/Lemma/Corollary/Proposition/Claim/Fact/
     Assumption/Example/Remark/Proof, 2 or 3 asterisks, optional number,
     optional parenthetical title) and replace each such block with a
     pandoc raw-LaTeX block invoking the matching amd.sty command
     (\defn, \thm, \lem, \cor, \prop, \clm, \fact, \asum, \exm, \rmkb) or,
     for proofs, a bare `proof` environment. LaTeX does its own numbering,
     so the notes' hand-assigned numbers are dropped; parenthetical titles
     are kept. A block ends at the next such marker, a heading, an
     explicit "---", or (for a Proof specifically) the first QED glyph.
  4. Run the whole, now-mixed markdown+raw-LaTeX document through a single
     pandoc call (with pygments syntax highlighting for code fences) to
     get the final LaTeX body, and wrap it in a subfiles preamble with
     \chapter{<title from filename>}.

Citations work the same way: a chapter cites a source as a plain Obsidian
hyperlink immediately (same line, no gap) followed by an HTML comment
carrying the BibTeX fields, invisible in Obsidian:
    [Bartlett & Mendelson (2002)](https://...)<!-- cite: bartlett2002rademacher
    | article | author={...}; title={...}; journal={...}; year={...};
    volume={...}; pages={...} -->
Obsidian renders this as a plain clickable link. The book instead replaces
the whole "[link](url)<!-- cite: ... -->" pair with a real natbib
\citep{key} against the generated References chapter (see
apply_inline_citations) -- so write the surrounding sentence as if
\citep{key} will render as "(Author, Year)" in Mondrian blue, not as the
link text. Separately, this script scans every chapter for these comments,
deduplicated by key, and (re)writes both References.bib (at the repo root
and in --out-dir) and the "References" chapter itself, formatted directly
with thebibliography (each \bibitem carries a hand-written natbib
"[Author(Year)]" label) so it needs no bibliography backend (biblatex/biber)
to render \citep{}/\citet{}.

Usage:
    python3 book_convert.py [--out-dir DIR] [FILE.md]
"""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EXCLUDE = set()
# Front-matter file rendered as an unnumbered chapter before Chapter 1,
# rather than as a numbered chapter in the main glob (see main()).
NOTATION_FILE = "00 Notation.md"
EXCLUDE.add(NOTATION_FILE)

ENV_KEYWORDS = (
    "Theorem", "Lemma", "Corollary", "Proposition", "Definition",
    "Claim", "Fact", "Assumption", "Example", "Remark", "Proof",
)

ENV_RE = re.compile(
    r"^(\*{2,3})\s*(" + "|".join(ENV_KEYWORDS) + r")\b(.*?)\1",
    re.S,
)

TITLE_ONLY_CMD = {
    "Theorem": "thm", "Lemma": "lem", "Definition": "defn",
    "Assumption": "asum", "Claim": "clm",
}
CONTENT_ONLY_CMD = {
    "Corollary": "cor", "Proposition": "prop", "Fact": "fact",
}

IMG_RE = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')

# Multi-panel figures (produced by plt.subplots with more than one axis,
# or by other explicitly multi-panel plotting code) keep the global default
# width (main.tex: \setkeys{Gin}{width=0.85\textwidth}); single-panel ones
# are rendered at half of that, via a pandoc width attribute (42.5%).
# Keys are figure stems under fig/generated/.
MULTI_PANEL_FIGURES = {
    "01_Basic_Concepts_4",   # 2x2 grid of polynomial fits
    "02_Linear_Predictors_3",  # 2x3 grid of Lp unit-ball contours
    "02_Linear_Predictors_5",  # side-by-side Lasso vs ridge weight bars
    "08_Neural_Networks_1",   # 2x3 activations + gradients
    "08_Neural_Networks_3",   # MNIST loss + accuracy side by side
    "09_Kernel_Methods_1",   # 2D data + 3D transform side by side
    "10_Ensemble_Methods_1",  # AdaBoost errors + margin side by side
}
SINGLE_PANEL_WIDTH = "42.5%"  # half of the 85% default in main.tex

CITE_RE = re.compile(r'<!--\s*cite:\s*([\w:-]+)\s*\|\s*(\w+)\s*\|\s*(.*?)-->', re.S)
CITE_FIELD_RE = re.compile(r'(\w+)\s*=\s*\{([^}]*)\}')
# A citation hyperlink immediately (same line, no gap) followed by its
# "<!-- cite: key | ... -->" comment: [text](url)<!-- cite: key | ... -->.
# Obsidian shows the plain clickable link; the book replaces the whole
# thing with a real \cite{key} against files/references.tex.
CITE_LINK_RE = re.compile(
    r'\[[^\]\n]+\]\([^\n]*?\)<!--\s*cite:\s*([\w:-]+)\s*\|[^\n]*?-->'
)
QED_RE = re.compile(r'(\$\\square\$|\$\\blacksquare\$|[\u220E\u25A0\u25AA\u25FC])')
HEADING_RE = re.compile(r'^(#{1,6})\s')
ABBREV_TIE_RE = re.compile(r'\b(e\.g\.|i\.e\.|cf\.|etc\.)~(?=\\\()')


def slugify(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')


def chapter_title_from_filename(stem: str) -> str:
    # "05a Statistical Learning Theory" -> "Statistical Learning Theory"
    title = re.sub(r'^\d+[a-zA-Z]?\s+', '', stem)
    return title.replace('_', ' ').strip()


def run_pandoc(markdown_text: str) -> str:
    result = subprocess.run(
        ["pandoc", "-f", "markdown+raw_attribute+lists_without_preceding_blankline",
         "-t", "latex", "--wrap=preserve", "--highlight-style=pygments"],
        input=markdown_text, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"pandoc failed:\n{result.stderr}")
    # Pandoc always ties "e.g."/"i.e." to what follows with a non-breaking
    # space; when that's a long inline formula, the whole unbreakable run
    # can overflow the page margin. A breakable space before math is safe.
    latex = ABBREV_TIE_RE.sub(lambda m: m.group(1) + " ", result.stdout)
    # Pandoc emits \def\LTcaptype{none} before caption-less longtables.
    # Together with the caption/subcaption packages (ltcaption) that \def
    # makes \begin{longtable} fail with "No counter 'none' defined", so
    # strip it: without a \caption the counter is never stepped anyway.
    latex = re.sub(r'\\def\\LTcaptype\{none\}[^\n]*\n', '', latex)
    return latex


def apply_inline_citations(text: str) -> str:
    """Replace each "[link](url)<!-- cite: key | ... -->" pair with a real
    natbib \\citep{key} (rendered "(Author, Year)") against the References
    chapter. The .md source keeps the plain hyperlink (and the comment)
    untouched -- this only rewrites the in-memory copy that gets fed to
    pandoc."""
    return CITE_LINK_RE.sub(lambda m: "`\\citep{%s}`{=latex}" % m.group(1), text)


def resolve_images(text: str, base_dir: Path) -> str:
    lines = text.split("\n")
    out = []
    in_code = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue

        def fix(m):
            alt, path = m.group(1), m.group(2)
            resolved = Path(path) if os.path.isabs(path) else (base_dir / path)
            if resolved.is_file():
                stem = resolved.stem
                width_attr = "" if stem in MULTI_PANEL_FIGURES else "{width=%s}" % SINGLE_PANEL_WIDTH
                return f"![{alt}]({resolved.resolve()}){width_attr}"
            return f"*[figure omitted: `{path}` not found under fig/]*"

        out.append(IMG_RE.sub(fix, line))
    return "\n".join(out)


def split_chunks(text: str):
    lines = text.split("\n")
    chunks = []
    buf = []
    in_code = False
    in_math = False
    for line in lines:
        stripped = line.strip()
        if not in_code and not in_math and stripped == "":
            if buf:
                chunks.append("\n".join(buf))
                buf = []
            continue
        buf.append(line)
        if stripped.startswith("```"):
            in_code = not in_code
        elif stripped == "$$":
            in_math = not in_math
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def is_boundary(chunk: str) -> bool:
    stripped = chunk.lstrip()
    first_line = stripped.split("\n", 1)[0].rstrip()
    return bool(ENV_RE.match(stripped)) or stripped.startswith("#") or first_line == "---"


def latex_escape_plain(s: str) -> str:
    if not s:
        return s
    return run_pandoc(s).strip()


def extract_title_and_qualifier(kw: str, middle: str):
    m = re.search(r'\(([^)]*)\)', middle)
    title = m.group(1).strip() if m else ""
    qualifier = ""
    if kw == "Proof":
        q = middle.strip()
        q = re.sub(r'^[\d.\sa-zA-Z]*?(?=\()', '', q) if '(' in q else q
        q = q.strip()
        q = re.sub(r'\.$', '', q).strip()
        if q.startswith('(') and q.endswith(')'):
            q = q[1:-1].strip()
        elif re.fullmatch(r'[\d.\sa-zA-Z:]*', q):
            q = ""
        qualifier = q
    return title, qualifier


def build_env_latex(kw: str, title: str, qualifier: str, content_raw: str) -> str:
    content_raw = content_raw.strip("\n")
    latex_content = run_pandoc(content_raw).strip() if content_raw.strip() else ""
    latex_title = latex_escape_plain(title)

    if kw in TITLE_ONLY_CMD:
        cmd = TITLE_ONLY_CMD[kw]
        return "\\%s{%s}{\n%s\n}" % (cmd, latex_title, latex_content)
    if kw in CONTENT_ONLY_CMD:
        cmd = CONTENT_ONLY_CMD[kw]
        body = ("\\textbf{%s.}\\quad %s" % (latex_title, latex_content)) if title else latex_content
        return "\\%s{\n%s\n}" % (cmd, body)
    if kw == "Example":
        return "\\exm{%s}{\n%s\n}" % (latex_title, latex_content)
    if kw == "Remark":
        return "\\rmkb{\n%s\n}" % latex_content
    if kw == "Proof":
        latex_qualifier = latex_escape_plain(qualifier)
        label = "Proof (%s)." % latex_qualifier if qualifier else "Proof."
        return "\\begin{proof}[\\noindent\\textbf{%s}]\n%s\n\\end{proof}" % (label, latex_content)
    raise AssertionError(kw)


def raw_latex_chunk(tex: str) -> str:
    return "```{=latex}\n" + tex + "\n```"


def convert_env_blocks(chunks):
    out = []
    i = 0
    n = len(chunks)
    while i < n:
        chunk = chunks[i]
        stripped = chunk.lstrip()
        if stripped.rstrip() == "---":
            # A bare "---" is only a structural marker for where a
            # Definition/Theorem/etc. box should end (see is_boundary);
            # it is not meant to render as a visible rule in the book.
            i += 1
            continue
        m = ENV_RE.match(stripped)
        if not m:
            out.append(chunk)
            i += 1
            continue

        kw = m.group(2)
        middle = m.group(3)
        rest_first = stripped[m.end():]
        title, qualifier = extract_title_and_qualifier(kw, middle)

        content_parts = []
        j = i + 1

        def consume(text):
            """For a Proof, stop right at the first QED glyph and push
            whatever follows it back as a separate, ordinary chunk."""
            if kw != "Proof":
                content_parts.append(text)
                return False
            qm = QED_RE.search(text)
            if not qm:
                content_parts.append(text)
                return False
            content_parts.append(text[:qm.start()])
            remainder = text[qm.end():]
            return True, remainder

        done = False
        r = consume(rest_first)
        if isinstance(r, tuple):
            done, remainder = r
            if remainder.strip():
                chunks.insert(j, remainder)
                n += 1
        while not done and j < n and not is_boundary(chunks[j]):
            r = consume(chunks[j])
            if isinstance(r, tuple):
                done, remainder = r
                if remainder.strip():
                    chunks[j] = remainder
                else:
                    j += 1
            else:
                j += 1
        content_raw = "\n\n".join(p for p in content_parts if p.strip())

        tex = build_env_latex(kw, title, qualifier, content_raw)
        out.append(raw_latex_chunk(tex))
        i = j
    return out


def collect_bib_entries(md_files):
    """Scan the chapters for '<!-- cite: key | type | field={value}; ... -->'
    comments (invisible in Obsidian, sitting right under the citation's
    visible hyperlink) and return an ordered {key: (type, {field: value})}
    dict."""
    entries = {}
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        for m in CITE_RE.finditer(text):
            key, entry_type, fields_raw = m.group(1), m.group(2), m.group(3)
            if key in entries:
                continue
            fields = dict(CITE_FIELD_RE.findall(fields_raw))
            entries[key] = (entry_type, fields)
    return entries


def write_references_bib(entries, out_path: Path):
    blocks = []
    for key, (entry_type, fields) in entries.items():
        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields.items())
        blocks.append("@%s{%s,\n%s\n}" % (entry_type, key, body))
    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return len(entries)


TEX_ESCAPE_RE = re.compile(r'([&%$#_{}])')


def tex_escape(s: str) -> str:
    return TEX_ESCAPE_RE.sub(r'\\\1', s)


def format_authors(author_field: str) -> str:
    names = [a.strip() for a in author_field.split(" and ") if a.strip()]
    names = [tex_escape(n) for n in names]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def short_author_year_label(author_field: str, year: str) -> str:
    """Build the natbib \\bibitem[<label>]{key} label, e.g. "Aronszajn(1950)"
    or "Boser et al.(1992)" -- what \\citet/\\citep display as the author
    part, since these entries are hand-written rather than run through
    BibTeX."""
    names = [a.strip() for a in author_field.split(" and ") if a.strip()]
    surnames = [tex_escape(n.split(",")[0].strip()) for n in names]
    if len(surnames) == 1:
        who = surnames[0]
    elif len(surnames) == 2:
        who = f"{surnames[0]} and {surnames[1]}"
    else:
        who = f"{surnames[0]} et al."
    return f"{who}({year})"


def format_bib_entry(entry_type: str, fields: dict) -> str:
    f = {k: tex_escape(v) for k, v in fields.items()}
    authors = format_authors(fields.get("author", fields.get("editor", "")))
    year = f.get("year", "n.d.")
    title = f.get("title", "")

    if entry_type == "book":
        tail = f"\\textit{{{title}}}. {f.get('publisher', '')}."
    elif entry_type == "inproceedings":
        pages = f", pp.~{f['pages']}" if "pages" in f else ""
        tail = f"{title}. In \\textit{{{f.get('booktitle', '')}}}{pages}."
    elif entry_type == "techreport":
        tail = f"{title}. Technical report, {f.get('institution', '')}."
    else:  # article
        vol = f.get("volume", "")
        num = f"({f['number']})" if "number" in f else ""
        pages = f", {f['pages']}" if "pages" in f else ""
        journal = f.get("journal", "")
        volinfo = f"{journal}, {vol}{num}{pages}" if vol else f"{journal}{pages}"
        tail = f"{title}. \\textit{{{volinfo}}}."

    return f"{authors} ({year}). {tail}"


def write_references_chapter(entries, out_path: Path):
    def sort_key(item):
        _, (_, fields) = item
        return fields.get("author", fields.get("editor", "")).lower()

    items = []
    for key, (entry_type, fields) in sorted(entries.items(), key=sort_key):
        author_field = fields.get("author", fields.get("editor", ""))
        year = fields.get("year", "n.d.")
        label = short_author_year_label(author_field, year)
        items.append(
            "\\bibitem[%s]{%s} %s" % (label, key, format_bib_entry(entry_type, fields))
        )

    body = "\n\n".join(items)
    out_path.write_text(
        "\\documentclass[main.tex]{subfiles}\n\n"
        "\\begin{document}\n"
        "\\renewcommand{\\bibname}{References}\n"
        "\\addcontentsline{toc}{chapter}{References}\n"
        "\\begin{thebibliography}{99}\n\n"
        f"{body}\n\n"
        "\\end{thebibliography}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )


def convert_file(md_path: Path, numbered: bool = True) -> str:
    text = md_path.read_text(encoding="utf-8")
    text = apply_inline_citations(text)
    text = resolve_images(text, ROOT)

    chunks = split_chunks(text)
    chunks = convert_env_blocks(chunks)

    doc_markdown = "\n\n".join(chunks)
    body = run_pandoc(doc_markdown)
    # Pandoc emits \includegraphics[width=...,height=\textheight] WITHOUT
    # keepaspectratio, which distorts figures (height is applied literally).
    # Drop the height key so the width alone governs, scaling the image
    # uniformly in both directions.
    body = re.sub(
        r'(\\includegraphics\[width=[^\]]*?),height=\\textheight(\])',
        r'\1\2',
        body,
    )

    title = chapter_title_from_filename(md_path.stem)
    latex_title = latex_escape_plain(title)

    if numbered:
        heading = "\\chapter{%s}\n\n" % latex_title
    else:
        heading = (
            "\\chapter*{%s}\n"
            "\\addcontentsline{toc}{chapter}{%s}\n\n" % (latex_title, latex_title)
        )

    return (
        "\\documentclass[main.tex]{subfiles}\n\n"
        "\\begin{document}\n"
        "%s"
        "%s\n"
        "\\end{document}\n" % (heading, body)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("md_file", nargs="?", help="Convert just this one chapter")
    parser.add_argument("--out-dir", default=str(ROOT), help="Where to write the .tex files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.md_file:
        md_files = [ROOT / args.md_file]
    else:
        # Numbered chapter notes only (00, 01, ..., 10); other root-level
        # .md files (todo.md, external-review.md, ...) are internal working
        # documents that must not enter the book.
        md_files = sorted(p for p in ROOT.glob("[0-9]*.md"))

    out_names = []
    for md_path in md_files:
        print(f"-> {md_path.name}")
        tex = convert_file(md_path)
        out_stem = slugify(md_path.stem)
        (out_dir / f"{out_stem}.tex").write_text(tex, encoding="utf-8")
        out_names.append(out_stem)

    if not args.md_file:
        notation_path = ROOT / NOTATION_FILE
        if notation_path.is_file():
            tex = convert_file(notation_path, numbered=False)
            (out_dir / "notation.tex").write_text(tex, encoding="utf-8")
            out_names.insert(0, "notation")
            print(f"Wrote {out_dir / 'notation.tex'}")

        entries = collect_bib_entries(md_files)

        n_refs = write_references_bib(entries, ROOT / "References.bib")
        if out_dir.resolve() != ROOT.resolve():
            shutil.copy(ROOT / "References.bib", out_dir / "References.bib")
        print(f"Wrote References.bib ({n_refs} entries).")

        if entries:
            write_references_chapter(entries, out_dir / "references.tex")
            out_names.append("references")
            print(f"Wrote {out_dir / 'references.tex'}")

        chapters_tex = "".join(f"\\subfile{{{name}}}\n" for name in out_names)
        (out_dir / "chapters.tex").write_text(chapters_tex, encoding="utf-8")
        print(f"Wrote {out_dir / 'chapters.tex'}")

    print(f"Converted {len(out_names)} chapter(s).")


if __name__ == "__main__":
    main()

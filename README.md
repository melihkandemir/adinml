# adinml — Machine Learning Lecture Notes

A book-length set of lecture notes on machine learning, written in Markdown
and compiled to PDF via LaTeX. Maintained by Melih Kandemir
(University of Southern Denmark).

## Contents

| Chapter | Topic |
| --- | --- |
| 00 | Notation |
| 01 | Basic Concepts |
| 02 | Linear Predictors |
| 03 | Classification |
| 04 | Probability Theory |
| 05a | Statistical Learning Theory |
| 05b | Complexity Measures and Generalization |
| 06 | Bayesian Learning |
| 07 | Decision Trees |
| 08 | Neural Networks |
| 09 | Kernel Methods |
| 10 | Ensemble Methods |

Also includes generated figures (`fig/`).

## Building the PDF

Requires a LaTeX distribution (e.g., TeX Live) and Python 3.

```bash
./build_book.sh        # converts Markdown chapters to LaTeX and builds the PDF
./build_pdf.sh         # builds the PDF from the existing LaTeX sources
python generate_figures.py   # regenerates figures
```

The main output is `Machine_Learning.pdf`.

## Writing and editing

The chapter sources are plain Markdown files, so they can be read and edited
with any editor. The repository also contains an
[Obsidian](https://obsidian.md) vault configuration (`.obsidian/`) for
convenient editing of the notes.

## License

The contents of this repository are licensed under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) License — see
[LICENSE](LICENSE). The software scripts are licensed under the
[MIT License](LICENSE-MIT).

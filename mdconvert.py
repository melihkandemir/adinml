import sys
from pathlib import Path
import nbformat
from nbconvert import MarkdownExporter


def convert_notebook(input_path: Path) -> Path:
    """Converts a .ipynb notebook to Markdown, preserving the input stem."""
    if input_path.suffix != ".ipynb":
        raise ValueError(
            f"Input file must have a .ipynb extension, got: {input_path.suffix}"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        notebook_node = nbformat.read(f, as_version=4)

    exporter = MarkdownExporter()
    body, _ = exporter.from_notebook_node(notebook_node)

    output_path = input_path.with_suffix(".md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <path_to_notebook.ipynb>")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    try:
        out_path = convert_notebook(input_file)
        print(f"Successfully converted: {input_file} -> {out_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)
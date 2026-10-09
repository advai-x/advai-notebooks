"""Validate or execute tutorials in a chosen kernel, optionally render HTML previews."""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Notebook paths relative to repository root")
    parser.add_argument("--execute", action="store_true", help="Run and save cell outputs")
    parser.add_argument("--kernel", default="python3", help="Installed kernel with task extras")
    parser.add_argument("--html-dir", help="Optional preview directory; requires nbconvert")
    args = parser.parse_args()
    for relative in args.paths:
        path = (ROOT / relative).resolve()
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        if args.execute:
            NotebookClient(
                notebook,
                timeout=180,
                kernel_name=args.kernel,
                allow_errors=False,
                resources={"metadata": {"path": str(ROOT)}},
            ).execute()
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
        if args.html_dir:
            from nbconvert import HTMLExporter

            destination = Path(args.html_dir).resolve() / path.parent.name
            destination.mkdir(parents=True, exist_ok=True)
            html, _ = HTMLExporter().from_notebook_node(notebook)
            (destination / (path.stem + ".html")).write_text(html, encoding="utf-8")
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        print(
            f"{relative}: {len(code)} code cells, "
            f"{sum(cell.execution_count is not None for cell in code)} executed"
        )


if __name__ == "__main__":
    main()

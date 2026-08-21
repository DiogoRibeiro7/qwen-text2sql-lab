#!/usr/bin/env python3
"""Compile every notebook code cell without executing expensive experiments."""

from __future__ import annotations

from pathlib import Path

import nbformat


def main() -> None:
    count = 0
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            try:
                compile(cell.source, f"{path}:cell-{index}", "exec")
            except SyntaxError as exc:
                raise SyntaxError(f"Invalid code in {path}, cell {index}: {exc}") from exc
            count += 1
    if count == 0:
        raise RuntimeError("No notebook code cells found")
    print(f"Compiled {count} notebook code cells")


if __name__ == "__main__":
    main()

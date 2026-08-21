#!/usr/bin/env python3
"""Statically validate notebooks without executing expensive experiments.

Two invariants are enforced:

1. Every code cell compiles, so a syntax error cannot reach ``main``.
2. No notebook carries committed outputs or execution counts. Stored outputs
   make diffs unreadable, can leak local paths or data, and let a stale result
   masquerade as a current one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

NOTEBOOK_DIR = Path("notebooks")


def main() -> int:
    compiled = 0
    problems: list[str] = []

    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue

            try:
                compile(cell.source, f"{path}:cell-{index}", "exec")
            except SyntaxError as exc:
                problems.append(f"{path}: cell {index} has invalid syntax: {exc}")
                continue

            if cell.get("outputs"):
                problems.append(f"{path}: cell {index} has committed outputs; strip them")
            if cell.get("execution_count") is not None:
                problems.append(f"{path}: cell {index} has an execution count; strip it")

            compiled += 1

    if compiled == 0:
        problems.append(f"No notebook code cells found under {NOTEBOOK_DIR}/")

    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        print(
            f"\n{len(problems)} notebook problem(s). "
            "Run `nbstripout notebooks/*.ipynb` to clear outputs.",
            file=sys.stderr,
        )
        return 1

    print(f"Validated {compiled} notebook code cells: all compile, none carry outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired-bootstrap comparison of two evaluated prediction files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from qwen_text2sql.evaluation.bootstrap import paired_bootstrap_binary
from qwen_text2sql.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    first = {str(row["example_id"]): bool(row["execution_match"]) for row in read_jsonl(args.first)}
    second = {str(row["example_id"]): bool(row["execution_match"]) for row in read_jsonl(args.second)}
    if first.keys() != second.keys():
        raise ValueError("Evaluation files must contain the same example IDs")
    ids = sorted(first)
    result = paired_bootstrap_binary(
        [first[i] for i in ids],
        [second[i] for i in ids],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Execute prediction files and write per-example plus aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_text2sql.evaluation.metrics import summarize
from qwen_text2sql.io import read_jsonl, write_jsonl
from qwen_text2sql.pipeline import evaluate_prediction_rows, prepared_from_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    args = parser.parse_args()

    examples = [prepared_from_mapping(row) for row in read_jsonl(args.data)]
    records = evaluate_prediction_rows(examples, read_jsonl(args.predictions))
    write_jsonl(args.records_output, (record.to_dict() for record in records))
    metrics = summarize(records)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate deterministic SQL predictions for a prepared dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_text2sql.config import load_config
from qwen_text2sql.inference.generator import generate_sql, load_inference_model
from qwen_text2sql.io import read_jsonl, write_jsonl
from qwen_text2sql.pipeline import apply_limit, prediction_row, prepared_from_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    model, tokenizer = load_inference_model(config.model.model_id, args.adapter)
    examples = apply_limit(
        [prepared_from_mapping(row) for row in read_jsonl(args.data)], args.limit
    )
    if not examples:
        raise ValueError("Prepared dataset is empty; nothing to generate")
    predictions = []
    for index, example in enumerate(examples, start=1):
        sql, latency_ms = generate_sql(model, tokenizer, example, config.model)
        predictions.append(
            prediction_row(
                example,
                sql,
                latency_ms,
                model_id=config.model.model_id,
                adapter_path=args.adapter,
            )
        )
        print(f"{index}/{len(examples)} {example.example_id}")
    write_jsonl(args.output, predictions)


if __name__ == "__main__":
    main()

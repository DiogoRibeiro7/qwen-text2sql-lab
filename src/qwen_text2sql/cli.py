"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_text2sql.config import load_config
from qwen_text2sql.data.bird import prepare_bird_rows
from qwen_text2sql.data.splits import split_by_database
from qwen_text2sql.evaluation.evaluator import evaluate_prediction
from qwen_text2sql.evaluation.metrics import summarize
from qwen_text2sql.io import read_jsonl, write_jsonl
from qwen_text2sql.pipeline import prepared_from_mapping
from qwen_text2sql.training.train import train_adapter


def _prepare(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    rows = [dict(row) for row in dataset]
    prepared = prepare_bird_rows(rows, args.db_root)
    write_jsonl(args.output, (row.to_dict() for row in prepared))
    print(f"Wrote {len(prepared)} examples to {args.output}")


def _split(args: argparse.Namespace) -> None:
    examples = [prepared_from_mapping(row) for row in read_jsonl(args.input)]
    train, validation = split_by_database(examples, args.validation_fraction, args.seed)
    write_jsonl(args.train_output, (row.to_dict() for row in train))
    write_jsonl(args.validation_output, (row.to_dict() for row in validation))
    print(f"train={len(train)} validation={len(validation)}")


def _train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    adapter = train_adapter(
        config,
        args.train_data,
        args.validation_data,
        max_train_examples=args.max_train_examples,
    )
    print(adapter)


def _evaluate(args: argparse.Namespace) -> None:
    examples_list = [prepared_from_mapping(row) for row in read_jsonl(args.data)]
    examples = {example.example_id: example for example in examples_list}
    prediction_rows = list(read_jsonl(args.predictions))
    records = []
    for row in prediction_rows:
        example_id = str(row["example_id"])
        if example_id not in examples:
            raise KeyError(f"Prediction has unknown example_id: {example_id}")
        prediction = str(row.get("prediction", row.get("predicted_sql", "")))
        records.append(evaluate_prediction(examples[example_id], prediction))
    write_jsonl(args.output, (record.to_dict() for record in records))
    print(json.dumps(summarize(records), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser."""
    parser = argparse.ArgumentParser(prog="qwen-text2sql")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-bird", help="Prepare BIRD records with SQLite schemas")
    prepare.add_argument("--dataset", default="birdsql/bird23-train-filtered")
    prepare.add_argument("--split", default="train")
    prepare.add_argument("--db-root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=_prepare)

    split = subparsers.add_parser("split", help="Create a database-disjoint train/validation split")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--train-output", type=Path, required=True)
    split.add_argument("--validation-output", type=Path, required=True)
    split.add_argument("--validation-fraction", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    split.set_defaults(func=_split)

    train = subparsers.add_parser("train", help="Run LoRA/QLoRA SFT")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--train-data", type=Path, required=True)
    train.add_argument("--validation-data", type=Path)
    train.add_argument("--max-train-examples", type=int)
    train.set_defaults(func=_train)

    evaluate = subparsers.add_parser("evaluate", help="Execute predictions and compute metrics")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.set_defaults(func=_evaluate)
    return parser


def main() -> None:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

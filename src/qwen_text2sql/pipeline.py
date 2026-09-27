"""Reusable experiment orchestration for prediction and evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from dataexcept import FileWriteError, wrapping

from qwen_text2sql.config import ModelConfig
from qwen_text2sql.evaluation.evaluator import evaluate_prediction
from qwen_text2sql.evaluation.metrics import summarize
from qwen_text2sql.inference.generator import generate_sql, load_inference_model
from qwen_text2sql.io import read_jsonl, write_jsonl
from qwen_text2sql.types import EvaluationRecord, PreparedExample


def prepared_from_mapping(row: dict[str, Any]) -> PreparedExample:
    """Create a prepared example while preserving backwards-compatible optional metadata."""
    required = ("example_id", "db_id", "question", "evidence", "gold_sql", "schema", "db_path")
    missing = [key for key in required if key not in row]
    if missing:
        raise KeyError(f"Prepared example is missing fields: {missing}")
    return PreparedExample(
        example_id=str(row["example_id"]),
        db_id=str(row["db_id"]),
        question=str(row["question"]),
        evidence=str(row["evidence"]),
        gold_sql=str(row["gold_sql"]),
        schema=str(row["schema"]),
        db_path=str(row["db_path"]),
        difficulty=str(row.get("difficulty", "")),
        source_id=str(row.get("source_id", "")),
    )


def prediction_row(
    example: PreparedExample,
    predicted_sql: str,
    generation_latency_ms: float,
    *,
    model_id: str,
    adapter_path: str | Path | None,
) -> dict[str, Any]:
    """Build one row of a predictions file.

    The single definition of that schema. It was previously written out twice —
    here and in ``scripts/generate_predictions.py`` — and the copies drifted:
    the script omitted ``difficulty``, so whether a prediction file could be
    stratified by difficulty depended on which entry point produced it.
    """
    return {
        "example_id": example.example_id,
        "db_id": example.db_id,
        "difficulty": example.difficulty,
        "prediction": predicted_sql,
        "generation_latency_ms": generation_latency_ms,
        "model_id": model_id,
        "adapter": str(adapter_path) if adapter_path is not None else None,
    }


def evaluate_prediction_rows(
    examples: Iterable[PreparedExample], prediction_rows: Iterable[Mapping[str, Any]]
) -> list[EvaluationRecord]:
    """Pair predictions with their examples and execute each one.

    Refuses a prediction whose ``example_id`` is not in the dataset. Scoring a
    mismatched pairing would corrupt every reported metric silently, and a bare
    ``KeyError`` from a dictionary lookup does not say which file is at fault.

    ``predicted_sql`` is accepted alongside ``prediction`` because older
    prediction dumps used the former.
    """
    by_id = {example.example_id: example for example in examples}
    records: list[EvaluationRecord] = []
    for row in prediction_rows:
        example_id = str(row["example_id"])
        if example_id not in by_id:
            raise KeyError(f"Prediction has unknown example_id: {example_id}")
        predicted = str(row.get("prediction", row.get("predicted_sql", "")))
        records.append(evaluate_prediction(by_id[example_id], predicted))
    if not records:
        raise ValueError("No predictions to evaluate")
    return records


def apply_limit(examples: list[PreparedExample], limit: int | None) -> list[PreparedExample]:
    """Truncate an evaluation set, refusing a limit that would silently mislead.

    A non-positive limit reaching a bare slice is quietly destructive: ``[:0]``
    evaluates nothing and ``[:-1]`` drops the last example, both without a word.
    """
    if limit is None:
        return examples
    if limit <= 0:
        raise ValueError("limit must be positive")
    return examples[:limit]


def generate_and_evaluate(
    *,
    model_config: ModelConfig,
    data_path: str | Path,
    predictions_path: str | Path,
    records_path: str | Path,
    metrics_path: str | Path,
    adapter_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, float | int | dict[str, int]]:
    """Generate SQL for a prepared dataset, execute it, and persist all evidence."""
    examples = apply_limit([prepared_from_mapping(row) for row in read_jsonl(data_path)], limit)
    if not examples:
        raise ValueError("Evaluation dataset is empty")

    model, tokenizer = load_inference_model(model_config.model_id, adapter_path)
    prediction_rows: list[dict[str, Any]] = []
    evaluation_rows = []
    for index, example in enumerate(examples, start=1):
        predicted_sql, generation_latency_ms = generate_sql(model, tokenizer, example, model_config)
        prediction_rows.append(
            prediction_row(
                example,
                predicted_sql,
                generation_latency_ms,
                model_id=model_config.model_id,
                adapter_path=adapter_path,
            )
        )
        evaluation_rows.append(evaluate_prediction(example, predicted_sql))
        print(f"{index}/{len(examples)} {example.example_id}")

    write_jsonl(predictions_path, prediction_rows)
    write_jsonl(records_path, (row.to_dict() for row in evaluation_rows))
    metrics = summarize(evaluation_rows)
    metrics_file = Path(metrics_path)
    with wrapping(OSError, FileWriteError, path=str(metrics_file.parent)):
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with wrapping((OSError, UnicodeError), FileWriteError, path=str(metrics_file)):
        metrics_file.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return metrics

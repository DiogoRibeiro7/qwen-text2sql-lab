"""Reusable experiment orchestration for prediction and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_text2sql.config import ModelConfig
from qwen_text2sql.evaluation.evaluator import evaluate_prediction
from qwen_text2sql.evaluation.metrics import summarize
from qwen_text2sql.inference.generator import generate_sql, load_inference_model
from qwen_text2sql.io import read_jsonl, write_jsonl
from qwen_text2sql.types import PreparedExample


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
    examples = [prepared_from_mapping(row) for row in read_jsonl(data_path)]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        examples = examples[:limit]
    if not examples:
        raise ValueError("Evaluation dataset is empty")

    model, tokenizer = load_inference_model(model_config.model_id, adapter_path)
    prediction_rows: list[dict[str, Any]] = []
    evaluation_rows = []
    for index, example in enumerate(examples, start=1):
        predicted_sql, generation_latency_ms = generate_sql(model, tokenizer, example, model_config)
        prediction_rows.append(
            {
                "example_id": example.example_id,
                "db_id": example.db_id,
                "difficulty": example.difficulty,
                "prediction": predicted_sql,
                "generation_latency_ms": generation_latency_ms,
                "model_id": model_config.model_id,
                "adapter": str(adapter_path) if adapter_path is not None else None,
            }
        )
        evaluation_rows.append(evaluate_prediction(example, predicted_sql))
        print(f"{index}/{len(examples)} {example.example_id}")

    write_jsonl(predictions_path, prediction_rows)
    write_jsonl(records_path, (row.to_dict() for row in evaluation_rows))
    metrics = summarize(evaluation_rows)
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics

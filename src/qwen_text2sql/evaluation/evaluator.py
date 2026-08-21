"""End-to-end per-example execution evaluation."""

from __future__ import annotations

from qwen_text2sql.evaluation.execution import execute_read_only, results_equivalent
from qwen_text2sql.sql.extract import extract_sql, normalize_sql
from qwen_text2sql.types import EvaluationRecord, PreparedExample


def evaluate_prediction(
    example: PreparedExample,
    raw_prediction: str,
    *,
    timeout_seconds: float = 10.0,
    tolerance: float = 1e-6,
) -> EvaluationRecord:
    """Evaluate a model response against the gold SQL by actual execution."""
    predicted_sql = extract_sql(raw_prediction)
    predicted_result = execute_read_only(
        example.db_path, predicted_sql, timeout_seconds=timeout_seconds
    )
    gold_result = execute_read_only(
        example.db_path, example.gold_sql, timeout_seconds=timeout_seconds
    )
    if not gold_result.ok:
        raise RuntimeError(
            f"Gold SQL failed for example {example.example_id}: {gold_result.error_message}"
        )
    return EvaluationRecord(
        example_id=example.example_id,
        db_id=example.db_id,
        gold_sql=example.gold_sql,
        predicted_sql=predicted_sql,
        valid_sql=predicted_result.ok,
        execution_match=results_equivalent(gold_result, predicted_result, tolerance=tolerance),
        exact_match=normalize_sql(example.gold_sql) == normalize_sql(predicted_sql),
        error_kind=predicted_result.error_kind,
        latency_ms=predicted_result.elapsed_ms,
    )

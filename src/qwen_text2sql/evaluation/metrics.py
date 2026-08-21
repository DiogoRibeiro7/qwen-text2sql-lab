"""Dataset-level Text-to-SQL metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from qwen_text2sql.types import EvaluationRecord


def summarize(records: Iterable[EvaluationRecord]) -> dict[str, float | int | dict[str, int]]:
    """Aggregate per-example records into reproducible scalar metrics."""
    rows = list(records)
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    count = len(rows)
    errors = Counter(row.error_kind for row in rows if row.error_kind != "none")
    return {
        "n": count,
        "valid_sql_rate": sum(row.valid_sql for row in rows) / count,
        "execution_accuracy": sum(row.execution_match for row in rows) / count,
        "exact_match": sum(row.exact_match for row in rows) / count,
        "mean_latency_ms": sum(row.latency_ms for row in rows) / count,
        "errors": dict(sorted(errors.items())),
    }

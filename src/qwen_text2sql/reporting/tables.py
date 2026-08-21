"""Tidy analysis frames built from per-example evaluation records.

Every function here takes records and returns a DataFrame. None of them print,
plot, or read configuration. That separation is what lets the notebooks stay
short and lets these analyses be unit tested, which is the part a notebook
cannot do for itself.

Accuracy is never returned as a bare point estimate: each row that reports a
proportion also carries a Wilson interval, because a per-database accuracy
computed over eleven examples and one computed over four hundred should not read
identically.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from qwen_text2sql.evaluation.bootstrap import BootstrapDifference, paired_bootstrap_binary
from qwen_text2sql.evaluation.intervals import wilson_interval
from qwen_text2sql.io import read_jsonl

__all__ = [
    "accuracy_by_database",
    "accuracy_by_difficulty",
    "compare_models",
    "error_breakdown",
    "headline_metrics",
    "load_records",
    "paired_comparison",
    "schema_statistics",
]

_RECORD_BOOL_COLUMNS = ("valid_sql", "execution_match", "exact_match")


def load_records(path: str | Path) -> pd.DataFrame:
    """Load evaluation records into a frame with stable dtypes.

    Reading the JSONL directly with ``pd.read_json`` infers dtypes per file, so
    a run in which every prediction failed would type ``execution_match`` as
    something other than bool and quietly break downstream comparisons.
    """
    frame = pd.DataFrame(list(read_jsonl(path)))
    if frame.empty:
        raise ValueError(f"No evaluation records in {path}")
    missing = [
        column for column in ("example_id", "db_id", *_RECORD_BOOL_COLUMNS) if column not in frame
    ]
    if missing:
        raise KeyError(f"Evaluation records are missing columns: {missing}")
    for column in _RECORD_BOOL_COLUMNS:
        frame[column] = frame[column].astype(bool)
    if "error_kind" in frame:
        frame["error_kind"] = frame["error_kind"].fillna("none").astype(str)
    return frame


def headline_metrics(records: pd.DataFrame, *, confidence: float = 0.95) -> pd.DataFrame:
    """Summarise a run's primary and secondary endpoints, each with an interval."""
    rows: list[dict[str, Any]] = []
    for column, label in (
        ("execution_match", "Execution accuracy (primary)"),
        ("valid_sql", "Valid SQL rate"),
        ("exact_match", "Exact match (diagnostic)"),
    ):
        interval = wilson_interval(records[column].tolist(), confidence=confidence)
        rows.append(
            {
                "metric": label,
                "estimate": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
                "successes": interval.successes,
                "n": interval.total,
            }
        )
    if "latency_ms" in records:
        rows.append(
            {
                "metric": "Median latency (ms)",
                "estimate": float(records["latency_ms"].median()),
                "lower": float(records["latency_ms"].quantile(0.25)),
                "upper": float(records["latency_ms"].quantile(0.75)),
                "successes": pd.NA,
                "n": len(records),
            }
        )
    return pd.DataFrame(rows)


def error_breakdown(records: pd.DataFrame) -> pd.DataFrame:
    """Count failures by category, including successes, as shares of the whole.

    Failure counts are reported against the full evaluation size rather than
    against the failures alone, so a category cannot look dominant merely
    because the model failed rarely.
    """
    if "error_kind" not in records:
        raise KeyError("Evaluation records have no error_kind column")
    counts = records["error_kind"].value_counts(dropna=False)
    frame = counts.rename_axis("error_kind").reset_index(name="count")
    frame["share"] = frame["count"] / len(records)
    frame["is_failure"] = frame["error_kind"] != "none"
    return frame.sort_values(["is_failure", "count"], ascending=[True, False]).reset_index(
        drop=True
    )


def accuracy_by_database(
    records: pd.DataFrame, *, confidence: float = 0.95, min_examples: int = 1
) -> pd.DataFrame:
    """Per-database execution accuracy with intervals, widest-interval last.

    Aggregate accuracy hides whether a model is uniformly mediocre or excellent
    on most schemas and helpless on a few. That distinction decides whether the
    next experiment is more data or better schema linking.
    """
    rows: list[dict[str, Any]] = []
    for db_id, group in records.groupby("db_id", sort=True):
        if len(group) < min_examples:
            continue
        interval = wilson_interval(group["execution_match"].tolist(), confidence=confidence)
        rows.append(
            {
                "db_id": db_id,
                "n": interval.total,
                "execution_accuracy": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
                "valid_sql_rate": float(group["valid_sql"].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("execution_accuracy", ascending=False).reset_index(drop=True)


def accuracy_by_difficulty(
    records: pd.DataFrame,
    difficulties: Mapping[str, str] | pd.Series | None = None,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Execution accuracy stratified by benchmark difficulty label.

    Difficulty is carried on the prepared examples rather than the evaluation
    records, so it is passed in as a mapping from example_id when needed.
    """
    frame = records
    if difficulties is not None:
        mapping = (
            dict(difficulties) if isinstance(difficulties, Mapping) else difficulties.to_dict()
        )
        frame = records.assign(difficulty=records["example_id"].map(mapping))
    if "difficulty" not in frame:
        raise KeyError("No difficulty column; pass a mapping from example_id to difficulty")

    rows: list[dict[str, Any]] = []
    for difficulty, group in frame.groupby(frame["difficulty"].fillna("unlabelled"), sort=True):
        interval = wilson_interval(group["execution_match"].tolist(), confidence=confidence)
        rows.append(
            {
                "difficulty": difficulty,
                "n": interval.total,
                "execution_accuracy": interval.estimate,
                "lower": interval.lower,
                "upper": interval.upper,
            }
        )
    return pd.DataFrame(rows)


def _align(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Inner-join two runs on example_id, refusing a silently partial overlap."""
    merged = first.merge(second, on="example_id", suffixes=("_first", "_second"), how="inner")
    if merged.empty:
        raise ValueError("The two runs share no example_id; they cannot be compared")
    if len(merged) != len(first) or len(merged) != len(second):
        raise ValueError(
            "The two runs were evaluated on different examples "
            f"({len(first)} vs {len(second)}, {len(merged)} shared). "
            "A paired comparison requires identical evaluation sets."
        )
    return merged


def paired_comparison(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapDifference:
    """Paired bootstrap for accuracy(first) - accuracy(second) on shared examples."""
    merged = _align(first, second)
    return paired_bootstrap_binary(
        merged["execution_match_first"].tolist(),
        merged["execution_match_second"].tolist(),
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=confidence,
    )


def compare_models(
    runs: Mapping[str, pd.DataFrame],
    *,
    baseline: str,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compare every run against one baseline on the same examples.

    Also reports how each comparison decomposes into examples the model fixed
    and examples it broke. A net gain of two points built from twenty fixes and
    eighteen regressions is a different result from one built from two fixes.
    """
    if baseline not in runs:
        raise KeyError(f"Baseline {baseline!r} is not among the runs: {sorted(runs)}")
    reference = runs[baseline]
    rows: list[dict[str, Any]] = []
    for name, frame in runs.items():
        if name == baseline:
            continue
        merged = _align(frame, reference)
        candidate = merged["execution_match_first"]
        control = merged["execution_match_second"]
        difference = paired_bootstrap_binary(
            candidate.tolist(),
            control.tolist(),
            n_bootstrap=n_bootstrap,
            seed=seed,
            confidence=confidence,
        )
        rows.append(
            {
                "model": name,
                "baseline": baseline,
                "n": len(merged),
                "accuracy": float(candidate.mean()),
                "baseline_accuracy": float(control.mean()),
                "difference": difference.observed_difference,
                "lower": difference.lower,
                "upper": difference.upper,
                "probability_positive": difference.probability_positive,
                "fixed": int((candidate & ~control).sum()),
                "broke": int((~candidate & control).sum()),
            }
        )
    return pd.DataFrame(rows)


def schema_statistics(examples: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Per-example input size statistics for prepared examples.

    Schema text dominates the prompt. If it routinely exceeds the configured
    context window, examples are being silently truncated and any accuracy drop
    on those databases is an artefact of the prompt, not of the model.
    """
    rows = [
        {
            "example_id": str(row["example_id"]),
            "db_id": str(row["db_id"]),
            "question_chars": len(str(row["question"])),
            "evidence_chars": len(str(row.get("evidence", ""))),
            "schema_chars": len(str(row["schema"])),
            "gold_sql_chars": len(str(row["gold_sql"])),
            "difficulty": str(row.get("difficulty", "")) or "unlabelled",
        }
        for row in examples
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No prepared examples supplied")
    frame["prompt_chars"] = (
        frame["question_chars"] + frame["evidence_chars"] + frame["schema_chars"]
    )
    return frame

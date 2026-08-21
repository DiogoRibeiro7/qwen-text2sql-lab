from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from qwen_text2sql.io import write_jsonl
from qwen_text2sql.reporting import tables


def _records(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _record(
    example_id: str,
    *,
    db_id: str = "shop",
    execution_match: bool = True,
    valid_sql: bool = True,
    exact_match: bool = False,
    error_kind: str = "none",
    latency_ms: float = 10.0,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "db_id": db_id,
        "gold_sql": "SELECT 1",
        "predicted_sql": "SELECT 1",
        "valid_sql": valid_sql,
        "execution_match": execution_match,
        "exact_match": exact_match,
        "error_kind": error_kind,
        "latency_ms": latency_ms,
    }


# --------------------------------------------------------------------------
# load_records
# --------------------------------------------------------------------------


def test_load_records_forces_stable_dtypes(tmp_path: Path) -> None:
    """A run where nothing succeeded must still type the flags as bool."""
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [_record("a", execution_match=False, valid_sql=False)])
    frame = tables.load_records(path)
    assert frame["execution_match"].dtype == bool
    assert frame["valid_sql"].dtype == bool


def test_load_records_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [])
    with pytest.raises(ValueError, match="No evaluation records"):
        tables.load_records(path)


def test_load_records_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"example_id": "a", "db_id": "shop"}])
    with pytest.raises(KeyError, match="missing columns"):
        tables.load_records(path)


# --------------------------------------------------------------------------
# headline_metrics
# --------------------------------------------------------------------------


def test_headline_metrics_reports_intervals_not_bare_estimates() -> None:
    records = _records([_record(str(i), execution_match=i < 8) for i in range(10)])
    frame = tables.headline_metrics(records)
    primary = frame.loc[frame["metric"].str.startswith("Execution accuracy")].iloc[0]
    assert primary["estimate"] == pytest.approx(0.8)
    assert primary["lower"] < 0.8 < primary["upper"]
    assert primary["n"] == 10


def test_headline_metrics_includes_latency_when_present() -> None:
    records = _records([_record(str(i), latency_ms=float(i)) for i in range(5)])
    frame = tables.headline_metrics(records)
    assert "Median latency (ms)" in set(frame["metric"])


# --------------------------------------------------------------------------
# error_breakdown
# --------------------------------------------------------------------------


def test_error_breakdown_shares_are_of_the_whole_evaluation() -> None:
    """A category must not look dominant merely because failures are rare."""
    records = _records(
        [_record(str(i)) for i in range(90)]
        + [_record(f"e{i}", execution_match=False, error_kind="syntax_error") for i in range(10)]
    )
    frame = tables.error_breakdown(records)
    syntax = frame.loc[frame["error_kind"] == "syntax_error"].iloc[0]
    assert syntax["count"] == 10
    assert syntax["share"] == pytest.approx(0.10)
    assert frame["share"].sum() == pytest.approx(1.0)


def test_error_breakdown_marks_success_as_not_a_failure() -> None:
    records = _records([_record("a"), _record("b", error_kind="timeout")])
    frame = tables.error_breakdown(records)
    assert not bool(frame.loc[frame["error_kind"] == "none", "is_failure"].iloc[0])
    assert bool(frame.loc[frame["error_kind"] == "timeout", "is_failure"].iloc[0])


# --------------------------------------------------------------------------
# accuracy_by_database
# --------------------------------------------------------------------------


def test_accuracy_by_database_widens_the_interval_for_small_strata() -> None:
    """A database with three examples is barely measured; the table must say so."""
    records = _records(
        [_record(f"big{i}", db_id="big", execution_match=i < 40) for i in range(50)]
        + [_record(f"tiny{i}", db_id="tiny", execution_match=i < 2) for i in range(3)]
    )
    frame = tables.accuracy_by_database(records).set_index("db_id")
    assert frame.loc["big", "n"] == 50
    assert frame.loc["tiny", "n"] == 3
    big_width = frame.loc["big", "upper"] - frame.loc["big", "lower"]
    tiny_width = frame.loc["tiny", "upper"] - frame.loc["tiny", "lower"]
    assert tiny_width > big_width


def test_accuracy_by_database_can_drop_tiny_strata() -> None:
    records = _records(
        [_record(f"big{i}", db_id="big") for i in range(10)] + [_record("t", db_id="tiny")]
    )
    frame = tables.accuracy_by_database(records, min_examples=5)
    assert set(frame["db_id"]) == {"big"}


# --------------------------------------------------------------------------
# accuracy_by_difficulty
# --------------------------------------------------------------------------


def test_accuracy_by_difficulty_maps_labels_from_prepared_examples() -> None:
    records = _records([_record("a", execution_match=True), _record("b", execution_match=False)])
    frame = tables.accuracy_by_difficulty(records, {"a": "simple", "b": "challenging"})
    assert set(frame["difficulty"]) == {"simple", "challenging"}
    assert frame.set_index("difficulty").loc["simple", "execution_accuracy"] == 1.0


def test_accuracy_by_difficulty_labels_unmapped_examples() -> None:
    records = _records([_record("a"), _record("b")])
    frame = tables.accuracy_by_difficulty(records, {"a": "simple"})
    assert "unlabelled" in set(frame["difficulty"])


# --------------------------------------------------------------------------
# Paired comparison
# --------------------------------------------------------------------------


def test_paired_comparison_detects_a_real_improvement() -> None:
    better = _records([_record(str(i), execution_match=i < 70) for i in range(100)])
    worse = _records([_record(str(i), execution_match=i < 40) for i in range(100)])
    difference = tables.paired_comparison(better, worse, n_bootstrap=2000)
    assert difference.observed_difference == pytest.approx(0.30)
    assert difference.lower > 0
    assert difference.probability_positive > 0.99


def test_paired_comparison_refuses_mismatched_evaluation_sets() -> None:
    """Comparing runs scored on different examples would silently invent a result."""
    first = _records([_record(str(i)) for i in range(10)])
    second = _records([_record(str(i)) for i in range(8)])
    with pytest.raises(ValueError, match="different examples"):
        tables.paired_comparison(first, second)


def test_paired_comparison_refuses_disjoint_runs() -> None:
    first = _records([_record(f"a{i}") for i in range(5)])
    second = _records([_record(f"b{i}") for i in range(5)])
    with pytest.raises(ValueError, match="no example_id"):
        tables.paired_comparison(first, second)


def test_compare_models_decomposes_into_fixes_and_regressions() -> None:
    """A net gain built from many fixes and many breaks is not the same result."""
    baseline = _records([_record(str(i), execution_match=i % 2 == 0) for i in range(10)])
    candidate = _records([_record(str(i), execution_match=i < 7) for i in range(10)])
    frame = tables.compare_models({"lora": candidate, "base": baseline}, baseline="base")
    row = frame.iloc[0]
    assert row["model"] == "lora"
    assert row["n"] == 10
    assert row["fixed"] + row["broke"] > 0
    assert row["difference"] == pytest.approx(row["accuracy"] - row["baseline_accuracy"])


def test_compare_models_requires_a_known_baseline() -> None:
    records = _records([_record("a")])
    with pytest.raises(KeyError, match="not among the runs"):
        tables.compare_models({"lora": records}, baseline="absent")


# --------------------------------------------------------------------------
# schema_statistics
# --------------------------------------------------------------------------


def test_schema_statistics_sums_the_prompt_budget() -> None:
    """Schema text dominates the prompt; truncation would silently skew results."""
    frame = tables.schema_statistics(
        [
            {
                "example_id": "a",
                "db_id": "shop",
                "question": "x" * 10,
                "evidence": "y" * 5,
                "gold_sql": "SELECT 1",
                "schema": "z" * 100,
                "difficulty": "simple",
            }
        ]
    )
    row = frame.iloc[0]
    assert row["prompt_chars"] == 115
    assert row["schema_chars"] == 100


def test_schema_statistics_rejects_no_examples() -> None:
    with pytest.raises(ValueError, match="No prepared examples"):
        tables.schema_statistics([])

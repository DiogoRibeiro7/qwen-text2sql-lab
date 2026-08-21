"""The guards that refuse malformed input across the package.

Each of these is a `raise` that exists so a mistake surfaces immediately instead
of propagating into a reported number. They were the least-covered code in the
package, which is the wrong way round: a guard nobody tests is a guard nobody
knows still works.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from qwen_text2sql.config import load_config
from qwen_text2sql.data.schema import quote_identifier, sqlite_schema_text
from qwen_text2sql.data.splits import split_by_database
from qwen_text2sql.evaluation.bootstrap import paired_bootstrap_binary
from qwen_text2sql.evaluation.evaluator import evaluate_prediction
from qwen_text2sql.evaluation.metrics import summarize
from qwen_text2sql.experiments import learning_curve_plan, rank_ablation_plan
from qwen_text2sql.io import read_jsonl
from qwen_text2sql.sql.extract import extract_sql
from qwen_text2sql.types import PreparedExample

BASE_CONFIG = """
model:
  model_id: Qwen/Qwen3.5-4B
lora:
  rank: 16
  alpha: 32
training:
  output_dir: results/checkpoints/demo
"""


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _example(db_path: Path, gold_sql: str) -> PreparedExample:
    return PreparedExample(
        example_id="a",
        db_id="shop",
        question="Who?",
        evidence="",
        gold_sql=gold_sql,
        schema="",
        db_path=str(db_path),
    )


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "match"),
    [
        pytest.param(
            BASE_CONFIG.replace("model_id: Qwen/Qwen3.5-4B", "model_id: X\n  max_seq_length: 0"),
            "max_seq_length",
            id="zero-context",
        ),
        pytest.param(BASE_CONFIG.replace("rank: 16", "rank: 0"), "rank and alpha", id="zero-rank"),
        pytest.param(
            BASE_CONFIG.replace("alpha: 32", "alpha: -1"), "rank and alpha", id="negative-alpha"
        ),
        pytest.param(
            BASE_CONFIG.replace("  alpha: 32", "  alpha: 32\n  dropout: 1.0"),
            "dropout",
            id="dropout-one",
        ),
        pytest.param(
            BASE_CONFIG.replace(
                "  output_dir: results/checkpoints/demo", "  output_dir: d\n  learning_rate: 0"
            ),
            "learning_rate",
            id="zero-learning-rate",
        ),
    ],
)
def test_invalid_configuration_is_rejected(tmp_path: Path, body: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_config(_config(tmp_path, body))


def test_a_non_mapping_section_is_rejected(tmp_path: Path) -> None:
    """A YAML typo turning a section into a list must not load as defaults."""
    with pytest.raises(TypeError, match="must be a mapping"):
        load_config(_config(tmp_path, "model:\n  - Qwen/Qwen3.5-4B\n"))


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def _rows(databases: int, per_db: int = 2) -> list[PreparedExample]:
    return [
        PreparedExample(
            example_id=f"{db}-{i}",
            db_id=f"db{db}",
            question="Who?",
            evidence="",
            gold_sql="SELECT 1",
            schema="",
            db_path="x.sqlite",
        )
        for db in range(databases)
        for i in range(per_db)
    ]


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_rejects_an_out_of_range_validation_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_by_database(_rows(4), fraction)


def test_split_refuses_a_single_database() -> None:
    """A database-disjoint split of one database is a contradiction, not an edge case."""
    with pytest.raises(ValueError, match="At least two databases"):
        split_by_database(_rows(1))


def test_split_is_deterministic_for_a_seed() -> None:
    first = split_by_database(_rows(10), 0.3, 7)
    second = split_by_database(_rows(10), 0.3, 7)
    assert [row.example_id for row in first[0]] == [row.example_id for row in second[0]]


def test_split_changes_with_the_seed() -> None:
    a_train, _ = split_by_database(_rows(10), 0.3, 1)
    b_train, _ = split_by_database(_rows(10), 0.3, 2)
    assert {row.db_id for row in a_train} != {row.db_id for row in b_train}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"n_bootstrap": 0}, "n_bootstrap", id="zero-resamples"),
        pytest.param({"confidence": 0.0}, "confidence", id="zero-confidence"),
        pytest.param({"confidence": 1.0}, "confidence", id="full-confidence"),
    ],
)
def test_bootstrap_rejects_invalid_parameters(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        paired_bootstrap_binary([True, False], [False, True], **kwargs)  # type: ignore[arg-type]


def test_bootstrap_refuses_empty_samples() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        paired_bootstrap_binary([], [])


def test_summarize_refuses_an_empty_evaluation() -> None:
    """Returning zeros would report a perfect-looking run for no data at all."""
    with pytest.raises(ValueError, match="Cannot summarize an empty evaluation"):
        summarize([])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def test_a_broken_reference_query_is_loud(sample_db: Path) -> None:
    """Scoring against a gold query that does not run would mark the model wrong."""
    example = _example(sample_db, "SELECT * FROM table_that_does_not_exist")
    with pytest.raises(RuntimeError, match="Gold SQL failed"):
        evaluate_prediction(example, "SELECT name FROM customers")


# --------------------------------------------------------------------------
# Schema extraction
# --------------------------------------------------------------------------


def test_schema_extraction_requires_a_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sqlite_schema_text(tmp_path / "absent.sqlite")


def test_an_empty_database_is_rejected(tmp_path: Path) -> None:
    """An empty schema would be embedded in every prompt for that database."""
    path = tmp_path / "empty.sqlite"
    sqlite3.connect(path).close()
    with pytest.raises(ValueError, match="no user tables"):
        sqlite_schema_text(path)


def test_identifier_quoting_escapes_embedded_quotes() -> None:
    assert quote_identifier('we"ird') == '"we""ird"'


# --------------------------------------------------------------------------
# SQL extraction
# --------------------------------------------------------------------------


def test_extract_requires_text() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        extract_sql(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_blank_output_extracts_to_empty(text: str) -> None:
    assert extract_sql(text) == ""


def test_prose_without_sql_is_returned_unchanged_rather_than_guessed() -> None:
    """No SELECT/WITH means nothing to anchor on; the evaluator will mark it invalid."""
    assert extract_sql("I cannot answer that.") == "I cannot answer that."


def test_only_the_first_statement_survives() -> None:
    assert extract_sql("SELECT 1; DROP TABLE t;") == "SELECT 1;"


# --------------------------------------------------------------------------
# Experiment planning
# --------------------------------------------------------------------------


def test_learning_curve_requires_a_positive_dataset() -> None:
    with pytest.raises(ValueError, match="total_examples must be positive"):
        learning_curve_plan(0)


@pytest.mark.parametrize("ranks", [(), (0,), (4, -8)])
def test_rank_ablation_rejects_non_positive_ranks(ranks: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="All ranks must be positive"):
        rank_ablation_plan(ranks)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def test_blank_lines_in_jsonl_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_a_non_object_line_is_rejected_with_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n[1, 2, 3]\n', encoding="utf-8")
    with pytest.raises(TypeError, match="Line 2"):
        list(read_jsonl(path))

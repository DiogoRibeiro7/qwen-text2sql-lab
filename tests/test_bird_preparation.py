"""Preparation is the entry point of the pipeline.

A defect here does not crash anything downstream; it silently changes what every
later number describes. These tests target the three ways that happens: an
unstable `example_id` (which is the join key between predictions and examples),
resolving the wrong database file, and admitting a malformed row.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from qwen_text2sql.data.bird import (
    find_sqlite_database,
    make_example_id,
    prepare_bird_rows,
)


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "db_id": "shop",
        "question": "Who are the customers?",
        "SQL": "SELECT name FROM customers",
    }
    row.update(overrides)
    return row


def _database(root: Path, db_id: str, *, nested: bool = True, suffix: str = ".sqlite") -> Path:
    directory = root / db_id if nested else root
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{db_id}{suffix}"
    connection = sqlite3.connect(path)
    try:
        connection.executescript("CREATE TABLE customers (customer_id INTEGER, name TEXT);")
        connection.commit()
    finally:
        connection.close()
    return path


# --------------------------------------------------------------------------
# example_id stability
# --------------------------------------------------------------------------


def test_example_id_is_stable_across_runs() -> None:
    """It joins predictions to examples; if it drifts, every comparison breaks."""
    first = make_example_id("shop", "Who?", "SELECT 1")
    second = make_example_id("shop", "Who?", "SELECT 1")
    assert first == second
    assert len(first) == 20


def test_example_id_is_pinned_to_a_known_value() -> None:
    """Regenerating IDs would orphan every stored prediction file."""
    assert make_example_id("shop", "Who?", "SELECT 1") == "3c593a23ad5ed760994b"


@pytest.mark.parametrize(
    ("db_id", "question", "gold_sql"),
    [
        pytest.param("other", "Who?", "SELECT 1", id="db_id"),
        pytest.param("shop", "What?", "SELECT 1", id="question"),
        pytest.param("shop", "Who?", "SELECT 2", id="gold_sql"),
    ],
)
def test_example_id_depends_on_every_component(db_id: str, question: str, gold_sql: str) -> None:
    assert make_example_id(db_id, question, gold_sql) != make_example_id("shop", "Who?", "SELECT 1")


def test_example_id_cannot_be_confused_by_field_boundaries() -> None:
    """Concatenating without a separator would collide these two examples."""
    assert make_example_id("a", "bc", "d") != make_example_id("ab", "c", "d")


# --------------------------------------------------------------------------
# Database resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nested", "suffix"),
    [
        pytest.param(True, ".sqlite", id="nested-sqlite"),
        pytest.param(True, ".db", id="nested-db"),
        pytest.param(False, ".sqlite", id="flat-sqlite"),
        pytest.param(False, ".db", id="flat-db"),
    ],
)
def test_resolves_the_common_bird_layouts(tmp_path: Path, nested: bool, suffix: str) -> None:
    expected = _database(tmp_path, "shop", nested=nested, suffix=suffix)
    assert find_sqlite_database(tmp_path, "shop") == expected.resolve()


def test_falls_back_to_a_recursive_search(tmp_path: Path) -> None:
    expected = _database(tmp_path / "extracted" / "train_databases", "shop")
    assert find_sqlite_database(tmp_path, "shop") == expected.resolve()


def test_ambiguous_databases_are_refused_rather_than_guessed(tmp_path: Path) -> None:
    """Silently picking one would score every query against the wrong data."""
    _database(tmp_path / "copy_a", "shop")
    _database(tmp_path / "copy_b", "shop")
    with pytest.raises(ValueError, match="Multiple database files"):
        find_sqlite_database(tmp_path, "shop")


def test_a_missing_database_is_reported_with_the_db_id(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="shop"):
        find_sqlite_database(tmp_path, "shop")


# --------------------------------------------------------------------------
# Row preparation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sql_key", ["SQL", "sql", "query", "gold_sql"])
def test_accepts_the_sql_key_spellings_bird_releases_use(tmp_path: Path, sql_key: str) -> None:
    _database(tmp_path, "shop")
    row = {"db_id": "shop", "question": "Who?", sql_key: "SELECT name FROM customers"}
    prepared = prepare_bird_rows([row], tmp_path)
    assert prepared[0].gold_sql == "SELECT name FROM customers"


@pytest.mark.parametrize("db_key", ["db_id", "database_id", "db_name"])
def test_accepts_the_database_key_spellings(tmp_path: Path, db_key: str) -> None:
    _database(tmp_path, "shop")
    row = {db_key: "shop", "question": "Who?", "SQL": "SELECT 1"}
    assert prepare_bird_rows([row], tmp_path)[0].db_id == "shop"


@pytest.mark.parametrize("evidence_key", ["evidence", "knowledge", "external_knowledge"])
def test_accepts_the_evidence_key_spellings(tmp_path: Path, evidence_key: str) -> None:
    _database(tmp_path, "shop")
    row = _row(**{evidence_key: "  names live in customers  "})
    assert prepare_bird_rows([row], tmp_path)[0].evidence == "names live in customers"


def test_optional_fields_default_to_empty(tmp_path: Path) -> None:
    _database(tmp_path, "shop")
    prepared = prepare_bird_rows([_row()], tmp_path)[0]
    assert prepared.evidence == ""
    assert prepared.difficulty == ""
    assert prepared.source_id == ""


def test_source_id_is_coerced_to_text(tmp_path: Path) -> None:
    """BIRD ships integer question_id; the data contract is all-strings."""
    _database(tmp_path, "shop")
    assert prepare_bird_rows([_row(question_id=17)], tmp_path)[0].source_id == "17"


def test_zero_is_kept_as_a_source_id(tmp_path: Path) -> None:
    """A falsy-but-present id must not be dropped by a truthiness test."""
    _database(tmp_path, "shop")
    assert prepare_bird_rows([_row(question_id=0)], tmp_path)[0].source_id == "0"


@pytest.mark.parametrize(
    "row",
    [
        pytest.param({"db_id": "shop", "SQL": "SELECT 1"}, id="no-question"),
        pytest.param({"db_id": "shop", "question": "Who?"}, id="no-sql"),
        pytest.param({"question": "Who?", "SQL": "SELECT 1"}, id="no-db"),
        pytest.param({"db_id": "shop", "question": "   ", "SQL": "SELECT 1"}, id="blank-question"),
        pytest.param({"db_id": "shop", "question": "Who?", "SQL": ""}, id="blank-sql"),
    ],
)
def test_malformed_rows_are_rejected(tmp_path: Path, row: dict[str, Any]) -> None:
    _database(tmp_path, "shop")
    with pytest.raises(KeyError):
        prepare_bird_rows([row], tmp_path)


def test_duplicate_examples_are_dropped_once(tmp_path: Path) -> None:
    """Duplicates would be scored twice and silently weight those databases."""
    _database(tmp_path, "shop")
    prepared = prepare_bird_rows([_row(), _row(), _row(question_id=9)], tmp_path)
    assert len(prepared) == 1


def test_rows_differing_only_in_question_are_both_kept(tmp_path: Path) -> None:
    _database(tmp_path, "shop")
    prepared = prepare_bird_rows([_row(), _row(question="Who spent most?")], tmp_path)
    assert len(prepared) == 2


def test_an_empty_result_is_an_error_not_an_empty_list(tmp_path: Path) -> None:
    _database(tmp_path, "shop")
    with pytest.raises(ValueError, match="No examples were prepared"):
        prepare_bird_rows([], tmp_path)


def test_schema_is_extracted_once_per_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-reading a schema for every row would dominate preparation time."""
    _database(tmp_path, "shop")
    calls: list[Path] = []
    import qwen_text2sql.data.bird as bird_module

    real = bird_module.sqlite_schema_text

    def counting(path: str | Path) -> str:
        calls.append(Path(path))
        return real(path)

    monkeypatch.setattr(bird_module, "sqlite_schema_text", counting)
    rows = [_row(question=f"Question {index}?") for index in range(5)]
    prepared = prepare_bird_rows(rows, tmp_path)

    assert len(prepared) == 5
    assert len(calls) == 1


def test_every_prepared_example_carries_the_full_contract(tmp_path: Path) -> None:
    _database(tmp_path, "shop")
    prepared = prepare_bird_rows([_row(evidence="hint", difficulty="simple")], tmp_path)[0]
    assert prepared.db_id == "shop"
    assert prepared.question == "Who are the customers?"
    assert prepared.evidence == "hint"
    assert prepared.difficulty == "simple"
    assert "customers" in prepared.schema
    assert Path(prepared.db_path).is_file()
    assert set(prepared.to_dict()) == {
        "example_id",
        "db_id",
        "question",
        "evidence",
        "gold_sql",
        "schema",
        "db_path",
        "difficulty",
        "source_id",
    }

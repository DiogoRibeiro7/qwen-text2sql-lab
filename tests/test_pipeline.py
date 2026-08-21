from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from qwen_text2sql.pipeline import prepared_from_mapping
from qwen_text2sql.types import PreparedExample

REQUIRED_FIELDS = ("example_id", "db_id", "question", "evidence", "gold_sql", "schema", "db_path")


def test_maps_every_contract_field(make_prepared_row: Callable[..., dict[str, Any]]) -> None:
    row = make_prepared_row()
    example = prepared_from_mapping(row)
    assert example.example_id == "example-0001"
    assert example.db_id == "shop"
    assert example.question == "Who are the customers?"
    assert example.evidence == "customers table holds names"
    assert example.gold_sql.startswith("SELECT name FROM customers")
    assert example.db_path == row["db_path"]
    assert example.difficulty == "simple"
    assert example.source_id == "bird-1"


def test_round_trips_through_to_dict(make_prepared_row: Callable[..., dict[str, Any]]) -> None:
    """The data contract must survive a write/read cycle unchanged."""
    original = prepared_from_mapping(make_prepared_row())
    assert prepared_from_mapping(original.to_dict()) == original


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_required_field_is_rejected(
    make_prepared_row: Callable[..., dict[str, Any]], field: str
) -> None:
    row = make_prepared_row()
    del row[field]
    with pytest.raises(KeyError, match=field):
        prepared_from_mapping(row)


def test_reports_every_missing_field_at_once(
    make_prepared_row: Callable[..., dict[str, Any]],
) -> None:
    """A partially prepared file should not need one run per missing column."""
    row = make_prepared_row()
    del row["question"]
    del row["gold_sql"]
    with pytest.raises(KeyError) as excinfo:
        prepared_from_mapping(row)
    message = str(excinfo.value)
    assert "question" in message
    assert "gold_sql" in message


def test_optional_metadata_defaults_to_empty(
    make_prepared_row: Callable[..., dict[str, Any]],
) -> None:
    """Older prepared files predate difficulty/source_id and must still load."""
    row = make_prepared_row()
    del row["difficulty"]
    del row["source_id"]
    example = prepared_from_mapping(row)
    assert example.difficulty == ""
    assert example.source_id == ""


def test_coerces_non_string_values(make_prepared_row: Callable[..., dict[str, Any]]) -> None:
    """Benchmark releases have shipped integer IDs; the contract is all-strings."""
    example = prepared_from_mapping(make_prepared_row(example_id=1234, source_id=7))
    assert example.example_id == "1234"
    assert example.source_id == "7"
    assert isinstance(example, PreparedExample)

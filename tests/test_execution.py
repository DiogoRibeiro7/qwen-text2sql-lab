from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from qwen_text2sql.evaluation.execution import (
    canonical_rows,
    execute_read_only,
    results_equivalent,
)
from qwen_text2sql.io import sha256_file

# Statements that never reach SQLite: the prefix check rejects them outright.
NON_READ_ONLY = [
    "INSERT INTO customers VALUES (3, 'Eve')",
    "UPDATE customers SET name = 'Eve'",
    "DELETE FROM customers",
    "DROP TABLE customers",
    "CREATE TABLE evil (x INTEGER)",
    "ALTER TABLE customers ADD COLUMN evil TEXT",
    "REPLACE INTO customers VALUES (1, 'Eve')",
    "ATTACH DATABASE 'other.sqlite' AS other",
    "PRAGMA writable_schema = ON",
    "VACUUM",
]


def test_execute_select(sample_db: Path) -> None:
    result = execute_read_only(sample_db, "SELECT name FROM customers ORDER BY name")
    assert result.ok
    assert result.rows == (("Ada",), ("Grace",))
    assert result.columns == ("name",)
    assert result.error_kind == "none"


# --------------------------------------------------------------------------
# Read-only sandbox. SECURITY.md treats escaping this as the security-relevant
# failure for this project, so each layer is asserted separately.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sql", NON_READ_ONLY)
def test_non_read_only_statements_are_rejected(sample_db: Path, sql: str) -> None:
    result = execute_read_only(sample_db, sql)
    assert not result.ok
    assert result.error_kind == "unsafe_statement"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("   \n\t  DELETE FROM customers", id="leading-whitespace"),
        pytest.param("DeLeTe FROM customers", id="mixed-case"),
        pytest.param("\ndrop table customers", id="leading-newline-lowercase"),
    ],
)
def test_prefix_check_is_not_fooled_by_whitespace_or_case(sample_db: Path, sql: str) -> None:
    result = execute_read_only(sample_db, sql)
    assert result.error_kind == "unsafe_statement"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("WITH t AS (SELECT 1) DELETE FROM customers", id="with-delete"),
        pytest.param(
            "WITH t AS (SELECT 1) INSERT INTO customers SELECT 3, 'Eve'", id="with-insert"
        ),
        pytest.param("WITH t AS (SELECT 1) UPDATE customers SET name = 'Eve'", id="with-update"),
    ],
)
def test_writes_smuggled_behind_a_cte_are_stopped_by_the_sandbox(sample_db: Path, sql: str) -> None:
    """A CTE-prefixed write passes the prefix check, so a later layer must hold.

    `_is_read_only_statement` only inspects the leading keyword, and SQLite
    accepts `WITH ... DELETE`. The mutation is prevented by the `mode=ro` URI
    and `PRAGMA query_only`, either of which suffices on its own; both are
    pinned by `test_both_sandbox_layers_are_configured`.
    """
    result = execute_read_only(sample_db, sql)
    assert not result.ok
    assert result.error_message is not None
    assert "readonly" in result.error_message.casefold()


def test_multiple_statements_cannot_be_chained(sample_db: Path) -> None:
    result = execute_read_only(sample_db, "SELECT 1; DROP TABLE customers")
    assert not result.ok
    assert result.error_kind == "execution_error"


def test_extension_loading_is_not_authorized(sample_db: Path) -> None:
    result = execute_read_only(sample_db, "SELECT load_extension('evil.so')")
    assert not result.ok
    assert result.error_kind == "execution_error"


class _SpyConnection:
    """Records the statements issued on a real connection."""

    def __init__(self, connection: sqlite3.Connection, statements: list[str]) -> None:
        self._connection = connection
        self._statements = statements

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        self._statements.append(sql)
        return self._connection.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def test_both_sandbox_layers_are_configured(
    sample_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Assert the URI mode and the PRAGMA separately, because each alone suffices.

    A write is blocked by `mode=ro` *or* by `PRAGMA query_only`, and both produce
    the same SQLite error. So no black-box test notices when one layer is
    dropped, even though that silently reduces the sandbox to a single point of
    failure. Pin both.
    """
    statements: list[str] = []
    seen_uris: list[str] = []
    real_connect = sqlite3.connect

    def spy_connect(database: str, *args: Any, **kwargs: Any) -> Any:
        # Named `database`, not `uri`: execute_read_only also passes `uri=True`.
        seen_uris.append(database)
        return _SpyConnection(real_connect(database, *args, **kwargs), statements)

    monkeypatch.setattr(sqlite3, "connect", spy_connect)
    assert execute_read_only(sample_db, "SELECT 1").ok

    assert seen_uris and "mode=ro" in seen_uris[0]
    issued = " ".join(statements).casefold()
    assert "pragma query_only" in issued
    assert "query_only = on" in issued


def test_database_is_byte_identical_after_every_attack(sample_db: Path) -> None:
    """The strongest statement available: the file itself never changes."""
    before = sha256_file(sample_db)
    attacks = [
        *NON_READ_ONLY,
        "WITH t AS (SELECT 1) DELETE FROM customers",
        "SELECT 1; DROP TABLE customers",
        "SELECT load_extension('evil.so')",
    ]
    for sql in attacks:
        execute_read_only(sample_db, sql)
    assert sha256_file(sample_db) == before

    connection = sqlite3.connect(sample_db)
    try:
        assert connection.execute("SELECT count(*) FROM customers").fetchone()[0] == 2
    finally:
        connection.close()


# --------------------------------------------------------------------------
# Error classification. The error_kind is a reported metric, so a
# misclassification silently distorts the published error breakdown.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        pytest.param("SELECT FROM customers", "syntax_error", id="syntax-error"),
        pytest.param("SELECT * FROM customers WHERE", "syntax_error", id="incomplete-input"),
        pytest.param("SELECT name FROM nope", "missing_table", id="missing-table"),
        pytest.param("SELECT missing FROM customers", "missing_column", id="missing-column"),
        pytest.param(
            "SELECT customer_id FROM customers a, customers b",
            "missing_column",
            id="ambiguous-column",
        ),
    ],
)
def test_sqlite_errors_are_classified(sample_db: Path, sql: str, expected: str) -> None:
    assert execute_read_only(sample_db, sql).error_kind == expected


def test_a_runaway_query_is_interrupted_and_reported_as_a_timeout(sample_db: Path) -> None:
    """An unbounded recursive CTE must be stopped by the progress handler."""
    result = execute_read_only(
        sample_db,
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c",
        timeout_seconds=0.15,
        progress_steps=1000,
    )
    assert not result.ok
    assert result.error_kind == "timeout"


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_non_string_sql_is_a_type_error(sample_db: Path) -> None:
    with pytest.raises(TypeError):
        execute_read_only(sample_db, None)  # type: ignore[arg-type]


@pytest.mark.parametrize("sql", ["", "   ", "\n\t "])
def test_blank_sql_is_reported_as_an_empty_prediction(sample_db: Path, sql: str) -> None:
    """A model that emitted nothing must be distinguishable from one that erred."""
    result = execute_read_only(sample_db, sql)
    assert not result.ok
    assert result.error_kind == "empty_prediction"


def test_a_missing_database_raises_rather_than_scoring_zero(tmp_path: Path) -> None:
    """A misconfigured db_path must fail loudly, not silently score every query wrong."""
    with pytest.raises(FileNotFoundError):
        execute_read_only(tmp_path / "absent.sqlite", "SELECT 1")


# --------------------------------------------------------------------------
# Result comparison
# --------------------------------------------------------------------------


def test_result_equivalence_is_order_independent(sample_db: Path) -> None:
    first = execute_read_only(sample_db, "SELECT customer_id FROM customers ORDER BY customer_id")
    second = execute_read_only(
        sample_db, "SELECT customer_id FROM customers ORDER BY customer_id DESC"
    )
    assert results_equivalent(first, second)


def test_numerical_tolerance(sample_db: Path) -> None:
    first = execute_read_only(sample_db, "SELECT 1.0000001")
    second = execute_read_only(sample_db, "SELECT 1.0000002")
    assert results_equivalent(first, second, tolerance=1e-6)


def test_zero_tolerance_compares_exactly(sample_db: Path) -> None:
    first = execute_read_only(sample_db, "SELECT 1.0000001")
    second = execute_read_only(sample_db, "SELECT 1.0000002")
    assert not results_equivalent(first, second, tolerance=0)


def test_differing_column_counts_are_not_equivalent(sample_db: Path) -> None:
    first = execute_read_only(sample_db, "SELECT 1")
    second = execute_read_only(sample_db, "SELECT 1, 2")
    assert not results_equivalent(first, second)


def test_a_failed_result_is_never_equivalent(sample_db: Path) -> None:
    ok = execute_read_only(sample_db, "SELECT 1")
    failed = execute_read_only(sample_db, "SELECT missing FROM customers")
    assert not results_equivalent(ok, failed)
    assert not results_equivalent(failed, ok)
    assert not results_equivalent(failed, failed)


def test_distinct_rows_are_not_equivalent(sample_db: Path) -> None:
    first = execute_read_only(sample_db, "SELECT name FROM customers")
    second = execute_read_only(sample_db, "SELECT name FROM customers WHERE customer_id = 1")
    assert not results_equivalent(first, second)


def test_nulls_and_blobs_survive_canonicalization(sample_db: Path) -> None:
    result = execute_read_only(sample_db, "SELECT NULL, x'414243'")
    assert result.ok
    assert canonical_rows(result.rows) == canonical_rows(((None, b"ABC"),))


def test_infinity_is_canonicalized(sample_db: Path) -> None:
    result = execute_read_only(sample_db, "SELECT 1e999")
    assert result.ok
    assert canonical_rows(result.rows) == canonical_rows(((float("inf"),),))


def test_special_scalars_are_distinguished() -> None:
    """SQLite cannot produce these directly, so exercise the comparator itself."""
    assert canonical_rows(((True,),)) != canonical_rows(((1,),))
    assert canonical_rows(((float("nan"),),)) == canonical_rows(((float("nan"),),))
    assert canonical_rows(((float("inf"),),)) != canonical_rows(((float("-inf"),),))
    assert canonical_rows(((None,),)) != canonical_rows((("",),))


def test_text_is_compared_after_stripping(sample_db: Path) -> None:
    first = execute_read_only(sample_db, "SELECT '  Ada  '")
    second = execute_read_only(sample_db, "SELECT 'Ada'")
    assert results_equivalent(first, second)

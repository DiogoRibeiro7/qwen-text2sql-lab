"""Safe, read-only SQLite execution and result comparison."""

from __future__ import annotations

import math
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_text2sql.types import ErrorKind, QueryResult

_READ_ONLY_PREFIXES = ("select", "with")


def _classify_sqlite_error(message: str) -> ErrorKind:
    lowered = message.casefold()
    if "syntax error" in lowered or "incomplete input" in lowered:
        return "syntax_error"
    if "no such table" in lowered:
        return "missing_table"
    if "no such column" in lowered or "ambiguous column" in lowered:
        return "missing_column"
    if "interrupted" in lowered:
        return "timeout"
    return "execution_error"


def _is_read_only_statement(sql: str) -> bool:
    normalized = sql.lstrip().casefold()
    return normalized.startswith(_READ_ONLY_PREFIXES)


def execute_read_only(
    db_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float = 10.0,
    progress_steps: int = 10_000,
) -> QueryResult:
    """Execute one SELECT/WITH statement against SQLite in read-only mode.

    A wall-clock-aware progress handler interrupts long-running queries. SQLite URI
    read-only mode and PRAGMA query_only provide a second layer against mutation.
    """
    if not isinstance(sql, str):
        raise TypeError("sql must be a string")
    if not sql.strip():
        return QueryResult(False, (), (), "empty_prediction", "SQL is empty", 0.0)
    if not _is_read_only_statement(sql):
        return QueryResult(False, (), (), "unsafe_statement", "Only SELECT/WITH is allowed", 0.0)
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    started = time.perf_counter()
    deadline = started + timeout_seconds
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.set_progress_handler(lambda: int(time.perf_counter() > deadline), progress_steps)
        try:
            cursor = connection.execute(sql)
            rows = tuple(tuple(row) for row in cursor.fetchall())
            columns = tuple(item[0] for item in (cursor.description or ()))
            elapsed = (time.perf_counter() - started) * 1000.0
            return QueryResult(True, rows, columns, "none", None, elapsed)
        except sqlite3.Error as exc:
            elapsed = (time.perf_counter() - started) * 1000.0
            message = str(exc)
            return QueryResult(False, (), (), _classify_sqlite_error(message), message, elapsed)
    finally:
        connection.close()


def _normalise_scalar(value: Any, tolerance: float) -> tuple[str, Any]:
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if math.isnan(numeric):
            return ("float", "nan")
        if math.isinf(numeric):
            return ("float", "inf" if numeric > 0 else "-inf")
        if tolerance <= 0:
            return ("number", numeric)
        return ("number", round(numeric / tolerance) * tolerance)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    return ("text", str(value).strip())


def canonical_rows(rows: tuple[tuple[Any, ...], ...], tolerance: float = 1e-6) -> Counter[tuple[Any, ...]]:
    """Canonicalize result rows as an order-independent multiset."""
    return Counter(
        tuple(_normalise_scalar(value, tolerance) for value in row)
        for row in rows
    )


def results_equivalent(gold: QueryResult, predicted: QueryResult, tolerance: float = 1e-6) -> bool:
    """Return whether two successful query results are multiset-equivalent."""
    if not gold.ok or not predicted.ok:
        return False
    if len(gold.columns) != len(predicted.columns):
        return False
    return canonical_rows(gold.rows, tolerance) == canonical_rows(predicted.rows, tolerance)

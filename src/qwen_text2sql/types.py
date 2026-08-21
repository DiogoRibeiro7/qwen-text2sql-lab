"""Typed data contracts shared across the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ErrorKind = Literal[
    "none",
    "empty_prediction",
    "unsafe_statement",
    "syntax_error",
    "missing_table",
    "missing_column",
    "timeout",
    "execution_error",
]


@dataclass(frozen=True, slots=True)
class PreparedExample:
    """One text-to-SQL example with the database schema embedded as text."""

    example_id: str
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    schema: str
    db_path: str
    difficulty: str = ""
    source_id: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Outcome of executing a single SQL statement in a read-only database."""

    ok: bool
    rows: tuple[tuple[Any, ...], ...]
    columns: tuple[str, ...]
    error_kind: ErrorKind
    error_message: str | None
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """Per-example model evaluation record."""

    example_id: str
    db_id: str
    gold_sql: str
    predicted_sql: str
    valid_sql: bool
    execution_match: bool
    exact_match: bool
    error_kind: ErrorKind
    latency_ms: float

    def to_dict(self) -> dict[str, str | bool | float]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    """Resolved file-system locations used by a run."""

    data_path: Path
    output_dir: Path
    model_output_dir: Path

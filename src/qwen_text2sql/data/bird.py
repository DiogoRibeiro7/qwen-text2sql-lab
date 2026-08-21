"""BIRD dataset preparation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from qwen_text2sql.data.schema import sqlite_schema_text
from qwen_text2sql.types import PreparedExample

QUESTION_KEYS = ("question", "Question")
SQL_KEYS = ("SQL", "sql", "query", "gold_sql")
DB_KEYS = ("db_id", "database_id", "db_name")
EVIDENCE_KEYS = ("evidence", "knowledge", "external_knowledge")
DIFFICULTY_KEYS = ("difficulty", "level")
SOURCE_ID_KEYS = ("question_id", "id")


def _first_text(row: Mapping[str, Any], keys: tuple[str, ...], *, required: bool) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if required:
        raise KeyError(f"None of the required keys are present with text: {keys}")
    return ""


def find_sqlite_database(db_root: str | Path, db_id: str) -> Path:
    """Resolve a BIRD SQLite database using common directory layouts."""
    root = Path(db_root)
    candidates = [
        root / db_id / f"{db_id}.sqlite",
        root / db_id / f"{db_id}.db",
        root / f"{db_id}.sqlite",
        root / f"{db_id}.db",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    glob_matches = sorted(root.glob(f"**/{db_id}.sqlite")) + sorted(root.glob(f"**/{db_id}.db"))
    if len(glob_matches) == 1:
        return glob_matches[0].resolve()
    if len(glob_matches) > 1:
        raise ValueError(f"Multiple database files found for {db_id}: {glob_matches}")
    raise FileNotFoundError(f"Could not locate SQLite database for db_id={db_id!r} under {root}")


def make_example_id(db_id: str, question: str, gold_sql: str) -> str:
    """Build a stable identifier from the semantic identity of an example."""
    payload = "\x1f".join((db_id, question, gold_sql)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def prepare_bird_rows(rows: Iterable[Mapping[str, Any]], db_root: str | Path) -> list[PreparedExample]:
    """Convert BIRD-like rows into schema-grounded examples."""
    schema_cache: dict[str, tuple[Path, str]] = {}
    prepared: list[PreparedExample] = []
    seen_ids: set[str] = set()
    for row in rows:
        question = _first_text(row, QUESTION_KEYS, required=True)
        gold_sql = _first_text(row, SQL_KEYS, required=True)
        db_id = _first_text(row, DB_KEYS, required=True)
        evidence = _first_text(row, EVIDENCE_KEYS, required=False)
        difficulty = _first_text(row, DIFFICULTY_KEYS, required=False)
        source_id_value = next((row.get(key) for key in SOURCE_ID_KEYS if row.get(key) is not None), "")
        source_id = str(source_id_value)
        if db_id not in schema_cache:
            db_path = find_sqlite_database(db_root, db_id)
            schema_cache[db_id] = (db_path, sqlite_schema_text(db_path))
        db_path, schema = schema_cache[db_id]
        example_id = make_example_id(db_id, question, gold_sql)
        if example_id in seen_ids:
            continue
        seen_ids.add(example_id)
        prepared.append(
            PreparedExample(
                example_id=example_id,
                db_id=db_id,
                question=question,
                evidence=evidence,
                gold_sql=gold_sql,
                schema=schema,
                db_path=str(db_path),
                difficulty=difficulty,
                source_id=source_id,
            )
        )
    if not prepared:
        raise ValueError("No examples were prepared")
    return prepared

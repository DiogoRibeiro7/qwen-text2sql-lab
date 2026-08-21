"""SQLite schema extraction utilities."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier for PRAGMA statements."""
    return '"' + identifier.replace('"', '""') + '"'


def sqlite_schema_text(db_path: str | Path) -> str:
    """Return a compact, deterministic text representation of a SQLite schema."""
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        blocks: list[str] = []
        for (table_name_raw,) in tables:
            table_name = str(table_name_raw)
            columns = connection.execute(
                f"PRAGMA table_info({quote_identifier(table_name)})"
            ).fetchall()
            column_lines = [
                f"  {str(row[1])} {str(row[2]) or 'UNKNOWN'}"
                + (" PRIMARY KEY" if int(row[5]) else "")
                for row in columns
            ]
            foreign_keys = connection.execute(
                f"PRAGMA foreign_key_list({quote_identifier(table_name)})"
            ).fetchall()
            for fk in foreign_keys:
                column_lines.append(
                    f"  FOREIGN KEY ({str(fk[3])}) REFERENCES {str(fk[2])}({str(fk[4])})"
                )
            blocks.append(f"TABLE {table_name} (\n" + ",\n".join(column_lines) + "\n)")
        if not blocks:
            raise ValueError(f"Database has no user tables: {path}")
        return "\n\n".join(blocks)
    finally:
        connection.close()

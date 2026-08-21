from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

GOLD_SQL = "SELECT name FROM customers ORDER BY customer_id"


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );
            INSERT INTO customers VALUES (1, 'Ada'), (2, 'Grace');
            INSERT INTO orders VALUES (10, 1, 12.5), (11, 1, 7.5), (12, 2, 30.0);
            """
        )
        connection.commit()
    finally:
        connection.close()
    return path


@pytest.fixture()
def make_prepared_row(sample_db: Path) -> Callable[..., dict[str, Any]]:
    """Build a prepared-example mapping backed by the sample database.

    Every field of the data contract is populated, so a test can override one
    field to isolate the behaviour it cares about.
    """

    def make(**overrides: Any) -> dict[str, Any]:
        row: dict[str, Any] = {
            "example_id": "example-0001",
            "db_id": "shop",
            "question": "Who are the customers?",
            "evidence": "customers table holds names",
            "gold_sql": GOLD_SQL,
            "schema": "CREATE TABLE customers (customer_id INTEGER, name TEXT);",
            "db_path": str(sample_db),
            "difficulty": "simple",
            "source_id": "bird-1",
        }
        row.update(overrides)
        return row

    return make

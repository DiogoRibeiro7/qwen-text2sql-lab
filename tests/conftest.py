from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


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

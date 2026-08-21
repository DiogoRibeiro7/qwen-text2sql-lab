from pathlib import Path

from qwen_text2sql.data.schema import sqlite_schema_text


def test_schema_contains_tables_columns_and_foreign_keys(sample_db: Path) -> None:
    schema = sqlite_schema_text(sample_db)
    assert "TABLE customers" in schema
    assert "customer_id INTEGER PRIMARY KEY" in schema
    assert "TABLE orders" in schema
    assert "FOREIGN KEY (customer_id) REFERENCES customers(customer_id)" in schema

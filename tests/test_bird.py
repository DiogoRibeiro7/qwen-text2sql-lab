from pathlib import Path

from qwen_text2sql.data.bird import find_sqlite_database, prepare_bird_rows


def test_find_database_in_nested_layout(sample_db: Path, tmp_path: Path) -> None:
    root = tmp_path / "dbs"
    nested = root / "shop"
    nested.mkdir(parents=True)
    destination = nested / "shop.sqlite"
    destination.write_bytes(sample_db.read_bytes())
    assert find_sqlite_database(root, "shop") == destination.resolve()


def test_prepare_rows_embeds_schema(sample_db: Path, tmp_path: Path) -> None:
    root = tmp_path / "dbs"
    root.mkdir()
    destination = root / "shop.sqlite"
    destination.write_bytes(sample_db.read_bytes())
    rows = [
        {
            "db_id": "shop",
            "question": "Who spent most?",
            "evidence": "amount means spend",
            "SQL": (
                "SELECT customer_id FROM orders "
                "GROUP BY customer_id ORDER BY SUM(amount) DESC LIMIT 1"
            ),
        }
    ]
    prepared = prepare_bird_rows(rows, root)
    assert len(prepared) == 1
    assert prepared[0].db_id == "shop"
    assert "TABLE orders" in prepared[0].schema
    assert len(prepared[0].example_id) == 20

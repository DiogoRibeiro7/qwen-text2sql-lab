from qwen_text2sql.sql.extract import extract_sql, normalize_sql


def test_extract_from_fence_and_explanation() -> None:
    raw = "Answer:\n```sql\nSELECT name FROM customers;\n```\nThis returns the name."
    assert extract_sql(raw) == "SELECT name FROM customers;"


def test_extract_first_statement() -> None:
    assert extract_sql("SELECT 1; SELECT 2;") == "SELECT 1;"


def test_normalize_sql() -> None:
    assert normalize_sql(" SELECT  *\nFROM x; ") == "select * from x"

from pathlib import Path

from qwen_text2sql.evaluation.execution import execute_read_only, results_equivalent


def test_execute_select(sample_db: Path) -> None:
    result = execute_read_only(sample_db, "SELECT name FROM customers ORDER BY name")
    assert result.ok
    assert result.rows == (("Ada",), ("Grace",))


def test_mutation_is_rejected(sample_db: Path) -> None:
    result = execute_read_only(sample_db, "DELETE FROM customers")
    assert not result.ok
    assert result.error_kind == "unsafe_statement"


def test_missing_column_is_classified(sample_db: Path) -> None:
    result = execute_read_only(sample_db, "SELECT missing FROM customers")
    assert result.error_kind == "missing_column"


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

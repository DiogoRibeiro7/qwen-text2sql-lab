from qwen_text2sql.data.splits import split_by_database
from qwen_text2sql.types import PreparedExample


def example(index: int, db_id: str) -> PreparedExample:
    return PreparedExample(
        example_id=str(index),
        db_id=db_id,
        question="q",
        evidence="",
        gold_sql="SELECT 1",
        schema="TABLE x (a INTEGER)",
        db_path="x.sqlite",
    )


def test_split_is_database_disjoint_and_deterministic() -> None:
    rows = [example(i, f"db_{i // 2}") for i in range(12)]
    train1, valid1 = split_by_database(rows, validation_fraction=0.25, seed=7)
    train2, valid2 = split_by_database(rows, validation_fraction=0.25, seed=7)
    assert [row.example_id for row in train1] == [row.example_id for row in train2]
    assert [row.example_id for row in valid1] == [row.example_id for row in valid2]
    assert {row.db_id for row in train1}.isdisjoint({row.db_id for row in valid1})

from pathlib import Path

from qwen_text2sql.evaluation.evaluator import evaluate_prediction
from qwen_text2sql.types import PreparedExample


def test_semantic_execution_match_can_differ_from_exact_match(sample_db: Path) -> None:
    example = PreparedExample(
        example_id="x",
        db_id="shop",
        question="List customers",
        evidence="",
        gold_sql="SELECT name FROM customers ORDER BY customer_id",
        schema="schema",
        db_path=str(sample_db),
    )
    result = evaluate_prediction(example, "```sql\nSELECT name FROM customers ORDER BY name\n```")
    assert result.valid_sql
    assert result.execution_match
    assert not result.exact_match

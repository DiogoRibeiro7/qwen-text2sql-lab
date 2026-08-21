from qwen_text2sql.training.formatting import SYSTEM_MESSAGE, sft_record, user_content
from qwen_text2sql.types import PreparedExample


def test_sft_format_contains_schema_evidence_and_gold() -> None:
    example = PreparedExample(
        example_id="1",
        db_id="db",
        question="How many?",
        evidence="x means y",
        gold_sql="SELECT COUNT(*) FROM t",
        schema="TABLE t (x INTEGER)",
        db_path="db.sqlite",
    )
    record = sft_record(example)
    assert record["prompt"][0]["content"] == SYSTEM_MESSAGE
    assert "TABLE t" in user_content(example)
    assert "x means y" in user_content(example)
    assert record["completion"][0]["content"] == "SELECT COUNT(*) FROM t"

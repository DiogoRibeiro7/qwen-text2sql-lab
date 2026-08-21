from qwen_text2sql.evaluation.metrics import summarize
from qwen_text2sql.types import EvaluationRecord


def record(ok: bool, error: str = "none") -> EvaluationRecord:
    return EvaluationRecord(
        example_id="x",
        db_id="db",
        gold_sql="SELECT 1",
        predicted_sql="SELECT 1" if ok else "SELECT bad",
        valid_sql=ok,
        execution_match=ok,
        exact_match=ok,
        error_kind=error,  # type: ignore[arg-type]
        latency_ms=10.0,
    )


def test_summary() -> None:
    result = summarize([record(True), record(False, "missing_column")])
    assert result["n"] == 2
    assert result["execution_accuracy"] == 0.5
    assert result["errors"] == {"missing_column": 1}

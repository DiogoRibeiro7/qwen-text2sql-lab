"""Tests for the prediction and evaluation entry points.

`scripts/generate_predictions.py` and `scripts/evaluate_predictions.py` had
reimplemented what `qwen_text2sql.pipeline` and the CLI already did, and the
copies had drifted apart:

* the script omitted `difficulty` from prediction rows, so whether a prediction
  file could be stratified by difficulty depended on which entry point wrote it;
* the script raised a bare `KeyError` for a prediction whose example was absent,
  where the CLI names the offending id;
* `--limit 0` silently produced an empty predictions file and `--limit -1`
  silently dropped the last example, where the library refuses both.

Both now delegate to shared helpers, so the schema and the guards have one
definition. These tests hold them to it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from qwen_text2sql import pipeline
from qwen_text2sql.io import read_jsonl, write_jsonl
from qwen_text2sql.types import PreparedExample

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
GOLD_SQL = "SELECT name FROM customers ORDER BY customer_id"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generate_predictions = _load("generate_predictions")
evaluate_predictions = _load("evaluate_predictions")

PREDICTION_KEYS = {
    "example_id",
    "db_id",
    "difficulty",
    "prediction",
    "generation_latency_ms",
    "model_id",
    "adapter",
}


def _example(example_id: str = "a", *, difficulty: str = "simple") -> PreparedExample:
    return PreparedExample(
        example_id=example_id,
        db_id="shop",
        question="Who?",
        evidence="",
        gold_sql=GOLD_SQL,
        schema="",
        db_path="unused.sqlite",
        difficulty=difficulty,
    )


# --------------------------------------------------------------------------
# One definition of the prediction schema
# --------------------------------------------------------------------------


def test_the_prediction_schema_has_a_single_definition() -> None:
    row = pipeline.prediction_row(
        _example(), GOLD_SQL, 12.5, model_id="Qwen/Qwen3.5-4B", adapter_path=None
    )
    assert set(row) == PREDICTION_KEYS
    assert row["difficulty"] == "simple"
    assert row["adapter"] is None


def test_an_adapter_path_is_recorded_as_text() -> None:
    row = pipeline.prediction_row(
        _example(), GOLD_SQL, 1.0, model_id="m", adapter_path=Path("results/adapter")
    )
    assert row["adapter"] == str(Path("results/adapter"))


def test_the_script_and_the_pipeline_agree_on_the_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_prepared_row: Callable[..., dict[str, Any]],
) -> None:
    """The drift that motivated this: difficulty was present in one and not the other."""
    data = tmp_path / "data.jsonl"
    write_jsonl(data, [make_prepared_row(example_id="a", gold_sql=GOLD_SQL, difficulty="simple")])
    config = tmp_path / "c.yaml"
    config.write_text(
        "model:\n  model_id: tiny\nlora:\n  rank: 8\n  alpha: 16\n"
        f"training:\n  output_dir: {tmp_path / 'out'}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        generate_predictions, "load_inference_model", lambda *a, **k: ("model", "tok")
    )
    monkeypatch.setattr(generate_predictions, "generate_sql", lambda *a, **k: (GOLD_SQL, 3.0))
    output = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions.py",
            "--config",
            str(config),
            "--data",
            str(data),
            "--output",
            str(output),
        ],
    )
    generate_predictions.main()

    row = next(iter(read_jsonl(output)))
    assert set(row) == PREDICTION_KEYS
    assert row["difficulty"] == "simple"


# --------------------------------------------------------------------------
# The limit guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_limit_is_refused(limit: int) -> None:
    """[:0] evaluates nothing and [:-1] drops the last example, both silently."""
    with pytest.raises(ValueError, match="limit must be positive"):
        pipeline.apply_limit([_example("a"), _example("b")], limit)


def test_a_limit_truncates_from_the_front() -> None:
    examples = [_example(f"e{i}") for i in range(5)]
    assert [e.example_id for e in pipeline.apply_limit(examples, 2)] == ["e0", "e1"]


def test_no_limit_keeps_everything() -> None:
    examples = [_example(f"e{i}") for i in range(3)]
    assert pipeline.apply_limit(examples, None) == examples


def test_the_script_rejects_a_bad_limit_before_loading_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Downloading several gigabytes only to reject an argument is unkind."""
    config = tmp_path / "c.yaml"
    config.write_text(
        "model:\n  model_id: tiny\nlora:\n  rank: 8\n  alpha: 16\n"
        f"training:\n  output_dir: {tmp_path / 'out'}\n",
        encoding="utf-8",
    )
    loaded: list[str] = []
    monkeypatch.setattr(
        generate_predictions,
        "load_inference_model",
        lambda *a, **k: (loaded.append("loaded"), ("m", "t"))[1],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_predictions.py",
            "--config",
            str(config),
            "--data",
            str(tmp_path / "absent.jsonl"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--limit",
            "0",
        ],
    )
    with pytest.raises(ValueError, match="--limit must be positive"):
        generate_predictions.main()
    assert loaded == []


# --------------------------------------------------------------------------
# Pairing predictions with examples
# --------------------------------------------------------------------------


def test_an_unknown_example_id_names_itself(sample_db: Path) -> None:
    """A bare KeyError from a dict lookup does not say which file is at fault."""
    example = _example("a")
    with pytest.raises(KeyError) as excinfo:
        pipeline.evaluate_prediction_rows(
            [example], [{"example_id": "not-in-dataset", "prediction": GOLD_SQL}]
        )
    # A bare dict lookup would also mention the id, so assert the explanation.
    message = str(excinfo.value)
    assert "unknown example_id" in message
    assert "not-in-dataset" in message


def test_the_legacy_predicted_sql_key_is_accepted(
    make_prepared_row: Callable[..., dict[str, Any]],
) -> None:
    example = pipeline.prepared_from_mapping(make_prepared_row(example_id="a", gold_sql=GOLD_SQL))
    records = pipeline.evaluate_prediction_rows(
        [example], [{"example_id": "a", "predicted_sql": GOLD_SQL}]
    )
    assert records[0].execution_match is True


def test_an_empty_prediction_file_is_refused(
    make_prepared_row: Callable[..., dict[str, Any]],
) -> None:
    """Summarising nothing would write a metrics file describing no data."""
    example = pipeline.prepared_from_mapping(make_prepared_row(example_id="a"))
    with pytest.raises(ValueError, match="No predictions to evaluate"):
        pipeline.evaluate_prediction_rows([example], [])


# --------------------------------------------------------------------------
# evaluate_predictions.py end to end
# --------------------------------------------------------------------------


def test_evaluate_writes_records_and_metrics_as_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_prepared_row: Callable[..., dict[str, Any]],
) -> None:
    """The metrics write had no explicit encoding, unlike every other write here."""
    data = tmp_path / "data.jsonl"
    write_jsonl(data, [make_prepared_row(example_id="a", gold_sql=GOLD_SQL)])
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [{"example_id": "a", "prediction": GOLD_SQL}])
    records_out = tmp_path / "records.jsonl"
    metrics_out = tmp_path / "nested" / "metrics.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_predictions.py",
            "--data",
            str(data),
            "--predictions",
            str(predictions),
            "--records-output",
            str(records_out),
            "--metrics-output",
            str(metrics_out),
        ],
    )
    evaluate_predictions.main()

    assert len(list(read_jsonl(records_out))) == 1
    metrics = json.loads(metrics_out.read_bytes().decode("utf-8"))
    assert metrics["n"] == 1
    assert metrics["execution_accuracy"] == 1.0


def test_evaluate_refuses_a_prediction_for_an_unknown_example(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_prepared_row: Callable[..., dict[str, Any]],
) -> None:
    data = tmp_path / "data.jsonl"
    write_jsonl(data, [make_prepared_row(example_id="a", gold_sql=GOLD_SQL)])
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [{"example_id": "ghost", "prediction": GOLD_SQL}])
    records_out = tmp_path / "records.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_predictions.py",
            "--data",
            str(data),
            "--predictions",
            str(predictions),
            "--records-output",
            str(records_out),
            "--metrics-output",
            str(tmp_path / "metrics.json"),
        ],
    )
    with pytest.raises(KeyError) as excinfo:
        evaluate_predictions.main()
    assert "unknown example_id" in str(excinfo.value)
    assert not records_out.exists()

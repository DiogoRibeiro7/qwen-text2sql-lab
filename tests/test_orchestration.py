"""The orchestration that turns a model into stored evidence.

`generate_and_evaluate` and the CLI subcommands are the code that actually runs
an experiment, and they were the least-covered non-GPU code in the package.
Only the model call is faked here; the rest — the loop, the limit handling, the
three output files and the metrics — is exercised for real against a real SQLite
database, because that is where a defect would silently corrupt a result.

`train_adapter` is deliberately absent. Faking `datasets`, `peft`, `trl` and the
model would leave a test that exercises only the mocks.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qwen_text2sql import pipeline
from qwen_text2sql.cli import _prepare, _train, build_parser
from qwen_text2sql.config import ModelConfig
from qwen_text2sql.io import read_jsonl, write_jsonl
from qwen_text2sql.types import PreparedExample

GOLD_SQL = "SELECT name FROM customers ORDER BY customer_id"
WRONG_SQL = "SELECT name FROM customers WHERE customer_id = 1"

MODEL = ModelConfig(model_id="Qwen/Qwen3.5-4B")


@pytest.fixture()
def dataset(tmp_path: Path, make_prepared_row: Callable[..., dict[str, Any]]) -> Path:
    path = tmp_path / "validation.jsonl"
    write_jsonl(
        path,
        [make_prepared_row(example_id=f"e{i}", gold_sql=GOLD_SQL) for i in range(4)],
    )
    return path


def _fake_generation(
    monkeypatch: pytest.MonkeyPatch, predictions: list[str], *, latency: float = 12.5
) -> list[PreparedExample]:
    """Replace only the model call, recording the examples it was asked about."""
    seen: list[PreparedExample] = []
    queue = list(predictions)

    def fake_load(model_id: str, adapter_path: str | Path | None = None) -> tuple[Any, Any]:
        return ("model", "tokenizer")

    def fake_generate(
        model: Any, tokenizer: Any, example: PreparedExample, config: ModelConfig
    ) -> tuple[str, float]:
        seen.append(example)
        return (queue.pop(0) if queue else GOLD_SQL), latency

    monkeypatch.setattr(pipeline, "load_inference_model", fake_load)
    monkeypatch.setattr(pipeline, "generate_sql", fake_generate)
    return seen


# --------------------------------------------------------------------------
# generate_and_evaluate
# --------------------------------------------------------------------------


def test_writes_predictions_records_and_metrics(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 4: keep per-example evidence, not only the aggregate score."""
    _fake_generation(monkeypatch, [GOLD_SQL, GOLD_SQL, WRONG_SQL, "SELECT * FROM nope"])
    metrics = pipeline.generate_and_evaluate(
        model_config=MODEL,
        data_path=dataset,
        predictions_path=tmp_path / "out" / "predictions.jsonl",
        records_path=tmp_path / "out" / "records.jsonl",
        metrics_path=tmp_path / "out" / "metrics.json",
    )

    predictions = list(read_jsonl(tmp_path / "out" / "predictions.jsonl"))
    records = list(read_jsonl(tmp_path / "out" / "records.jsonl"))
    on_disk = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))

    assert len(predictions) == len(records) == 4
    assert metrics["n"] == 4
    assert metrics["execution_accuracy"] == 0.5
    assert on_disk == metrics


def test_every_prediction_row_carries_its_provenance(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prediction file that does not say which model produced it is not evidence."""
    _fake_generation(monkeypatch, [GOLD_SQL] * 4)
    pipeline.generate_and_evaluate(
        model_config=MODEL,
        data_path=dataset,
        predictions_path=tmp_path / "predictions.jsonl",
        records_path=tmp_path / "records.jsonl",
        metrics_path=tmp_path / "metrics.json",
        adapter_path=tmp_path / "adapter",
    )
    row = next(iter(read_jsonl(tmp_path / "predictions.jsonl")))
    assert set(row) == {
        "example_id",
        "db_id",
        "difficulty",
        "prediction",
        "generation_latency_ms",
        "model_id",
        "adapter",
    }
    assert row["model_id"] == "Qwen/Qwen3.5-4B"
    assert row["adapter"] == str(tmp_path / "adapter")
    assert row["generation_latency_ms"] == 12.5


def test_an_absent_adapter_is_recorded_as_null(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinguishing the baseline from an adapted run must not rely on a filename."""
    _fake_generation(monkeypatch, [GOLD_SQL] * 4)
    pipeline.generate_and_evaluate(
        model_config=MODEL,
        data_path=dataset,
        predictions_path=tmp_path / "predictions.jsonl",
        records_path=tmp_path / "records.jsonl",
        metrics_path=tmp_path / "metrics.json",
    )
    assert next(iter(read_jsonl(tmp_path / "predictions.jsonl")))["adapter"] is None


def test_limit_truncates_from_the_front_deterministically(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_generation(monkeypatch, [GOLD_SQL] * 4)
    metrics = pipeline.generate_and_evaluate(
        model_config=MODEL,
        data_path=dataset,
        predictions_path=tmp_path / "predictions.jsonl",
        records_path=tmp_path / "records.jsonl",
        metrics_path=tmp_path / "metrics.json",
        limit=2,
    )
    assert metrics["n"] == 2
    assert [example.example_id for example in seen] == ["e0", "e1"]


@pytest.mark.parametrize("limit", [0, -1])
def test_a_non_positive_limit_is_rejected(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    _fake_generation(monkeypatch, [GOLD_SQL] * 4)
    with pytest.raises(ValueError, match="limit must be positive"):
        pipeline.generate_and_evaluate(
            model_config=MODEL,
            data_path=dataset,
            predictions_path=tmp_path / "predictions.jsonl",
            records_path=tmp_path / "records.jsonl",
            metrics_path=tmp_path / "metrics.json",
            limit=limit,
        )


def test_an_empty_dataset_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scoring nothing would write a metrics file describing no data."""
    empty = tmp_path / "empty.jsonl"
    write_jsonl(empty, [])
    _fake_generation(monkeypatch, [])
    with pytest.raises(ValueError, match="Evaluation dataset is empty"):
        pipeline.generate_and_evaluate(
            model_config=MODEL,
            data_path=empty,
            predictions_path=tmp_path / "predictions.jsonl",
            records_path=tmp_path / "records.jsonl",
            metrics_path=tmp_path / "metrics.json",
        )


def test_no_output_is_written_when_generation_fails(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-written prediction file would be silently scored on the next run."""

    def exploding(*args: Any, **kwargs: Any) -> tuple[str, float]:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(pipeline, "load_inference_model", lambda *a, **k: ("m", "t"))
    monkeypatch.setattr(pipeline, "generate_sql", exploding)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        pipeline.generate_and_evaluate(
            model_config=MODEL,
            data_path=dataset,
            predictions_path=tmp_path / "predictions.jsonl",
            records_path=tmp_path / "records.jsonl",
            metrics_path=tmp_path / "metrics.json",
        )
    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "metrics.json").exists()


# --------------------------------------------------------------------------
# CLI subcommands
# --------------------------------------------------------------------------


def test_prepare_bird_writes_prepared_records(
    tmp_path: Path,
    sample_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The `datasets` dependency is faked; preparation and writing are real."""
    db_root = tmp_path / "databases" / "shop"
    db_root.mkdir(parents=True)
    (db_root / "shop.sqlite").write_bytes(sample_db.read_bytes())

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda name, split: [  # type: ignore[attr-defined]
        {"db_id": "shop", "question": "Who?", "SQL": GOLD_SQL},
        {"db_id": "shop", "question": "Who spent most?", "SQL": "SELECT customer_id FROM orders"},
    ]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    output = tmp_path / "prepared.jsonl"
    args = build_parser().parse_args(
        [
            "prepare-bird",
            "--dataset",
            "birdsql/bird23-train-filtered",
            "--db-root",
            str(tmp_path / "databases"),
            "--output",
            str(output),
        ]
    )
    _prepare(args)

    rows = list(read_jsonl(output))
    assert len(rows) == 2
    assert {row["db_id"] for row in rows} == {"shop"}
    assert all("customers" in row["schema"] for row in rows)
    assert "Wrote 2 examples" in capsys.readouterr().out


def test_train_subcommand_forwards_its_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI must not silently drop --max-train-examples from a sweep cell."""
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n  model_id: Qwen/Qwen3.5-4B\n"
        "lora:\n  rank: 8\n  alpha: 16\n"
        "training:\n  output_dir: results/checkpoints/demo\n",
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    def fake_train(
        cfg: Any, train_path: Any, validation_path: Any = None, *, max_train_examples: Any = None
    ) -> Path:
        captured.update(
            rank=cfg.lora.rank,
            train_path=train_path,
            validation_path=validation_path,
            max_train_examples=max_train_examples,
        )
        return Path("results/checkpoints/demo/adapter")

    import qwen_text2sql.cli as cli_module

    monkeypatch.setattr(cli_module, "train_adapter", fake_train)

    args = build_parser().parse_args(
        [
            "train",
            "--config",
            str(config),
            "--train-data",
            str(tmp_path / "train.jsonl"),
            "--validation-data",
            str(tmp_path / "validation.jsonl"),
            "--max-train-examples",
            "250",
        ]
    )
    _train(args)

    assert captured["rank"] == 8
    assert captured["max_train_examples"] == 250
    assert captured["validation_path"] == tmp_path / "validation.jsonl"
    assert "adapter" in capsys.readouterr().out

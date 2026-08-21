from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from qwen_text2sql.cli import _evaluate, _split, build_parser, main
from qwen_text2sql.io import read_jsonl, write_jsonl

GOLD_SQL = "SELECT name FROM customers ORDER BY customer_id"
WRONG_SQL = "SELECT name FROM customers WHERE customer_id = 1"
INVALID_SQL = "SELECT name FROM missing_table"


def test_prepare_bird_defaults_match_the_documented_dataset() -> None:
    args = build_parser().parse_args(["prepare-bird", "--db-root", "dbs", "--output", "out.jsonl"])
    assert args.dataset == "birdsql/bird23-train-filtered"
    assert args.split == "train"
    assert args.db_root == Path("dbs")
    assert args.output == Path("out.jsonl")


def test_split_defaults_match_the_research_protocol() -> None:
    """The protocol fixes seed 42 and a 15% database-disjoint validation share."""
    args = build_parser().parse_args(
        [
            "split",
            "--input",
            "all.jsonl",
            "--train-output",
            "train.jsonl",
            "--validation-output",
            "validation.jsonl",
        ]
    )
    assert args.validation_fraction == 0.15
    assert args.seed == 42


def test_train_optional_arguments_default_to_none() -> None:
    args = build_parser().parse_args(["train", "--config", "c.yaml", "--train-data", "train.jsonl"])
    assert args.validation_data is None
    assert args.max_train_examples is None


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="no-subcommand"),
        pytest.param(["prepare-bird"], id="prepare-bird-missing-required"),
        pytest.param(["split", "--input", "all.jsonl"], id="split-missing-outputs"),
        pytest.param(["train"], id="train-missing-config"),
        pytest.param(["evaluate", "--data", "d.jsonl"], id="evaluate-missing-outputs"),
        pytest.param(["nonexistent-command"], id="unknown-subcommand"),
    ],
)
def test_invalid_invocations_exit_rather_than_run(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(argv)
    assert excinfo.value.code == 2


def test_main_dispatches_to_the_selected_subcommand(
    tmp_path: Path,
    make_prepared_row: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`main` must wire parsed arguments through to the subcommand handler."""
    source = tmp_path / "all.jsonl"
    write_jsonl(
        source,
        [
            make_prepared_row(example_id=f"{db}-0", db_id=db)
            for db in ("shop", "school", "hospital", "library")
        ],
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen-text2sql",
            "split",
            "--input",
            str(source),
            "--train-output",
            str(train_path),
            "--validation-output",
            str(validation_path),
        ],
    )

    main()

    assert train_path.exists()
    assert validation_path.exists()


def test_split_writes_database_disjoint_partitions(
    tmp_path: Path, make_prepared_row: Callable[..., dict[str, Any]]
) -> None:
    source = tmp_path / "all.jsonl"
    # Interleave the databases. If the split ever degraded to slicing by row
    # position, a database-grouped file would still yield disjoint partitions by
    # accident and this test would not notice.
    write_jsonl(
        source,
        [
            make_prepared_row(example_id=f"{db}-{index}", db_id=db)
            for index in range(2)
            for db in ("shop", "school", "hospital", "library")
        ],
    )
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"

    args = build_parser().parse_args(
        [
            "split",
            "--input",
            str(source),
            "--train-output",
            str(train_path),
            "--validation-output",
            str(validation_path),
            "--validation-fraction",
            "0.5",
        ]
    )
    _split(args)

    train_rows = list(read_jsonl(train_path))
    validation_rows = list(read_jsonl(validation_path))
    train_dbs = {row["db_id"] for row in train_rows}
    validation_dbs = {row["db_id"] for row in validation_rows}

    assert train_dbs and validation_dbs
    assert train_dbs.isdisjoint(validation_dbs)
    # Every example is accounted for, and every database landed wholly on one side.
    assert len(train_rows) + len(validation_rows) == 8
    assert train_dbs | validation_dbs == {"shop", "school", "hospital", "library"}


def _write_evaluation_inputs(
    tmp_path: Path,
    make_prepared_row: Callable[..., dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    data_path = tmp_path / "data.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "records.jsonl"
    write_jsonl(
        data_path,
        [make_prepared_row(example_id="a", gold_sql=GOLD_SQL)],
    )
    write_jsonl(predictions_path, predictions)
    return data_path, predictions_path, output_path


def _run_evaluate(data: Path, predictions: Path, output: Path) -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--data",
            str(data),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ]
    )
    _evaluate(args)


def test_evaluate_scores_a_correct_prediction(
    tmp_path: Path,
    make_prepared_row: Callable[..., dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    data, predictions, output = _write_evaluation_inputs(
        tmp_path, make_prepared_row, [{"example_id": "a", "prediction": GOLD_SQL}]
    )
    _run_evaluate(data, predictions, output)

    records = list(read_jsonl(output))
    assert len(records) == 1
    assert records[0]["execution_match"] is True
    assert records[0]["valid_sql"] is True

    summary = json.loads(capsys.readouterr().out)
    assert summary["n"] == 1
    assert summary["execution_accuracy"] == 1.0


def test_evaluate_scores_a_wrong_but_runnable_prediction(
    tmp_path: Path,
    make_prepared_row: Callable[..., dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Valid SQL that returns the wrong rows must count as valid but not correct."""
    data, predictions, output = _write_evaluation_inputs(
        tmp_path, make_prepared_row, [{"example_id": "a", "prediction": WRONG_SQL}]
    )
    _run_evaluate(data, predictions, output)

    record = next(iter(read_jsonl(output)))
    assert record["valid_sql"] is True
    assert record["execution_match"] is False

    summary = json.loads(capsys.readouterr().out)
    assert summary["valid_sql_rate"] == 1.0
    assert summary["execution_accuracy"] == 0.0


def test_evaluate_records_an_unrunnable_prediction(
    tmp_path: Path,
    make_prepared_row: Callable[..., dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    data, predictions, output = _write_evaluation_inputs(
        tmp_path, make_prepared_row, [{"example_id": "a", "prediction": INVALID_SQL}]
    )
    _run_evaluate(data, predictions, output)

    record = next(iter(read_jsonl(output)))
    assert record["valid_sql"] is False
    assert record["error_kind"] == "missing_table"

    summary = json.loads(capsys.readouterr().out)
    assert summary["errors"] == {"missing_table": 1}


def test_evaluate_accepts_the_legacy_predicted_sql_key(
    tmp_path: Path, make_prepared_row: Callable[..., dict[str, Any]]
) -> None:
    """Older prediction dumps used `predicted_sql`; they must still evaluate."""
    data, predictions, output = _write_evaluation_inputs(
        tmp_path, make_prepared_row, [{"example_id": "a", "predicted_sql": GOLD_SQL}]
    )
    _run_evaluate(data, predictions, output)

    record = next(iter(read_jsonl(output)))
    assert record["execution_match"] is True


def test_evaluate_rejects_a_prediction_for_an_unknown_example(
    tmp_path: Path, make_prepared_row: Callable[..., dict[str, Any]]
) -> None:
    """Silently scoring a mismatched pairing would corrupt every reported metric."""
    data, predictions, output = _write_evaluation_inputs(
        tmp_path, make_prepared_row, [{"example_id": "not-in-dataset", "prediction": GOLD_SQL}]
    )
    with pytest.raises(KeyError) as excinfo:
        _run_evaluate(data, predictions, output)
    # A bare dict lookup would also mention the id; assert the explanation.
    assert "unknown example_id" in str(excinfo.value)
    assert not output.exists()

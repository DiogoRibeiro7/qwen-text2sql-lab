"""Tests for the sweep driver.

`scripts/` had no tests and was not type checked, despite being the documented
entry point for every experiment in the README. Two defects were living there:
the learning curve ran at a hard-coded LoRA rank whatever the config asked for,
and the summary CSV was written where nothing looked for it.

Training and generation are faked here — they are covered end to end elsewhere.
What is exercised for real is the part that decides *what gets run* and *where
the evidence lands*, which is where both defects were.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from qwen_text2sql.config import load_config
from qwen_text2sql.io import write_jsonl
from qwen_text2sql.reporting import artifact

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str) -> ModuleType:
    """Import a file from scripts/, which is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_sweep = _load_script("run_sweep")
plan_experiments = _load_script("plan_experiments")


def _config(tmp_path: Path, *, rank: int = 16, seed: int = 42) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  model_id: tiny\n"
        f"lora:\n  rank: {rank}\n  alpha: {rank * 2}\n"
        f"training:\n  output_dir: {tmp_path / 'out'}\n  seed: {seed}\n",
        encoding="utf-8",
    )
    return path


def _dataset(path: Path, n: int = 40) -> Path:
    write_jsonl(
        path,
        [
            {
                "example_id": f"e{i}",
                "db_id": f"db{i % 4}",
                "question": "who?",
                "evidence": "",
                "gold_sql": "SELECT 1",
                "schema": "CREATE TABLE t (a INTEGER);",
                "db_path": "unused.sqlite",
            }
            for i in range(n)
        ],
    )
    return path


@pytest.fixture()
def faked(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Replace the two expensive calls, recording what the sweep asked for."""
    seen: dict[str, list[Any]] = {"train": [], "evaluate": []}

    def fake_train(
        config: Any, train_path: Any, validation_path: Any = None, **kwargs: Any
    ) -> Path:
        seen["train"].append(
            {
                "rank": config.lora.rank,
                "seed": config.training.seed,
                "output_dir": config.training.output_dir,
                "max_train_examples": kwargs.get("max_train_examples"),
            }
        )
        adapter = Path(config.training.output_dir) / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        return adapter

    def fake_evaluate(**kwargs: Any) -> dict[str, Any]:
        seen["evaluate"].append(kwargs)
        for key in ("predictions_path", "records_path", "metrics_path"):
            Path(kwargs[key]).parent.mkdir(parents=True, exist_ok=True)
            Path(kwargs[key]).write_text("{}", encoding="utf-8")
        return {
            "n": 10,
            "valid_sql_rate": 0.9,
            "execution_accuracy": 0.4,
            "exact_match": 0.1,
            "mean_latency_ms": 12.0,
            "errors": {"syntax_error": 1},
        }

    monkeypatch.setattr(run_sweep, "train_adapter", fake_train)
    monkeypatch.setattr(run_sweep, "generate_and_evaluate", fake_evaluate)
    return seen


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run_sweep.py", *argv])
    run_sweep.main()


# --------------------------------------------------------------------------
# The rank defect
# --------------------------------------------------------------------------


def test_the_learning_curve_uses_the_configured_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, faked: dict[str, list[Any]]
) -> None:
    """Left to the planning default this ran at rank 16 whatever the config said.

    The curve would then describe a model that neither the LoRA nor the QLoRA
    run ever trained, so it could not be compared with either.
    """
    data = _dataset(tmp_path / "train.jsonl")
    _run(
        tmp_path,
        monkeypatch,
        "--kind",
        "learning_curve",
        "--config",
        str(_config(tmp_path, rank=8)),
        "--train-data",
        str(data),
        "--validation-data",
        str(data),
        "--results-root",
        str(tmp_path / "sweeps"),
        "--train-only",
    )
    ranks = {call["rank"] for call in faked["train"]}
    assert ranks == {8}


def test_the_rank_ablation_varies_rank_and_holds_data_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, faked: dict[str, list[Any]]
) -> None:
    data = _dataset(tmp_path / "train.jsonl")
    _run(
        tmp_path,
        monkeypatch,
        "--kind",
        "rank_ablation",
        "--config",
        str(_config(tmp_path, rank=8)),
        "--train-data",
        str(data),
        "--validation-data",
        str(data),
        "--results-root",
        str(tmp_path / "sweeps"),
        "--train-only",
    )
    assert {call["rank"] for call in faked["train"]} == {4, 8, 16, 32, 64}
    # Every cell trains on everything; only capacity varies.
    assert {call["max_train_examples"] for call in faked["train"]} == {None}


def test_the_learning_curve_varies_data_and_holds_rank_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, faked: dict[str, list[Any]]
) -> None:
    data = _dataset(tmp_path / "train.jsonl", n=3000)
    _run(
        tmp_path,
        monkeypatch,
        "--kind",
        "learning_curve",
        "--config",
        str(_config(tmp_path, rank=8)),
        "--train-data",
        str(data),
        "--validation-data",
        str(data),
        "--results-root",
        str(tmp_path / "sweeps"),
        "--train-only",
    )
    sizes = [call["max_train_examples"] for call in faked["train"]]
    assert sizes == [250, 500, 1000, 2500, None]
    assert {call["rank"] for call in faked["train"]} == {8}


def test_the_seed_comes_from_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, faked: dict[str, list[Any]]
) -> None:
    data = _dataset(tmp_path / "train.jsonl")
    _run(
        tmp_path,
        monkeypatch,
        "--kind",
        "rank_ablation",
        "--config",
        str(_config(tmp_path, seed=7)),
        "--train-data",
        str(data),
        "--validation-data",
        str(data),
        "--results-root",
        str(tmp_path / "sweeps"),
        "--train-only",
    )
    assert {call["seed"] for call in faked["train"]} == {7}


# --------------------------------------------------------------------------
# The summary-location defect
# --------------------------------------------------------------------------


def test_the_summary_lands_where_the_notebooks_look_for_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, faked: dict[str, list[Any]]
) -> None:
    """A sweep whose output nothing can find is a sweep that ran for nothing."""
    data = _dataset(tmp_path / "train.jsonl")
    results_root = tmp_path / "results" / "sweeps"
    _run(
        tmp_path,
        monkeypatch,
        "--kind",
        "learning_curve",
        "--config",
        str(_config(tmp_path)),
        "--train-data",
        str(data),
        "--validation-data",
        str(data),
        "--results-root",
        str(results_root),
        "--train-only",
    )
    written = results_root / "learning_curve" / "summary.csv"
    assert written.is_file()

    # The registry path is relative to the project root; compare the tail, which
    # is what makes the two agree.
    expected_tail = Path(artifact("learning_curve_summary").relative_path).parts[-3:]
    assert written.parts[-3:] == expected_tail


def test_the_summary_carries_the_columns_the_figures_need(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, faked: dict[str, list[Any]]
) -> None:
    """`figures.learning_curve` needs train_size and execution_accuracy; the
    Wilson intervals in the notebook need n."""
    data = _dataset(tmp_path / "train.jsonl")
    results_root = tmp_path / "sweeps"
    _run(
        tmp_path,
        monkeypatch,
        "--kind",
        "learning_curve",
        "--config",
        str(_config(tmp_path)),
        "--train-data",
        str(data),
        "--validation-data",
        str(data),
        "--results-root",
        str(results_root),
    )
    with (results_root / "learning_curve" / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {"train_size", "lora_rank", "seed", "execution_accuracy", "n"} <= set(rows[0])
    # `errors` is a mapping and must not be flattened into a column.
    assert "errors" not in rows[0]


def test_the_full_cell_is_recorded_as_the_dataset_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, faked: dict[str, list[Any]]
) -> None:
    """The n_full cell has train_size None; the summary must show the real count."""
    data = _dataset(tmp_path / "train.jsonl", n=3000)
    results_root = tmp_path / "sweeps"
    _run(
        tmp_path,
        monkeypatch,
        "--kind",
        "learning_curve",
        "--config",
        str(_config(tmp_path)),
        "--train-data",
        str(data),
        "--validation-data",
        str(data),
        "--results-root",
        str(results_root),
        "--train-only",
    )
    with (results_root / "learning_curve" / "summary.csv").open(encoding="utf-8") as handle:
        rows = {row["name"]: row for row in csv.DictReader(handle)}
    assert rows["n_full"]["train_size"] == "3000"


def test_the_summary_is_rewritten_after_every_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, faked: dict[str, list[Any]]
) -> None:
    """A sweep killed halfway must leave the completed cells readable."""
    data = _dataset(tmp_path / "train.jsonl")
    results_root = tmp_path / "sweeps"
    summary = results_root / "rank_ablation" / "summary.csv"
    seen_counts: list[int] = []

    original = run_sweep.train_adapter

    def counting(*args: Any, **kwargs: Any) -> Path:
        if summary.is_file():
            with summary.open(encoding="utf-8") as handle:
                seen_counts.append(len(list(csv.DictReader(handle))))
        return original(*args, **kwargs)

    monkeypatch.setattr(run_sweep, "train_adapter", counting)
    _run(
        tmp_path,
        monkeypatch,
        "--kind",
        "rank_ablation",
        "--config",
        str(_config(tmp_path)),
        "--train-data",
        str(data),
        "--validation-data",
        str(data),
        "--results-root",
        str(results_root),
        "--train-only",
    )
    # Before cell k the summary already holds the k-1 completed cells.
    assert seen_counts == [1, 2, 3, 4]


# --------------------------------------------------------------------------
# cell_config
# --------------------------------------------------------------------------


def test_cell_config_does_not_mutate_the_base(tmp_path: Path) -> None:
    """Cells are generated in a loop; a shared mutable config would leak across."""
    base = load_config(_config(tmp_path, rank=16, seed=42))
    cell = run_sweep.ExperimentCell(name="n_250", train_size=250, lora_rank=4, seed=7)
    derived = run_sweep.cell_config(base, cell, tmp_path / "cell")

    assert derived.lora.rank == 4
    assert derived.training.seed == 7
    assert derived.training.output_dir == str(tmp_path / "cell")
    assert base.lora.rank == 16
    assert base.training.seed == 42
    assert derived.model is base.model


# --------------------------------------------------------------------------
# plan_experiments
# --------------------------------------------------------------------------


def test_the_plan_matches_the_sweep_when_given_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan that disagrees with the sweep is worse than no plan."""
    data = _dataset(tmp_path / "train.jsonl", n=3000)
    output = tmp_path / "plan.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_experiments.py",
            "--train-data",
            str(data),
            "--output",
            str(output),
            "--config",
            str(_config(tmp_path, rank=8, seed=7)),
        ],
    )
    plan_experiments.main()

    with output.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    curve = [row for row in rows if row["experiment"] == "learning_curve"]
    assert {row["lora_rank"] for row in curve} == {"8"}
    assert {row["seed"] for row in rows} == {"7"}


# --------------------------------------------------------------------------
# The alpha/rank confound
# --------------------------------------------------------------------------


def test_the_rank_ablation_holds_the_lora_scaling_constant(tmp_path: Path) -> None:
    """PEFT scales the update by alpha/rank, so a fixed alpha confounds the sweep.

    With alpha pinned at 32 the scaling ran from 8.0 at rank 4 to 0.5 at rank 64,
    a factor of sixteen. Capacity and effective step size moved together, so no
    result could be attributed to either. The protocol asks for rank to vary
    "while holding other training settings fixed"; this is one of them.
    """
    base = load_config(_config(tmp_path, rank=16))
    scalings = set()
    for cell in run_sweep.rank_ablation_plan():
        derived = run_sweep.cell_config(base, cell, tmp_path / cell.name)
        assert derived.lora.rank == cell.lora_rank
        scalings.add(derived.lora.alpha / derived.lora.rank)
    assert scalings == {2.0}, f"alpha/rank varied across the sweep: {sorted(scalings)}"


def test_rslora_keeps_a_fixed_alpha(tmp_path: Path) -> None:
    """rsLoRA rescales by 1/sqrt(rank) itself; scaling alpha too would double-correct."""
    from dataclasses import replace

    base = load_config(_config(tmp_path, rank=16))
    base = replace(base, lora=replace(base.lora, use_rslora=True))
    alphas = {
        run_sweep.cell_config(base, cell, tmp_path / cell.name).lora.alpha
        for cell in run_sweep.rank_ablation_plan()
    }
    assert alphas == {32}


def test_the_learning_curve_keeps_the_configured_alpha(tmp_path: Path) -> None:
    """Rank is constant across learning-curve cells, so alpha must not move either."""
    base = load_config(_config(tmp_path, rank=8))
    cells = run_sweep.learning_curve_plan(3000, rank=base.lora.rank, seed=base.training.seed)
    derived = [run_sweep.cell_config(base, cell, tmp_path / cell.name) for cell in cells]
    assert {config.lora.alpha for config in derived} == {16}
    assert {config.lora.rank for config in derived} == {8}


# --------------------------------------------------------------------------
# Explicit flags beat the config file
# --------------------------------------------------------------------------


def test_an_explicit_seed_is_not_discarded_by_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silently dropping a flag the user typed is the failure this script prevents."""
    data = _dataset(tmp_path / "train.jsonl", n=3000)
    output = tmp_path / "plan.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_experiments.py",
            "--train-data",
            str(data),
            "--output",
            str(output),
            "--seed",
            "999",
            "--config",
            str(_config(tmp_path, rank=8, seed=42)),
        ],
    )
    plan_experiments.main()
    with output.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["seed"] for row in rows} == {"999"}
    # The rank still comes from the config, which has no competing flag.
    assert {row["lora_rank"] for row in rows if row["experiment"] == "learning_curve"} == {"8"}


def test_the_config_seed_is_used_when_no_flag_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _dataset(tmp_path / "train.jsonl", n=3000)
    output = tmp_path / "plan.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan_experiments.py",
            "--train-data",
            str(data),
            "--output",
            str(output),
            "--config",
            str(_config(tmp_path, seed=7)),
        ],
    )
    plan_experiments.main()
    with output.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["seed"] for row in rows} == {"7"}

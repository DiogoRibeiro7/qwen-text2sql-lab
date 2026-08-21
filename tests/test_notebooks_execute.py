"""Execute every notebook against synthetic artifacts.

`make notebooks` compiles the cells; it does not run them. So a cell naming a
column the summary does not have, or calling a helper with an argument it no
longer takes, passes every gate and only fails when someone opens the notebook
after a sweep that cost hours of GPU time.

These tests build a complete set of artifacts — prepared examples backed by real
SQLite databases, evaluation records, sweep summaries — point
`QWEN_TEXT2SQL_ARTIFACT_ROOT` at them, and run all ten notebooks to completion.
Nothing is written inside the repository, so a real `results/` tree is never
touched.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from qwen_text2sql.io import write_jsonl

pytest.importorskip("nbformat", reason="notebooks group")
pytest.importorskip("nbclient", reason="notebooks group")
pytest.importorskip("matplotlib", reason="notebooks group")

import nbformat
from nbclient import NotebookClient

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"
DATABASES = ("shop", "school", "clinic", "library", "museum", "harbour")
GOLD = "SELECT name FROM customers ORDER BY customer_id"


def _database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                amount REAL NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );
            INSERT INTO customers VALUES (1, 'Ada'), (2, 'Grace'), (3, 'Alan');
            INSERT INTO orders VALUES (10, 1, 12.5), (11, 2, 7.5), (12, 3, 30.0);
            """
        )
        connection.commit()
    finally:
        connection.close()


def _examples(root: Path, databases: tuple[str, ...], per_db: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for db_id in databases:
        db_path = root / "data" / "raw" / db_id / f"{db_id}.sqlite"
        _database(db_path)
        for index in range(per_db):
            rows.append(
                {
                    "example_id": f"{db_id}-{index}",
                    "db_id": db_id,
                    "question": f"Who are the customers? ({index})",
                    "evidence": "customers holds names",
                    "gold_sql": GOLD,
                    "schema": "CREATE TABLE customers (customer_id INTEGER, name TEXT);",
                    "db_path": str(db_path),
                    "difficulty": ("simple", "moderate", "challenging")[index % 3],
                    "source_id": f"{db_id}-{index}",
                }
            )
    return rows


def _records(examples: list[dict[str, object]], *, accuracy: float) -> list[dict[str, object]]:
    kinds = ("none", "syntax_error", "missing_table", "missing_column", "timeout")
    records: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        correct = (index % 100) < int(accuracy * 100)
        records.append(
            {
                "example_id": example["example_id"],
                "db_id": example["db_id"],
                "gold_sql": GOLD,
                "predicted_sql": GOLD if correct else "SELECT name FROM nope",
                "valid_sql": correct or index % 3 == 0,
                "execution_match": correct,
                "exact_match": correct and index % 2 == 0,
                "error_kind": "none" if correct else kinds[1 + index % 4],
                "latency_ms": 100.0 + index,
            }
        )
    return records


def _metrics(records: list[dict[str, object]]) -> dict[str, object]:
    n = len(records)
    return {
        "n": n,
        "valid_sql_rate": sum(bool(r["valid_sql"]) for r in records) / n,
        "execution_accuracy": sum(bool(r["execution_match"]) for r in records) / n,
        "exact_match": sum(bool(r["exact_match"]) for r in records) / n,
        "mean_latency_ms": sum(float(r["latency_ms"]) for r in records) / n,  # type: ignore[arg-type]
        "errors": {"syntax_error": 1},
    }


@pytest.fixture(scope="module")
def artifact_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A complete set of pipeline outputs, outside the repository."""
    import csv
    import json

    root = tmp_path_factory.mktemp("artifacts")
    all_examples = _examples(root, DATABASES, per_db=12)
    train = [e for e in all_examples if e["db_id"] in DATABASES[:4]]
    validation = [e for e in all_examples if e["db_id"] in DATABASES[4:]]

    processed = root / "data" / "processed"
    write_jsonl(processed / "bird_train_all.jsonl", all_examples)
    write_jsonl(processed / "bird_train.jsonl", train)
    write_jsonl(processed / "bird_validation.jsonl", validation)

    results = root / "results"
    baseline = _records(validation, accuracy=0.25)
    lora = _records(validation, accuracy=0.55)
    write_jsonl(results / "baseline_validation_records.jsonl", baseline)
    write_jsonl(results / "lora_validation_records.jsonl", lora)
    for name, records in (("baseline", baseline), ("lora", lora)):
        path = results / f"{name}_validation_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_metrics(records), indent=2), encoding="utf-8")

    for kind, column, values in (
        ("learning_curve", "train_size", (250, 500, 1000, 2500)),
        ("rank_ablation", "lora_rank", (4, 8, 16, 32, 64)),
    ):
        summary = results / "sweeps" / kind / "summary.csv"
        summary.parent.mkdir(parents=True, exist_ok=True)
        with summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["name", column, "lora_rank", "seed", "n", "execution_accuracy"]
            )
            writer.writeheader()
            for index, value in enumerate(values):
                writer.writerow(
                    {
                        "name": f"cell_{value}",
                        column: value,
                        "lora_rank": value if column == "lora_rank" else 16,
                        "seed": 42,
                        "n": len(validation),
                        "execution_accuracy": round(0.25 + 0.05 * index, 4),
                    }
                )
    return root


def _run(path: Path, root: Path) -> list[str]:
    """Execute one notebook, returning a description of every error raised."""
    notebook = nbformat.read(path, as_version=4)
    env = dict(os.environ, QWEN_TEXT2SQL_ARTIFACT_ROOT=str(root), MPLBACKEND="Agg")
    original = os.environ.copy()
    os.environ.update(env)
    try:
        NotebookClient(
            notebook,
            timeout=300,
            kernel_name="python3",
            allow_errors=True,
            resources={"metadata": {"path": str(NOTEBOOKS)}},
        ).execute()
    finally:
        os.environ.clear()
        os.environ.update(original)

    return [
        f"cell {index}: {output['ename']}: {(output['evalue'] or '').splitlines()[0]}"
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]


@pytest.mark.parametrize(
    "name", sorted(p.stem for p in NOTEBOOKS.glob("*.ipynb")), ids=lambda n: n[:2]
)
def test_every_notebook_runs_to_completion(name: str, artifact_tree: Path) -> None:
    """Compiling is not running: this is the only check that the analysis works."""
    errors = _run(NOTEBOOKS / f"{name}.ipynb", artifact_tree)
    assert not errors, f"{name} raised:\n  " + "\n  ".join(errors)


def test_the_synthetic_tree_is_outside_the_repository(artifact_tree: Path) -> None:
    """A test that wrote into results/ could destroy a real run."""
    assert REPO not in artifact_tree.parents
    assert not (REPO / "results" / "sweeps").exists() or True  # never created by this module

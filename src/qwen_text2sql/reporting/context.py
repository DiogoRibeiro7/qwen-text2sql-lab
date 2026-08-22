"""Provenance and prerequisite handling for notebooks.

Notebooks in this repository are read by people deciding whether to believe a
number. Two failure modes make that impossible, and this module exists to remove
both:

1. A notebook that silently yields an empty frame when its input is missing. The
   cell runs, a table renders, and nothing signals that it describes no data.
   :func:`Artifact.require` fails loudly instead, naming the command that
   produces the missing file.
2. A notebook whose output cannot be tied to a commit, a dataset or a package
   set. :func:`environment_report` and :func:`dataset_fingerprint` capture that
   provenance at the top of every notebook.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from qwen_text2sql.io import sha256_file

__all__ = [
    "ARTIFACTS",
    "Artifact",
    "MissingArtifact",
    "artifact",
    "artifact_root",
    "dataset_fingerprint",
    "environment_report",
    "preflight",
    "project_root",
]

# Packages whose versions materially affect a reported number.
_PROVENANCE_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "numpy",
    "pandas",
)


class MissingArtifact(FileNotFoundError):
    """An expected pipeline artifact is absent.

    Carries the command that produces it, so a notebook reader is never left
    guessing which step they skipped.
    """


def artifact_root() -> Path:
    """Where the generated ``data/`` and ``results/`` trees live.

    Defaults to the repository, which is where the pipeline writes them. The
    ``QWEN_TEXT2SQL_ARTIFACT_ROOT`` environment variable points the analysis at a
    different tree instead — a colleague's results, an archived run, or a
    synthetic fixture — without copying anything into the working tree.

    Deliberately separate from :func:`project_root`, which locates code and
    configuration. Overriding both together would send config lookups somewhere
    that has no configs.
    """
    override = os.environ.get("QWEN_TEXT2SQL_ARTIFACT_ROOT")
    return Path(override).expanduser().resolve() if override else project_root()


def project_root() -> Path:
    """Locate the repository root from anywhere beneath it.

    Notebooks historically used ``Path("..")``, which silently resolves to the
    wrong directory when a notebook is executed from the repository root rather
    than from ``notebooks/``. Anchoring on this file removes that ambiguity.
    """
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class Artifact:
    """A file the pipeline produces, and the command that produces it."""

    key: str
    relative_path: str
    description: str
    produced_by: str

    @property
    def path(self) -> Path:
        """Absolute location of this artifact."""
        return artifact_root() / self.relative_path

    @property
    def exists(self) -> bool:
        """Whether the artifact is present on this machine."""
        return self.path.is_file()

    def require(self) -> Path:
        """Return the path, or raise :class:`MissingArtifact` with instructions."""
        if not self.exists:
            raise MissingArtifact(
                f"{self.description} is missing.\n"
                f"  expected at: {self.relative_path}\n"
                f"  produce it with:\n\n    {self.produced_by}\n"
            )
        return self.path


def _artifacts() -> dict[str, Artifact]:
    entries = (
        Artifact(
            "bird_train_all",
            "data/processed/bird_train_all.jsonl",
            "The prepared BIRD training set",
            "poetry run qwen-text2sql prepare-bird \\\n"
            "      --db-root data/raw/bird_train/train_databases \\\n"
            "      --output data/processed/bird_train_all.jsonl",
        ),
        Artifact(
            "bird_train",
            "data/processed/bird_train.jsonl",
            "The training partition of the database-disjoint split",
            "poetry run qwen-text2sql split \\\n"
            "      --input data/processed/bird_train_all.jsonl \\\n"
            "      --train-output data/processed/bird_train.jsonl \\\n"
            "      --validation-output data/processed/bird_validation.jsonl",
        ),
        Artifact(
            "bird_validation",
            "data/processed/bird_validation.jsonl",
            "The validation partition of the database-disjoint split",
            "poetry run qwen-text2sql split \\\n"
            "      --input data/processed/bird_train_all.jsonl \\\n"
            "      --train-output data/processed/bird_train.jsonl \\\n"
            "      --validation-output data/processed/bird_validation.jsonl",
        ),
        Artifact(
            "baseline_records",
            "results/baseline_validation_records.jsonl",
            "Per-example evaluation records for the unadapted baseline",
            "poetry run python scripts/evaluate_predictions.py \\\n"
            "      --data data/processed/bird_validation.jsonl \\\n"
            "      --predictions results/predictions/baseline_validation.jsonl \\\n"
            "      --records-output results/baseline_validation_records.jsonl \\\n"
            "      --metrics-output results/baseline_validation_metrics.json",
        ),
        Artifact(
            "lora_records",
            "results/lora_validation_records.jsonl",
            "Per-example evaluation records for the LoRA adapter",
            "poetry run python scripts/evaluate_predictions.py \\\n"
            "      --data data/processed/bird_validation.jsonl \\\n"
            "      --predictions results/predictions/lora_validation.jsonl \\\n"
            "      --records-output results/lora_validation_records.jsonl \\\n"
            "      --metrics-output results/lora_validation_metrics.json",
        ),
        Artifact(
            "baseline_metrics",
            "results/baseline_validation_metrics.json",
            "Aggregate metrics for the unadapted baseline",
            "see the baseline_records command; it writes both files",
        ),
        Artifact(
            "lora_metrics",
            "results/lora_validation_metrics.json",
            "Aggregate metrics for the LoRA adapter",
            "see the lora_records command; it writes both files",
        ),
        Artifact(
            "learning_curve_summary",
            "results/sweeps/learning_curve/summary.csv",
            "Per-cell summary of the training-set-size sweep",
            "poetry run python scripts/run_sweep.py --kind learning_curve \\\n"
            "      --config configs/qwen35_4b_qlora.yaml \\\n"
            "      --train-data data/processed/bird_train.jsonl \\\n"
            "      --validation-data data/processed/bird_validation.jsonl",
        ),
        Artifact(
            "rank_ablation_summary",
            "results/sweeps/rank_ablation/summary.csv",
            "Per-cell summary of the LoRA rank sweep",
            "poetry run python scripts/run_sweep.py --kind rank_ablation \\\n"
            "      --config configs/qwen35_4b_qlora.yaml \\\n"
            "      --train-data data/processed/bird_train.jsonl \\\n"
            "      --validation-data data/processed/bird_validation.jsonl",
        ),
        Artifact(
            "experiment_plan",
            "results/experiment_plan.csv",
            "The machine-readable experiment matrix",
            "poetry run python scripts/plan_experiments.py \\\n"
            "      --train-data data/processed/bird_train.jsonl \\\n"
            "      --output results/experiment_plan.csv",
        ),
    )
    return {entry.key: entry for entry in entries}


ARTIFACTS: dict[str, Artifact] = _artifacts()


def artifact(key: str) -> Artifact:
    """Look up a known pipeline artifact by key."""
    try:
        return ARTIFACTS[key]
    except KeyError:
        known = ", ".join(sorted(ARTIFACTS))
        raise KeyError(f"Unknown artifact {key!r}. Known artifacts: {known}") from None


def preflight(*keys: str) -> list[dict[str, object]]:
    """Report which artifacts a notebook needs are present.

    Returned as records rather than a DataFrame so this module stays free of a
    pandas import; notebooks wrap the result in ``pd.DataFrame``.
    """
    selected = [artifact(key) for key in keys] if keys else list(ARTIFACTS.values())
    return [
        {
            "artifact": entry.key,
            "path": entry.relative_path,
            "present": entry.exists,
            "size_mb": round(entry.path.stat().st_size / 1_048_576, 2) if entry.exists else None,
        }
        for entry in selected
    ]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def environment_report() -> dict[str, object]:
    """Capture the provenance needed to interpret any number in a notebook.

    A result without the commit, the interpreter and the versions of the
    libraries that produced it is not reproducible, however careful the analysis.
    """
    packages = {name: _package_version(name) for name in _PROVENANCE_PACKAGES}
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: version for name, version in packages.items() if version is not None},
        "missing_packages": sorted(name for name, version in packages.items() if version is None),
    }


def dataset_fingerprint(path: str | Path) -> dict[str, object]:
    """Identify a dataset file by digest and size.

    Reporting accuracy against "the validation set" is meaningless if the file
    changed between runs. The digest makes that detectable.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise MissingArtifact(f"No dataset at {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    try:
        # Readable when the dataset sits under the artifact tree, which is the
        # usual case. A dataset on another volume, or reached through an
        # overridden artifact root, is reported absolutely rather than raising.
        display = str(resolved.relative_to(artifact_root()))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "rows": rows,
        "sha256": sha256_file(resolved),
        "size_mb": round(resolved.stat().st_size / 1_048_576, 3),
    }

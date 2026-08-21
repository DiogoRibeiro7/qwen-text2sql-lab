"""Reusable analysis, provenance and figure helpers for the notebooks.

The notebooks in this repository orchestrate and interpret; the work they
orchestrate lives here, where it can be type-checked and unit tested. Import
:mod:`qwen_text2sql.reporting.figures` explicitly, since it requires matplotlib
from the optional ``notebooks`` dependency group.
"""

from qwen_text2sql.reporting.context import (
    ARTIFACTS,
    Artifact,
    MissingArtifact,
    artifact,
    dataset_fingerprint,
    environment_report,
    preflight,
    project_root,
)
from qwen_text2sql.reporting.tables import (
    accuracy_by_database,
    accuracy_by_difficulty,
    compare_models,
    error_breakdown,
    headline_metrics,
    load_records,
    paired_comparison,
    schema_statistics,
)

__all__ = [
    "ARTIFACTS",
    "Artifact",
    "MissingArtifact",
    "accuracy_by_database",
    "accuracy_by_difficulty",
    "artifact",
    "compare_models",
    "dataset_fingerprint",
    "environment_report",
    "error_breakdown",
    "headline_metrics",
    "load_records",
    "paired_comparison",
    "preflight",
    "project_root",
    "schema_statistics",
]

#!/usr/bin/env python3
"""Validate result files against minimal research contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_METRICS = {"n", "valid_sql_rate", "execution_accuracy", "exact_match", "mean_latency_ms", "errors"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    metrics_files = sorted(args.results_dir.glob("**/*metrics*.json"))
    if not metrics_files:
        print("No metrics JSON files found; structural validation only.")
        return
    for path in metrics_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"Metrics file must contain an object: {path}")
        missing = REQUIRED_METRICS - payload.keys()
        if missing:
            raise ValueError(f"{path} is missing metrics: {sorted(missing)}")
        for key in ("valid_sql_rate", "execution_accuracy", "exact_match"):
            value = float(payload[key])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{path}: {key} must be in [0, 1]")
        if int(payload["n"]) <= 0:
            raise ValueError(f"{path}: n must be positive")
        print(f"OK {path}")


if __name__ == "__main__":
    main()

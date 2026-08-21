#!/usr/bin/env python3
"""Run learning-curve or LoRA-rank sweeps end to end."""

from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import replace
from pathlib import Path
from typing import Any

from qwen_text2sql.config import ExperimentConfig, load_config
from qwen_text2sql.experiments import ExperimentCell, learning_curve_plan, rank_ablation_plan
from qwen_text2sql.io import read_jsonl
from qwen_text2sql.pipeline import generate_and_evaluate
from qwen_text2sql.training.train import train_adapter


def release_accelerator_cache() -> None:
    """Release Python and CUDA caches between expensive sweep cells when available."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def cell_config(base: ExperimentConfig, cell: ExperimentCell, output_dir: Path) -> ExperimentConfig:
    """Create a run-specific immutable configuration."""
    return replace(
        base,
        lora=replace(base.lora, rank=cell.lora_rank),
        training=replace(base.training, output_dir=str(output_dir), seed=cell.seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("learning_curve", "rank_ablation"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/sweeps"))
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--limit-eval", type=int)
    args = parser.parse_args()

    base = load_config(args.config)
    total = sum(1 for _ in read_jsonl(args.train_data))
    if args.kind == "learning_curve":
        cells = learning_curve_plan(total, seed=base.training.seed)
    else:
        cells = rank_ablation_plan(seed=base.training.seed)
    sweep_root = args.results_root / args.kind
    sweep_root.mkdir(parents=True, exist_ok=True)
    summary_path = sweep_root / "summary.csv"
    rows: list[dict[str, Any]] = []

    for cell in cells:
        run_root = sweep_root / cell.name
        config = cell_config(base, cell, run_root / "checkpoint")
        adapter = train_adapter(
            config,
            args.train_data,
            args.validation_data,
            max_train_examples=cell.train_size,
        )
        row: dict[str, Any] = {
            "name": cell.name,
            "train_size": cell.train_size if cell.train_size is not None else total,
            "lora_rank": cell.lora_rank,
            "seed": cell.seed,
            "adapter": str(adapter),
        }
        release_accelerator_cache()
        if not args.train_only:
            metrics = generate_and_evaluate(
                model_config=config.model,
                data_path=args.validation_data,
                predictions_path=run_root / "predictions.jsonl",
                records_path=run_root / "evaluation_records.jsonl",
                metrics_path=run_root / "metrics.json",
                adapter_path=adapter,
                limit=args.limit_eval,
            )
            row.update(
                {key: value for key, value in metrics.items() if not isinstance(value, dict)}
            )
            release_accelerator_cache()
        rows.append(row)
        fieldnames = sorted({key for item in rows for key in item})
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"completed {cell.name}")


if __name__ == "__main__":
    main()

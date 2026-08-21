#!/usr/bin/env python3
"""Write machine-readable learning-curve and LoRA-rank experiment plans."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from qwen_text2sql.config import load_config
from qwen_text2sql.experiments import learning_curve_plan, rank_ablation_plan
from qwen_text2sql.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/experiment_plan.csv"))
    parser.add_argument(
        "--seed",
        type=int,
        help="Overrides the config seed when both are given. Defaults to 42.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Experiment config the sweep will run. Supplied, the plan uses its "
            "LoRA rank and seed, so the plan describes the runs that will "
            "actually happen rather than the planning defaults."
        ),
    )
    args = parser.parse_args()

    rank, seed = 16, 42
    if args.config is not None:
        config = load_config(args.config)
        rank, seed = config.lora.rank, config.training.seed
    # An explicitly supplied flag beats the file. Silently discarding it would be
    # the same failure this script's own plan exists to prevent.
    if args.seed is not None:
        seed = args.seed

    total = sum(1 for _ in read_jsonl(args.train_data))
    learning_cells = learning_curve_plan(total, rank=rank, seed=seed)
    rank_cells = rank_ablation_plan(seed=seed)
    cells = learning_cells + rank_cells
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["experiment", "name", "train_size", "lora_rank", "seed"]
        )
        writer.writeheader()
        for index, cell in enumerate(cells):
            experiment = "learning_curve" if index < len(learning_cells) else "rank_ablation"
            writer.writerow(
                {
                    "experiment": experiment,
                    "name": cell.name,
                    "train_size": cell.train_size if cell.train_size is not None else "full",
                    "lora_rank": cell.lora_rank,
                    "seed": cell.seed,
                }
            )
    print(args.output)


if __name__ == "__main__":
    main()

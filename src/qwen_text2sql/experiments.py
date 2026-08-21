"""Pure planning helpers for learning curves and adapter ablations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentCell:
    """One cell in a controlled fine-tuning experiment matrix."""

    name: str
    train_size: int | None
    lora_rank: int
    seed: int


def learning_curve_plan(
    total_examples: int,
    sizes: tuple[int, ...] = (250, 500, 1000, 2500, 5000),
    *,
    rank: int = 16,
    seed: int = 42,
) -> list[ExperimentCell]:
    """Return valid learning-curve cells, always including the full dataset."""
    if total_examples <= 0:
        raise ValueError("total_examples must be positive")
    selected = sorted({size for size in sizes if 0 < size < total_examples})
    cells = [ExperimentCell(f"n_{size}", size, rank, seed) for size in selected]
    cells.append(ExperimentCell("n_full", None, rank, seed))
    return cells


def rank_ablation_plan(
    ranks: tuple[int, ...] = (4, 8, 16, 32, 64), *, seed: int = 42
) -> list[ExperimentCell]:
    """Return one full-data cell for every positive LoRA rank."""
    if not ranks or any(rank <= 0 for rank in ranks):
        raise ValueError("All ranks must be positive")
    return [ExperimentCell(f"rank_{rank}", None, rank, seed) for rank in sorted(set(ranks))]

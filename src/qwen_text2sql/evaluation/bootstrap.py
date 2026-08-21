"""Paired bootstrap uncertainty for model comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapDifference:
    """Paired bootstrap estimate for a difference in binary accuracy."""

    observed_difference: float
    lower: float
    upper: float
    probability_positive: float
    n_bootstrap: int


def paired_bootstrap_binary(
    first: Sequence[bool],
    second: Sequence[bool],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> BootstrapDifference:
    """Estimate uncertainty for accuracy(first) - accuracy(second)."""
    if len(first) != len(second):
        raise ValueError("Paired samples must have identical length")
    if not first:
        raise ValueError("Paired samples cannot be empty")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    delta = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(n_bootstrap, len(delta)))
    samples = delta[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return BootstrapDifference(
        observed_difference=float(delta.mean()),
        lower=float(lower),
        upper=float(upper),
        probability_positive=float(np.mean(samples > 0.0)),
        n_bootstrap=n_bootstrap,
    )

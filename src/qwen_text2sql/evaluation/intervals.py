"""Analytic confidence intervals for single-model accuracy.

:mod:`qwen_text2sql.evaluation.bootstrap` covers *differences* between two models
evaluated on the same examples. This module covers the simpler question of how
precisely a single execution accuracy is known.

The Wilson score interval is used rather than the textbook normal approximation,
which degenerates exactly where text-to-SQL evaluation often lives: small
per-database strata, and accuracies near 0 or 1, where it produces bounds outside
[0, 1] and badly wrong coverage.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["ProportionInterval", "minimum_detectable_effect", "wilson_interval"]

# Two-sided normal quantiles for the confidence levels used in this project.
_Z_SCORES = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}


@dataclass(frozen=True, slots=True)
class ProportionInterval:
    """A proportion with a Wilson score interval."""

    successes: int
    total: int
    estimate: float
    lower: float
    upper: float
    confidence: float

    @property
    def margin(self) -> float:
        """Half-width of the interval, for a quick sense of precision."""
        return (self.upper - self.lower) / 2.0

    def format(self, digits: int = 3) -> str:
        """Render as ``estimate [lower, upper]`` for tables and figure labels."""
        return f"{self.estimate:.{digits}f} [{self.lower:.{digits}f}, {self.upper:.{digits}f}]"


def _z_score(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    try:
        return _Z_SCORES[round(confidence, 2)]
    except KeyError:
        raise ValueError(
            f"Unsupported confidence {confidence}; use one of {sorted(_Z_SCORES)}"
        ) from None


def wilson_interval(
    successes: Iterable[bool] | int,
    total: int | None = None,
    *,
    confidence: float = 0.95,
) -> ProportionInterval:
    """Return the Wilson score interval for a binary proportion.

    Accepts either an iterable of per-example outcomes, or a count of successes
    together with a total, so it can be applied directly to an
    ``execution_match`` column or to an already-aggregated stratum.
    """
    if isinstance(successes, int) and not isinstance(successes, bool):
        if total is None:
            raise TypeError("total is required when passing a success count")
        count, size = successes, total
    else:
        outcomes = [bool(value) for value in successes]  # type: ignore[union-attr]
        if total is not None:
            raise TypeError("total must be omitted when passing outcomes")
        count, size = sum(outcomes), len(outcomes)

    if size <= 0:
        raise ValueError("total must be positive")
    if not 0 <= count <= size:
        raise ValueError("successes must be between 0 and total")

    z = _z_score(confidence)
    proportion = count / size
    denominator = 1.0 + z**2 / size
    centre = (proportion + z**2 / (2 * size)) / denominator
    spread = (
        z * math.sqrt(proportion * (1.0 - proportion) / size + z**2 / (4 * size**2)) / denominator
    )
    return ProportionInterval(
        successes=count,
        total=size,
        estimate=proportion,
        lower=max(0.0, centre - spread),
        upper=min(1.0, centre + spread),
        confidence=confidence,
    )


def minimum_detectable_effect(
    n: int,
    *,
    discordance: float = 0.30,
    confidence: float = 0.95,
    power: float = 0.80,
) -> float:
    """Smallest accuracy difference a paired comparison of ``n`` examples can detect.

    Uses the normal approximation to McNemar's test. Only the examples the two
    models *disagree* on carry information about the difference, so the
    ``discordance`` rate — the share of examples where exactly one model is
    correct — drives the answer far more than the raw evaluation size does.

    This is a planning tool, deliberately approximate. Its purpose is to answer
    "is this evaluation set large enough to detect the effect I care about?"
    before a sweep is run, not to replace the paired bootstrap afterwards.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < discordance <= 1.0:
        raise ValueError("discordance must be in (0, 1]")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    z_alpha = _z_score(confidence)
    z_beta = _z_score(2.0 * power - 1.0) if power != 0.80 else 0.8416212335729143
    return (z_alpha + z_beta) * math.sqrt(discordance / n)

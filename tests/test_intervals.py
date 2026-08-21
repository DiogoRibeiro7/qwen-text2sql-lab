from __future__ import annotations

import pytest

from qwen_text2sql.evaluation.intervals import minimum_detectable_effect, wilson_interval


def test_matches_published_wilson_values() -> None:
    """Pin against a worked example rather than against our own implementation."""
    interval = wilson_interval(8, 10)
    assert interval.estimate == pytest.approx(0.8)
    assert interval.lower == pytest.approx(0.4901, abs=1e-4)
    assert interval.upper == pytest.approx(0.9433, abs=1e-4)


def test_accepts_outcomes_or_counts_identically() -> None:
    from_outcomes = wilson_interval([True] * 8 + [False] * 2)
    from_counts = wilson_interval(8, 10)
    assert from_outcomes == from_counts


def test_stays_inside_the_unit_interval_at_the_boundaries() -> None:
    """The normal approximation fails here; this is why Wilson is used."""
    perfect = wilson_interval(20, 20)
    assert perfect.estimate == 1.0
    assert perfect.upper == 1.0
    assert 0.0 < perfect.lower < 1.0

    hopeless = wilson_interval(0, 20)
    assert hopeless.estimate == 0.0
    assert hopeless.lower == 0.0
    assert 0.0 < hopeless.upper < 1.0


def test_more_data_narrows_the_interval() -> None:
    small = wilson_interval(5, 10)
    large = wilson_interval(500, 1000)
    assert small.estimate == large.estimate
    assert large.margin < small.margin


def test_higher_confidence_widens_the_interval() -> None:
    narrow = wilson_interval(50, 100, confidence=0.90)
    wide = wilson_interval(50, 100, confidence=0.99)
    assert wide.lower < narrow.lower
    assert wide.upper > narrow.upper


def test_format_is_stable_for_tables() -> None:
    assert wilson_interval(8, 10).format(digits=2) == "0.80 [0.49, 0.94]"


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        pytest.param((5, 0), {}, ValueError, id="zero-total"),
        pytest.param((11, 10), {}, ValueError, id="successes-exceed-total"),
        pytest.param((-1, 10), {}, ValueError, id="negative-successes"),
        pytest.param((5, 10), {"confidence": 0.0}, ValueError, id="confidence-zero"),
        pytest.param((5, 10), {"confidence": 1.0}, ValueError, id="confidence-one"),
        pytest.param((5, 10), {"confidence": 0.80}, ValueError, id="unsupported-confidence"),
        pytest.param((5,), {}, TypeError, id="count-without-total"),
        pytest.param(([True, False], 2), {}, TypeError, id="outcomes-with-total"),
    ],
)
def test_invalid_inputs_are_rejected(
    args: tuple[object, ...], kwargs: dict[str, object], expected: type[Exception]
) -> None:
    with pytest.raises(expected):
        wilson_interval(*args, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Minimum detectable effect
# --------------------------------------------------------------------------


def test_mde_shrinks_with_more_examples() -> None:
    assert minimum_detectable_effect(5000) < minimum_detectable_effect(500)


def test_mde_scales_with_the_square_root_of_n() -> None:
    """Quadrupling the evaluation set halves the detectable effect."""
    assert minimum_detectable_effect(400) == pytest.approx(
        2 * minimum_detectable_effect(1600), rel=1e-9
    )


def test_mde_grows_when_the_models_disagree_more() -> None:
    """Only discordant examples carry information about a paired difference."""
    assert minimum_detectable_effect(1000, discordance=0.40) > minimum_detectable_effect(
        1000, discordance=0.10
    )


def test_mde_is_a_plausible_magnitude() -> None:
    """A 1000-example paired comparison detects a few points, not a fraction of one."""
    effect = minimum_detectable_effect(1000, discordance=0.30)
    assert 0.02 < effect < 0.10


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"discordance": 0.0}, id="zero-discordance"),
        pytest.param({"discordance": 1.5}, id="discordance-above-one"),
        pytest.param({"power": 0.0}, id="zero-power"),
        pytest.param({"power": 1.0}, id="full-power"),
    ],
)
def test_mde_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        minimum_detectable_effect(1000, **kwargs)


def test_mde_rejects_a_non_positive_evaluation_size() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        minimum_detectable_effect(0)

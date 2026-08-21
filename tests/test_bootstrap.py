import pytest

from qwen_text2sql.evaluation.bootstrap import paired_bootstrap_binary


def test_bootstrap_detects_strictly_better_model() -> None:
    result = paired_bootstrap_binary([True] * 10, [False] * 10, n_bootstrap=500, seed=1)
    assert result.observed_difference == 1.0
    assert result.lower == 1.0
    assert result.upper == 1.0
    assert result.probability_positive == 1.0


def test_bootstrap_requires_pairs() -> None:
    with pytest.raises(ValueError, match="identical length"):
        paired_bootstrap_binary([True], [True, False])

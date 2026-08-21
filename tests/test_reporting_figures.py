from __future__ import annotations

import pandas as pd
import pytest

from qwen_text2sql.reporting.style import DARK, LIGHT, palette_for

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib is in the notebooks group")
matplotlib.use("Agg")

from qwen_text2sql.reporting import figures  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> object:
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture()
def curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "train_size": [250, 500, 1000, 2500],
            "execution_accuracy": [0.21, 0.28, 0.34, 0.39],
            "lower": [0.17, 0.24, 0.30, 0.35],
            "upper": [0.25, 0.32, 0.38, 0.43],
        }
    )


@pytest.fixture()
def per_database() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "db_id": [f"db{i}" for i in range(30)],
            "n": [20] * 30,
            "execution_accuracy": [i / 30 for i in range(30)],
            "lower": [max(0.0, i / 30 - 0.1) for i in range(30)],
            "upper": [min(1.0, i / 30 + 0.1) for i in range(30)],
        }
    )


# --------------------------------------------------------------------------
# Palette discipline
# --------------------------------------------------------------------------


def test_palette_refuses_to_cycle_past_its_validated_slots() -> None:
    """A ninth hue would be indistinguishable under CVD; fold or facet instead."""
    for palette in (LIGHT, DARK):
        assert palette.slot(0) == palette.series[0]
        with pytest.raises(IndexError, match="validated slots"):
            palette.slot(len(palette.series))


def test_light_and_dark_are_the_same_hues_restepped() -> None:
    assert len(LIGHT.series) == len(DARK.series)
    assert LIGHT.surface != DARK.surface
    assert LIGHT.text_primary != DARK.text_primary


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        palette_for("sepia")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Figures render, and refuse to render nonsense
# --------------------------------------------------------------------------


def test_learning_curve_draws_the_interval_band(curve: pd.DataFrame) -> None:
    ax = figures.learning_curve(curve, baseline=0.19)
    assert ax.get_ylabel() == "Execution accuracy"
    assert ax.collections, "no interval band was drawn"
    assert any("baseline" in text.get_text() for text in ax.texts)


def test_learning_curve_works_without_intervals(curve: pd.DataFrame) -> None:
    ax = figures.learning_curve(curve.drop(columns=["lower", "upper"]))
    assert ax.lines


def test_trend_labels_only_the_endpoint(curve: pd.DataFrame) -> None:
    """A value beside every marker is the anti-pattern; label the endpoint only."""
    ax = figures.learning_curve(curve)
    percent_labels = [text for text in ax.texts if text.get_text().endswith("%")]
    assert len(percent_labels) == 1


def test_rank_ablation_uses_a_log_axis(curve: pd.DataFrame) -> None:
    frame = curve.rename(columns={"train_size": "lora_rank"})
    ax = figures.rank_ablation(frame)
    assert ax.get_xscale() == "log"
    assert ax.get_xlabel() == "LoRA rank"


@pytest.mark.parametrize(
    "figure",
    [figures.learning_curve, figures.rank_ablation],
)
def test_trend_figures_reject_empty_input(figure: object) -> None:
    frame = pd.DataFrame(columns=["train_size", "lora_rank", "execution_accuracy"])
    with pytest.raises(ValueError, match="empty"):
        figure(frame)  # type: ignore[operator]


def test_error_breakdown_excludes_successes_and_labels_bars() -> None:
    breakdown = pd.DataFrame(
        {
            "error_kind": ["none", "syntax_error", "missing_table"],
            "count": [80, 15, 5],
            "share": [0.80, 0.15, 0.05],
            "is_failure": [False, True, True],
        }
    )
    ax = figures.error_breakdown(breakdown)
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert "none" not in labels
    assert "syntax error" in labels
    assert not ax.get_xaxis().get_visible()


def test_error_breakdown_refuses_a_flawless_run() -> None:
    breakdown = pd.DataFrame(
        {"error_kind": ["none"], "count": [10], "share": [1.0], "is_failure": [False]}
    )
    with pytest.raises(ValueError, match="every prediction executed"):
        figures.error_breakdown(breakdown)


def test_accuracy_by_database_announces_what_it_dropped(per_database: pd.DataFrame) -> None:
    """Silent truncation reads as 'this is everything'."""
    ax = figures.accuracy_by_database(per_database, top=10)
    assert len(ax.get_yticklabels()) == 10
    assert "20 lower-scoring databases not shown" in ax.get_title()


def test_accuracy_by_database_says_nothing_when_nothing_is_dropped(
    per_database: pd.DataFrame,
) -> None:
    ax = figures.accuracy_by_database(per_database, top=None)
    assert "not shown" not in ax.get_title()


def test_model_comparison_colours_by_sign(per_database: pd.DataFrame) -> None:
    """Difference is a polarity, so the diverging pair encodes direction."""
    comparison = pd.DataFrame(
        {
            "model": ["better", "worse"],
            "difference": [0.10, -0.05],
            "lower": [0.04, -0.11],
            "upper": [0.16, 0.01],
        }
    )
    ax = figures.model_comparison(comparison)
    palette = palette_for("light")
    colours = {line.get_color() for line in ax.lines}
    assert palette.positive in colours
    assert palette.negative in colours


def test_prompt_budget_marks_examples_over_the_limit() -> None:
    stats = pd.DataFrame({"prompt_chars": [100, 200, 300, 20_000]})
    ax = figures.prompt_budget(stats, limit_chars=10_000)
    annotations = " ".join(text.get_text() for text in ax.texts)
    assert "context limit" in annotations
    assert "1 example(s) over" in annotations


def test_house_style_context_restores_previous_settings() -> None:
    """A notebook that opts into the style for one figure must not leak it."""
    import matplotlib.pyplot as plt

    from qwen_text2sql.reporting.style import house_style

    before = plt.rcParams["axes.facecolor"]
    with house_style("dark") as palette:
        assert plt.rcParams["axes.facecolor"] == palette.surface
    assert plt.rcParams["axes.facecolor"] == before


def test_use_house_style_applies_globally_and_returns_the_palette() -> None:
    import matplotlib.pyplot as plt

    from qwen_text2sql.reporting.style import use_house_style

    try:
        palette = use_house_style("light")
        assert plt.rcParams["axes.facecolor"] == palette.surface
        # Gridlines are solid: dashing reads as "threshold", not "grid".
        assert plt.rcParams["grid.linestyle"] == "-"
        # Tick marks are suppressed because most spines are hidden.
        assert plt.rcParams["xtick.major.size"] == 0.0
    finally:
        plt.rcdefaults()

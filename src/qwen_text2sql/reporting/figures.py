"""Publication-quality figures for the experiments in this repository.

Each function takes a tidy frame from :mod:`qwen_text2sql.reporting.tables`,
draws onto an axis, and returns it. Nothing here reads a file, computes a
statistic, or calls ``plt.show``; that keeps the figures testable and lets a
notebook compose them.

Uncertainty is drawn wherever it exists. A learning curve of five point
estimates invites a reader to see a trend in what may be sampling noise, so the
interval is part of the mark rather than an optional extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from qwen_text2sql.reporting.style import Palette, palette_for

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "accuracy_by_database",
    "annotate_figure",
    "error_breakdown",
    "learning_curve",
    "model_comparison",
    "prompt_budget",
    "rank_ablation",
]


def _axes(ax: Axes | None, *, figsize: tuple[float, float] | None = None) -> Axes:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, created = plt.subplots(figsize=figsize) if figsize else plt.subplots()
    return created


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], what: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"{what} is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{what} is empty; nothing to plot")


def _has_interval(frame: pd.DataFrame) -> bool:
    return "lower" in frame and "upper" in frame


def _percent_axis(ax: Axes, which: str = "y", *, signed: bool = False) -> None:
    """Label an accuracy axis in percent, matching the direct labels."""
    sign = "+" if signed else ""
    axis = ax.get_yaxis() if which == "y" else ax.get_xaxis()
    axis.set_major_formatter(lambda value, _: f"{value * 100:{sign}.0f}%")


def _reference_line(ax: Axes, value: float, label: str, palette: Palette) -> None:
    """Draw a baseline. Dashed here means threshold, which is what it is."""
    ax.axhline(value, color=palette.neutral, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate(
        label,
        xy=(1.0, value),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color=palette.text_secondary,
    )


def _trend(
    frame: pd.DataFrame,
    x: str,
    *,
    ax: Axes | None,
    palette: Palette,
    title: str,
    xlabel: str,
    log_x: bool,
    baseline: float | None,
    baseline_label: str,
) -> Axes:
    _require_columns(frame, (x, "execution_accuracy"), title)
    axis = _axes(ax)
    ordered = frame.sort_values(x)
    colour = palette.slot(0)

    if _has_interval(ordered):
        axis.fill_between(
            ordered[x],
            ordered["lower"],
            ordered["upper"],
            color=colour,
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )
    axis.plot(
        ordered[x],
        ordered["execution_accuracy"],
        marker="o",
        color=colour,
        markeredgecolor=palette.surface,
        markeredgewidth=1.6,
        zorder=3,
    )

    if baseline is not None:
        _reference_line(axis, baseline, baseline_label, palette)

    # Direct-label the endpoint only. A value beside every marker goes unread.
    last = ordered.iloc[-1]
    axis.annotate(
        f"{last['execution_accuracy']:.1%}",
        xy=(last[x], last["execution_accuracy"]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=palette.text_primary,
        fontweight="medium",
    )

    if log_x:
        axis.set_xscale("log", base=2)
        axis.set_xticks(list(ordered[x]))
        axis.get_xaxis().set_major_formatter(lambda value, _: f"{int(value)}")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Execution accuracy")
    axis.set_title(title)
    _percent_axis(axis, "y")
    # Extra right margin so the endpoint label sits inside the axes.
    axis.margins(x=0.14)
    return axis


def learning_curve(
    summary: pd.DataFrame,
    *,
    ax: Axes | None = None,
    baseline: float | None = None,
    mode: str = "light",
) -> Axes:
    """Execution accuracy against the number of training examples.

    ``baseline`` draws the unadapted foundation model as a reference line, which
    is the only thing that makes the curve interpretable: the question is not
    whether accuracy rises with data but where it overtakes no fine-tuning.
    """
    return _trend(
        summary,
        "train_size",
        ax=ax,
        palette=palette_for(mode),  # type: ignore[arg-type]
        title="Learning curve",
        xlabel="Training examples",
        log_x=True,
        baseline=baseline,
        baseline_label="unadapted baseline",
    )


def rank_ablation(
    summary: pd.DataFrame,
    *,
    ax: Axes | None = None,
    baseline: float | None = None,
    mode: str = "light",
) -> Axes:
    """Execution accuracy against LoRA rank, with the dataset and recipe fixed."""
    return _trend(
        summary,
        "lora_rank",
        ax=ax,
        palette=palette_for(mode),  # type: ignore[arg-type]
        title="LoRA rank ablation",
        xlabel="LoRA rank",
        log_x=True,
        baseline=baseline,
        baseline_label="unadapted baseline",
    )


def error_breakdown(
    breakdown: pd.DataFrame, *, ax: Axes | None = None, mode: str = "light"
) -> Axes:
    """Failure categories as shares of the whole evaluation.

    One colour for every bar: the category is already on the axis, so shading by
    size would double-encode the bar length and spend the only free channel.
    """
    _require_columns(breakdown, ("error_kind", "count", "share"), "Error breakdown")
    palette = palette_for(mode)  # type: ignore[arg-type]
    failures = breakdown[breakdown["error_kind"] != "none"].sort_values("count")
    if failures.empty:
        raise ValueError("No failures to plot: every prediction executed successfully")

    height = max(2.2, 0.34 * len(failures) + 1.2)
    axis = _axes(ax, figsize=(7.2, height))
    positions = range(len(failures))
    axis.barh(
        list(positions),
        failures["count"],
        color=palette.slot(0),
        height=0.5,
        zorder=3,
    )
    axis.set_yticks(list(positions))
    axis.set_yticklabels([str(kind).replace("_", " ") for kind in failures["error_kind"]])

    # The value replaces the x-axis rather than duplicating it.
    for position, (count, share) in enumerate(
        zip(failures["count"], failures["share"], strict=True)
    ):
        axis.annotate(
            f"{int(count)}  ({share:.1%})",
            xy=(count, position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=palette.text_secondary,
        )

    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", visible=False)
    axis.get_xaxis().set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.set_title("Failures by category")
    axis.margins(x=0.18)
    return axis


def accuracy_by_database(
    frame: pd.DataFrame,
    *,
    ax: Axes | None = None,
    top: int | None = 25,
    mode: str = "light",
) -> Axes:
    """Per-database accuracy as a dot plot with confidence intervals.

    A bar chart here would imply each database is known equally well. The
    interval shows that a database with eight examples is barely measured, which
    is usually the more actionable fact.
    """
    _require_columns(
        frame, ("db_id", "execution_accuracy", "lower", "upper", "n"), "Per-database accuracy"
    )
    palette = palette_for(mode)  # type: ignore[arg-type]
    ordered = frame.sort_values("execution_accuracy")
    truncated = 0
    if top is not None and len(ordered) > top:
        # Say what was dropped: a silently truncated chart reads as complete.
        truncated = len(ordered) - top
        ordered = ordered.tail(top)

    height = max(2.6, 0.28 * len(ordered) + 1.6)
    axis = _axes(ax, figsize=(7.2, height))
    positions = list(range(len(ordered)))
    axis.hlines(
        positions,
        ordered["lower"],
        ordered["upper"],
        color=palette.slot(0),
        alpha=0.35,
        linewidth=2.0,
        zorder=2,
    )
    axis.plot(
        ordered["execution_accuracy"],
        positions,
        "o",
        color=palette.slot(0),
        markeredgecolor=palette.surface,
        markeredgewidth=1.4,
        zorder=3,
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [f"{db}  (n={n})" for db, n in zip(ordered["db_id"], ordered["n"], strict=True)]
    )
    axis.set_xlim(0, 1)
    axis.set_xlabel("Execution accuracy")
    _percent_axis(axis, "x")
    title = "Execution accuracy by database"
    if truncated:
        title = f"{title}  ·  {truncated} lower-scoring databases not shown"
    axis.set_title(title)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", visible=True)
    axis.spines["left"].set_visible(False)
    axis.margins(y=0.02)
    return axis


def prompt_budget(
    stats: pd.DataFrame,
    *,
    limit_chars: int | None = None,
    ax: Axes | None = None,
    mode: str = "light",
) -> Axes:
    """Distribution of prompt size, against the context window.

    Schema text dominates the prompt. Examples past the limit are silently
    truncated, so any accuracy drop on the databases with the largest schemas
    would be an artefact of the prompt rather than a property of the model.
    """
    _require_columns(stats, ("prompt_chars",), "Prompt statistics")
    palette = palette_for(mode)  # type: ignore[arg-type]
    axis = _axes(ax)
    axis.hist(
        stats["prompt_chars"],
        bins=40,
        color=palette.slot(0),
        alpha=0.85,
        zorder=3,
    )
    if limit_chars is not None:
        over = int((stats["prompt_chars"] > limit_chars).sum())
        axis.axvline(limit_chars, color=palette.negative, linewidth=1.4, linestyle="--", zorder=4)
        axis.annotate(
            f"context limit\n{over} example(s) over ({over / len(stats):.1%})",
            xy=(limit_chars, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(6, -6),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=palette.negative,
        )
    axis.set_xlabel("Prompt characters (question + evidence + schema)")
    axis.set_ylabel("Examples")
    axis.set_title("Prompt size against the context window")
    return axis


def model_comparison(
    comparison: pd.DataFrame, *, ax: Axes | None = None, mode: str = "light"
) -> Axes:
    """Paired accuracy differences against a common baseline, with intervals.

    Colour is diverging here because the quantity is a polarity: an improvement
    and a regression are opposite, and zero is the meaningful midpoint. An
    interval crossing the zero rule is the whole point of the figure.
    """
    _require_columns(comparison, ("model", "difference", "lower", "upper"), "Model comparison")
    palette = palette_for(mode)  # type: ignore[arg-type]
    ordered = comparison.sort_values("difference")
    height = max(2.4, 0.5 * len(ordered) + 1.6)
    axis = _axes(ax, figsize=(7.2, height))
    positions = list(range(len(ordered)))

    # The zero rule is the reference the whole figure is read against.
    axis.axvline(0.0, color=palette.text_secondary, linewidth=1.4, zorder=1)
    differences = [float(value) for value in ordered["difference"]]
    lowers = [float(value) for value in ordered["lower"]]
    uppers = [float(value) for value in ordered["upper"]]

    for position, difference, lower, upper in zip(
        positions, differences, lowers, uppers, strict=True
    ):
        colour = palette.positive if difference >= 0 else palette.negative
        axis.hlines(position, lower, upper, color=colour, alpha=0.35, linewidth=2.0, zorder=2)
        axis.plot(
            difference,
            position,
            "o",
            color=colour,
            markeredgecolor=palette.surface,
            markeredgewidth=1.4,
            zorder=3,
        )
        axis.annotate(
            f"{difference:+.1%}",
            xy=(upper, position),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=palette.text_secondary,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(list(ordered["model"]))
    axis.set_xlabel("Difference in execution accuracy vs baseline")
    _percent_axis(axis, "x", signed=True)
    axis.set_title("Paired model comparison")
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", visible=False)
    axis.spines["left"].set_visible(False)
    axis.margins(x=0.22, y=0.25)
    return axis


def annotate_figure(ax: Axes, note: str, *, mode: str = "light") -> Axes:
    """Attach a provenance or caveat note beneath a figure."""
    palette = palette_for(mode)  # type: ignore[arg-type]
    ax.figure.text(
        0.0,
        -0.02,
        note,
        ha="left",
        va="top",
        fontsize=8,
        color=palette.text_secondary,
        transform=ax.transAxes,
    )
    return ax

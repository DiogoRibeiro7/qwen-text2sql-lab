"""A single house style for every figure in the repository.

Figures that appear in a research write-up have to be legible in isolation, so
this module fixes the chrome once rather than leaving each notebook to restyle
matplotlib. Two rules drive the choices:

* **Colour encodes identity or polarity, never magnitude that a position already
  shows.** A single-series bar chart gets one colour for every bar; shading bars
  darker-where-larger spends the only free channel on information the bar length
  already carries.
* **Chrome recedes.** Hairline solid gridlines one shade off the surface, no top
  or right spine. Dashed rules are reserved for thresholds and reference values,
  where the dashing actually means something.

The palette is a validated colourblind-safe set: adjacent slots are separated by
ΔE ≥ 8 in OKLab under simulated CVD. Take slots in order and never cycle past the
end of the list.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["DARK", "LIGHT", "Palette", "house_style", "palette_for", "use_house_style"]

Mode = Literal["light", "dark"]


@dataclass(frozen=True, slots=True)
class Palette:
    """Colour roles for one rendering mode."""

    surface: str
    text_primary: str
    text_secondary: str
    grid: str
    series: tuple[str, ...]
    positive: str
    negative: str
    neutral: str

    def slot(self, index: int) -> str:
        """Return categorical slot ``index``, refusing to cycle past the palette."""
        if not 0 <= index < len(self.series):
            raise IndexError(
                f"Categorical slot {index} is out of range: this palette has "
                f"{len(self.series)} validated slots. Fold the tail into 'Other' "
                "or facet into small multiples rather than reusing a hue."
            )
        return self.series[index]


LIGHT = Palette(
    surface="#fcfcfb",
    text_primary="#0b0b0b",
    text_secondary="#52514e",
    grid="#e6e5e1",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    positive="#2a78d6",
    negative="#e34948",
    neutral="#8a8984",
)

DARK = Palette(
    surface="#1a1a19",
    text_primary="#ffffff",
    text_secondary="#c3c2b7",
    grid="#33332f",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    positive="#3987e5",
    negative="#e66767",
    neutral="#8a8984",
)


def palette_for(mode: Mode = "light") -> Palette:
    """Return the palette for a rendering mode."""
    if mode == "light":
        return LIGHT
    if mode == "dark":
        return DARK
    raise ValueError(f"Unknown mode {mode!r}; use 'light' or 'dark'")


def _rc_params(palette: Palette) -> dict[str, Any]:
    return {
        "figure.facecolor": palette.surface,
        "figure.figsize": (7.2, 4.2),
        "figure.dpi": 120,
        "savefig.facecolor": palette.surface,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
        "axes.facecolor": palette.surface,
        "axes.edgecolor": palette.grid,
        "axes.labelcolor": palette.text_secondary,
        "axes.titlecolor": palette.text_primary,
        "axes.titlesize": 12,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.prop_cycle": _prop_cycle(palette),
        # Solid hairline gridlines: dashing reads as "threshold" and is reserved
        # for reference lines that actually are one.
        "grid.color": palette.grid,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "xtick.color": palette.text_secondary,
        "ytick.color": palette.text_secondary,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Spines are mostly hidden; leaving tick marks behind reads as debris.
        "xtick.major.size": 0.0,
        "ytick.major.size": 0.0,
        "xtick.major.pad": 6.0,
        "ytick.major.pad": 6.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": palette.text_secondary,
        "text.color": palette.text_primary,
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    }


def _prop_cycle(palette: Palette) -> Any:
    from cycler import cycler

    return cycler(color=list(palette.series))


def use_house_style(mode: Mode = "light") -> Palette:
    """Apply the house style globally and return the active palette."""
    import matplotlib.pyplot as plt

    palette = palette_for(mode)
    # matplotlib types rcParams keys as a Literal union of every valid setting
    # name, which a dict[str, Any] cannot satisfy. The keys are validated by
    # matplotlib at runtime, and test_use_house_style_applies_globally asserts
    # they take effect.
    plt.rcParams.update(_rc_params(palette))  # type: ignore[arg-type]
    return palette


@contextmanager
def house_style(mode: Mode = "light") -> Iterator[Palette]:
    """Apply the house style for one block, then restore the previous settings."""
    import matplotlib.pyplot as plt

    palette = palette_for(mode)
    with plt.rc_context(_rc_params(palette)):  # type: ignore[arg-type]
        yield palette

"""One figure for an `Analysis`: what was measured, and what follows from it."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

from measly.analysis import Analysis, _score_name

__all__ = ["plot"]

PALETTE = ("#2563eb", "#ea580c", "#059669", "#7c3aed", "#db2777")
GUIDE = "#94a3b8"


def plot(analysis: Analysis, factor: float = 2.0, interval: float = 0.9, ax=None):
    """Measured points, the fitted band, and the projection out to `factor`.

    Points are the mean over draws, with bars at one standard deviation. The
    band is the percentile range across fitted curves, so it widens right of
    the dashed line, where nothing was measured.

    Returns the `Axes`, so several analyses MAY share a figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    measured = np.asarray(analysis.results.fraction.values, dtype=float)
    grid = np.geomspace(measured.min(), max(factor, measured.max()), 200)
    projected = analysis.project(grid, interval)

    for index, name in enumerate(analysis.models):
        scores = analysis.results.sel(model=name)
        band = projected[name]
        colour = PALETTE[index % len(PALETTE)]

        error = analysis.validation.get(name, float("nan"))
        label = name if np.isnan(error) else f"{name}   held-back {error:.1%}"

        ax.fill_between(grid * analysis.n_train, band.low, band.high,
                        color=colour, alpha=0.12, lw=0)
        for edge in (band.low, band.high):
            ax.plot(grid * analysis.n_train, edge, color=colour, alpha=0.35, lw=0.8)
        # The pointwise median of a finite ensemble kinks where members cross.
        # Smoothing is cosmetic. `project` and `summary` report the raw median.
        centre = gaussian_filter1d(band.loss, sigma=4, mode="nearest")
        ax.plot(grid * analysis.n_train, centre, color=colour, lw=2,
                label=label, zorder=4)
        ax.errorbar(measured * analysis.n_train, scores.mean("draw"),
                    yerr=scores.std("draw", ddof=1), fmt="o", ms=5,
                    color=colour, elinewidth=1.1, capsize=0, zorder=5)

    ax.axvline(analysis.n_train, color=GUIDE, ls=(0, (2, 3)), lw=1, zorder=1)
    ax.annotate("measured", xy=(analysis.n_train, 0.0),
                xycoords=("data", "axes fraction"), xytext=(-5, 7),
                textcoords="offset points", ha="right", va="bottom",
                color=GUIDE, fontsize=8)
    ax.annotate("projected", xy=(analysis.n_train, 0.0),
                xycoords=("data", "axes fraction"), xytext=(5, 7),
                textcoords="offset points", ha="left", va="bottom",
                color=GUIDE, fontsize=8)

    ax.set_xscale("log")
    ax.set_xlim(min(grid) * analysis.n_train / 1.05, max(grid) * analysis.n_train * 1.05)
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 3, 5)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GUIDE)
    ax.tick_params(colors="#475569", length=4)

    ax.set_xlabel("training examples", color="#334155")
    ax.set_ylabel(_score_name(analysis.score), color="#334155")
    ax.set_title(f"{analysis.law!r} fitted to {analysis.n_train} examples, "
                 f"projected to {round(analysis.n_train * factor):,}",
                 loc="left", color="#1e293b", fontsize=11, pad=14)
    ax.legend(frameon=False, fontsize=9, loc="upper right",
              handlelength=1.6, borderaxespad=0)
    return ax

"""Fit the scaling law to a sweep's measured losses.

One curve per draw. Since `sweep` draws independently at each fraction, a single
member pairs unrelated subsets and means nothing on its own — the ensemble is a
resampling of the curve, not a collection of trajectories. Read percentiles
across members, never an individual one.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

from measly.sweep import SweepResults

__all__ = ["scaling_law", "CurveEnsemble", "fit_scaling_law", "fit_scaling_laws"]

_ALPHA_BOUNDS = (0.01, 5.0)
"""alpha below 0 would mean loss rising with data; above 5 is never physical."""


def scaling_law(n, lower_bound, A, alpha):
    """L(n) = L_inf + A * n^(-alpha), the three-parameter inverse power law.

    Known as `pow3` in the learning-curve literature. `n` may be absolute
    counts or fractions of the pool: `lower_bound` and `alpha` come out
    identical either way, since rescaling n is absorbed entirely by `A`.
    """
    return lower_bound + A * np.asarray(n, dtype=float) ** (-alpha)


@dataclass
class CurveEnsemble:
    """Fitted curves from one model's draws, read by evaluating them all."""

    curves: list[Callable] = field(default_factory=list)
    n_failed: int = 0 # draw fits that didn't converge
  
    def __len__(self) -> int:
        return len(self.curves)

    def predict(self, x: ArrayLike) -> NDArray:
        """Evaluate every member at every n.

        Returns `(len(x), n_members)`. Quantiles across members are the
        interval; a single member means nothing on its own.
        """
        return np.stack([curve(x) for curve in self.curves], axis=-1)


def _initial_guess(y: NDArray) -> tuple[list[float], tuple]:
    """Data-driven start and bounds; curve_fit's default p0 of [1, 1, 1] is a
    poor start for a fit with a long ridge between L_inf, A and alpha."""
    floor_guess = max(float(y.min()) * 0.9, 0.0)
    p0 = [floor_guess, max(float(y.max() - y.min()), 1e-6), 1.0]
    lower = (0.0, 0.0, _ALPHA_BOUNDS[0])
    upper = (np.inf, np.inf, _ALPHA_BOUNDS[1])
    return p0, (lower, upper)


def fit_scaling_law(model_results: xr.DataArray, floor: float = 0.0) -> CurveEnsemble:
    """Fit one curve per draw for a single model.

    `model_results` has dims `(fraction, draw)`. `floor` raises the lower bound
    on `L_inf` — pass a measured measurement-noise floor if you have one, since
    constraining it is the largest single improvement to alpha's precision.
    """
    ensemble = CurveEnsemble()

    for _, draw_data in model_results.groupby("draw"):
        # groupby does not squeeze, so the group keeps a length-1 draw axis and
        # curve_fit would reject the 2-D ydata.
        y = np.asarray(draw_data.squeeze("draw").values, dtype=float)
        x = np.asarray(draw_data.fraction.values, dtype=float)

        p0, (lower, upper) = _initial_guess(y)
        lower = (max(lower[0], floor),) + lower[1:]
        p0[0] = max(p0[0], floor)

        try:
            popt, _ = curve_fit(
                scaling_law, x, y, p0=p0, bounds=(lower, upper), maxfev=10_000
            )
        except (RuntimeError, ValueError):
            ensemble.n_failed += 1
            continue

        lower_bound, A, alpha = popt
        ensemble.curves.append(
            partial(scaling_law, lower_bound=lower_bound, A=A, alpha=alpha)
        )

    return ensemble


def fit_scaling_laws(
    results: SweepResults, floor: float = 0.0
) -> dict[str, CurveEnsemble]:
    """Fit every model in a sweep, keyed by its label on the `model` axis."""
    return {
        str(name): fit_scaling_law(group.squeeze("model"), floor=floor)
        for name, group in results.groupby("model")
    }

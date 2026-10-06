"""Fit a scaling law to a sweep's measured losses.

One curve per draw. Draws are independent, so a single curve means nothing.
Read percentiles across the ensemble.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit


__all__ = [
    "pow3",
    "ScalingLaw",
    "POW3",
    "POW4",
    "CurveEnsemble",
    "fit_scaling_law",
    "fit_scaling_laws",
]

# Below 0, loss would rise with data. Above 5 is not physical.
_ALPHA_BOUNDS = (0.01, 5.0)


def pow3(n, lower_bound, A, alpha):
    """L(n) = L_inf + A * n^(-alpha). Called `pow3` in the literature.

    `n` MAY be counts or fractions of the pool. `lower_bound` and `alpha` are
    identical either way. Rescaling n is absorbed entirely by `A`.
    """
    return lower_bound + A * np.asarray(n, dtype=float) ** (-alpha)


@dataclass(frozen=True)
class ScalingLaw:
    """A functional form plus the starting point and bounds it needs.

    These MUST travel together. Forms differ in parameter count, and in which
    parameter is the asymptote that `floor` constrains. A generic guess would
    constrain the wrong one. `guess(y, floor)` returns `(p0, (lower, upper))`.
    """

    name: str
    func: Callable[..., NDArray]
    guess: Callable[[NDArray, float], tuple[list[float], tuple]]

    def __call__(self, x, *params) -> NDArray:
        return self.func(x, *params)

    def __repr__(self) -> str:
        return self.name


def _pow4(n, lower_bound, A, alpha, shift):
    """L(n) = L_inf + A * (n + shift)^(-alpha). Called `pow4`.

    `pow3` is this with `shift` at zero. So `pow4` never fits worse.

    `shift` separates where the curve bends from how low it ends up. Under
    `pow3` one parameter sets both. A curve still descending at the largest
    measured size can then only be fitted by driving `L_inf` to zero.
    """
    return lower_bound + A * (np.asarray(n, dtype=float) + shift) ** (-alpha)


def _pow3_guess(y: NDArray, floor: float) -> tuple[list[float], tuple]:
    start = max(float(y.min()) * 0.9, floor)
    p0 = [start, max(float(y.max() - y.min()), 1e-6), 1.0]
    return p0, ((floor, 0.0, _ALPHA_BOUNDS[0]), (np.inf, np.inf, _ALPHA_BOUNDS[1]))


def _pow4_guess(y: NDArray, floor: float) -> tuple[list[float], tuple]:
    p0, (lower, upper) = _pow3_guess(y, floor)
    return p0 + [0.01], (lower + (0.0,), upper + (10.0,))


POW3 = ScalingLaw("pow3", pow3, _pow3_guess)
POW4 = ScalingLaw("pow4", _pow4, _pow4_guess)


@dataclass
class CurveEnsemble:
    """Fitted curves from one model's draws. Read them by evaluating all."""

    curves: list[Callable] = field(default_factory=list)
    n_failed: int = 0  # draws whose fit did not converge

    def __len__(self) -> int:
        return len(self.curves)

    def predict(self, x: ArrayLike) -> NDArray:
        """Evaluate every member at every x.

        Returns `(len(x), n_members)`. Take quantiles across members for an
        interval. A single member MUST NOT be read on its own.
        """
        return np.stack([curve(x) for curve in self.curves], axis=-1)


def fit_scaling_law(
    model_results: xr.DataArray, law: ScalingLaw = POW4, floor: float = 0.0
) -> CurveEnsemble:
    """Fit one curve per draw for a single model.

    `model_results` has dims `(fraction, draw)`. `floor` raises the lower bound
    on the asymptote. Pass a measured noise floor if you have one.

    `law` picks the form. `POW4` is the default. On curves that have not yet
    bent it extrapolates far better than `POW3`. `POW3` in that regime loses to
    assuming no further improvement at all. `POW3` is what the literature
    usually quotes.
    """
    ensemble = CurveEnsemble()

    x = np.asarray(model_results.fraction.values, dtype=float)

    for y in model_results.transpose("draw", "fraction").values:
        p0, bounds = law.guess(y, floor)

        try:
            popt, _ = curve_fit(law.func, x, y, p0=p0, bounds=bounds, maxfev=10_000)
        except (RuntimeError, ValueError):
            ensemble.n_failed += 1
            continue

        # popt MUST be bound as a default argument. A closure over the loop
        # variable leaves every curve holding the last draw's parameters.
        ensemble.curves.append(lambda t, p=popt: law.func(t, *p))

    return ensemble


def fit_scaling_laws(
    results: xr.DataArray, law: ScalingLaw = POW4, floor: float = 0.0
) -> dict[str, CurveEnsemble]:
    """Fit every model in a sweep, keyed by its label on the `model` axis."""
    return {
        str(name): fit_scaling_law(results.sel(model=name), law=law, floor=floor)
        for name in results.model.values
    }

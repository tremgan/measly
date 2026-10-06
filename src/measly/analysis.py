from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from measly.fit import POW4, CurveEnsemble, ScalingLaw, fit_scaling_law, fit_scaling_laws
from measly.interfaces import Model, Score
from measly.sweep import sweep

__all__ = [
    "mean_squared_error",
    "DEFAULT_FRACTIONS",
    "VALIDATION_WARN",
    "Projection",
    "Analysis",
    "analyse",
]

DEFAULT_FRACTIONS: tuple[float, ...] = tuple(
    round(f, 4) for f in np.geomspace(0.1, 1.0, 8)
)


VALIDATION_WARN = 0.15
"""Above this held-back error, a projection MUST NOT be relied upon."""


def mean_squared_error(predicted, true) -> float:
    """The default `Score`. Regression only. Supply your own otherwise."""
    return float(np.mean((np.asarray(predicted) - np.asarray(true)) ** 2))


def _score_name(score: Score) -> str:
    """A label for the report. Falls back to the type for a callable object."""
    return getattr(score, "__name__", None) or type(score).__name__


def _unwrap(values: NDArray, scalar: bool) -> float | NDArray:
    """Give back a float for a scalar `factor`, an array for several."""
    return float(values[0]) if scalar else values


@dataclass(frozen=True)
class Projection:
    """What a model is expected to score on a larger training set.

    Every field is a float when `project` is given one factor, and an array
    aligned with the factors when it is given several.
    """

    loss: float | NDArray
    low: float | NDArray
    high: float | NDArray
    gain: float | NDArray
    gain_low: float | NDArray
    gain_high: float | NDArray


@dataclass(frozen=True)
class Analysis:
    """The measured curve, the fitted curves, and whether to believe them."""

    results: xr.DataArray
    curves: dict[str, CurveEnsemble]
    validation: dict[str, float]
    law: ScalingLaw
    score: Score
    n_train: int

    @property
    def models(self) -> list[str]:
        return list(self.curves)

    def measured(self) -> dict[str, float]:
        """Mean score at the full training pool, straight from the sweep."""
        largest = float(self.results.fraction.max())
        return {
            name: float(self.results.sel(model=name, fraction=largest).mean("draw"))
            for name in self.models
        }

    def project(self, factor: float = 2.0, interval: float = 0.9
                ) -> dict[str, Projection]:
        """Loss and gain at `factor` times the current training pool.

        `factor` MAY be a sequence. One call then gives the whole
        extrapolation curve, which is what a plot needs.

        The interval on `loss` is a percentile range across the fitted curves.
        It is dominated by test-set noise: every point in a draw shares one
        test set, so that draw's curve is shifted bodily. It is conservative,
        covering 98% or more at a nominal 90% against simulated ground truth.

        The interval on `gain` is far tighter, because each member is paired
        with itself and the shared offset cancels. `gain` is therefore not
        `measured()` minus `loss`; it is what the curve itself says it climbs.
        """
        tail = (1.0 - interval) / 2.0
        quantiles = [tail, 1.0 - tail]

        scalar = np.ndim(factor) == 0
        factors = np.atleast_1d(np.asarray(factor, dtype=float))

        projections = {}
        for name, ensemble in self.curves.items():
            # (len(factors), n_members). Quantiles MUST reduce the member axis
            # only; flattening both would mix factors into one distribution.
            future = ensemble.predict(factors)
            # Pair each member against its own value at the measured size.
            # Both carry that draw's test-set offset, so it cancels; comparing
            # against a mean over draws leaves it in and swamps the gain.
            gains = ensemble.predict(np.ones(1)) - future
            lo, hi = np.quantile(future, quantiles, axis=-1)
            gain_lo, gain_hi = np.quantile(gains, quantiles, axis=-1)
            projections[name] = Projection(
                loss=_unwrap(np.median(future, axis=-1), scalar),
                low=_unwrap(lo, scalar),
                high=_unwrap(hi, scalar),
                gain=_unwrap(np.median(gains, axis=-1), scalar),
                gain_low=_unwrap(gain_lo, scalar),
                gain_high=_unwrap(gain_hi, scalar),
            )
        return projections

    def summary(self, factor: float = 2.0) -> str:
        now = self.measured()
        projected = self.project(factor)
        target = round(self.n_train * factor)

        lines = [
            f"measly: {len(self.models)} model(s), {self.n_train} training examples, "
            f"{self.results.sizes['draw']} draws",
            f"        {self.law!r}, {_score_name(self.score)} (lower is better)",
            "",
        ]
        for name in self.models:
            p = projected[name]

            error = self.validation.get(name, float("nan"))
            if np.isnan(error):
                check = "not run"
            elif error > VALIDATION_WARN:
                check = (f"{error:.1%} error — this curve does not predict its own "
                         "measured sizes, so do not rely on the projection")
            else:
                check = f"{error:.1%} error"

            rows = [
                (f"now ({self.n_train} examples)", f"{now[name]:.4f}"),
                (f"at {target} examples",
                 f"{p.loss:.4f} [{p.low:.4f}, {p.high:.4f}]"),
                ("gain from getting there",
                 f"{p.gain:+.4f} [{p.gain_low:+.4f}, {p.gain_high:+.4f}]"),
                ("held-back check", check),
            ]
            width = max(len(label) for label, _ in rows)

            lines.append(f"  {name}")
            lines += [f"    {label:<{width}}  {value}" for label, value in rows]
            lines.append("")

        return "\n".join(lines).rstrip()

    def __repr__(self) -> str:
        worst = max((v for v in self.validation.values() if not np.isnan(v)),
                    default=float("nan"))
        checked = "unchecked" if np.isnan(worst) else f"worst held-back {worst:.1%}"
        return (f"<Analysis {len(self.models)} model(s), n_train={self.n_train}, "
                f"{self.law!r}, {checked}>")


def _validate(results: xr.DataArray, fractions: Sequence[float], hold_back: int,
              law: ScalingLaw, floor: float) -> dict[str, float]:
    """Refit on the smaller sizes, then predict the largest measured ones.

    Returns the worst relative error per model. Returns nan when too few points
    remain to fit the law once some are held back.
    """
    names = [str(n) for n in np.atleast_1d(results.model.values)]
    kept, held = list(fractions[:-hold_back]), list(fractions[-hold_back:])

    n_params = law.func.__code__.co_argcount - 1
    if hold_back <= 0 or len(kept) <= n_params:
        return {name: float("nan") for name in names}

    errors = {}
    for name in names:
        ensemble = fit_scaling_law(
            results.sel(model=name, fraction=kept), law=law, floor=floor
        )
        if len(ensemble) == 0:
            errors[name] = float("nan")
            continue

        worst = 0.0
        for fraction in held:
            actual = float(results.sel(model=name, fraction=fraction).mean("draw"))
            predicted = float(np.median(np.ravel(ensemble.predict(fraction))))
            if actual != 0.0:
                worst = max(worst, abs(predicted - actual) / abs(actual))
        errors[name] = worst

    return errors


def analyse(
    models: Mapping[str, Model] | Sequence[Model],
    X,
    y,
    score: Score = mean_squared_error,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    n_draws: int = 100,
    test_fraction: float = 0.25,
    law: ScalingLaw = POW4,
    floor: float = 0.0,
    rng: int | np.random.Generator | None = None,
    hold_back: int = 2,
) -> Analysis:
    """Measure a learning curve for each model and project it forward.

    `rng` accepts a seed, a `Generator`, or nothing. Pass a seed for a
    reproducible run. The split and the subsets both depend on it.

    `hold_back` sets how many of the largest measured sizes are withheld from a
    second fit, used to check the projection. Set it to 0 to skip the check.
    You then have no evidence the form suits your data.

    `models` MAY be a `{name: model}` mapping. Name them whenever `repr` is
    unhelpful, as it is for an sklearn `Pipeline`.
    """
    generator = np.random.default_rng(rng)
    fractions = list(fractions)

    results = sweep(
        models=models,
        X=X,
        y=y,
        fractions=fractions,
        rng=generator,
        score=score,
        n_draws=n_draws,
        test_fraction=test_fraction,
    )

    curves = fit_scaling_laws(results, law=law, floor=floor)
    validation = _validate(results, fractions, hold_back, law, floor)
    n_train = int(round(len(y) * (1.0 - test_fraction) * max(fractions)))

    return Analysis(results=results, curves=curves, validation=validation,
                    law=law, score=score, n_train=n_train)


if __name__ == "__main__":
    # uv run python src/measly/analysis.py
    # The `-m` form warns: __init__ imports this module, so runpy loads it twice.
    class _Ridge:
        """A model in the fit/predict convention. Penalty sets its capacity."""

        def __init__(self, penalty: float) -> None:
            self.penalty, self.weights = penalty, None

        def __repr__(self) -> str:
            return f"ridge({self.penalty:g})"

        def fit(self, X, y) -> None:
            gram = X.T @ X + self.penalty * np.eye(X.shape[1])
            self.weights = np.linalg.solve(gram, X.T @ y)

        def predict(self, X):
            return X @ self.weights

    _rng = np.random.default_rng(0)
    _X = _rng.normal(size=(2000, 12))
    _y = _X @ _rng.normal(size=12) + _rng.normal(0.0, 1.0, 2000)
    _models = [_Ridge(1.0), _Ridge(200.0)]

    # One call does the sweep, the fits and the hold-back check.
    _result = analyse(_models, _X, _y, rng=0, score=mean_squared_error)
    print(_result.summary())


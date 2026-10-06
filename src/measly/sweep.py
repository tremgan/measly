"""Measure performance across a grid of training-set sizes."""

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

from measly.interfaces import Score, Model

__all__ = ["train_test_split", "sweep"]


def train_test_split(X, y, test_fraction: float, rng: np.random.Generator) -> tuple:

    order = rng.permutation(len(X))
    n_test = int(round(len(y) * test_fraction))
    test, train = order[:n_test], order[n_test:]

    return X[train], X[test], y[train], y[test]


def sweep(
    models: Mapping[str, Model] | Sequence[Model],
    X,
    y,
    fractions: list[float],
    rng: np.random.Generator,
    score: Score,
    n_draws: int = 1000,
    test_fraction: float = 0.25,
) -> xr.DataArray:
    """Score every model at every fraction of the training pool, `n_draws` times.

    Only the training pool is downsampled. The test set is fixed within a draw
    and redrawn between draws. Models MUST be scored on held-out rows. Scoring
    on the training rows makes the curve rise with n rather than fall.

    Subsets are drawn without replacement. Fraction 1.0 still varies across
    draws, because each draw resplits the pool. A bootstrap is not needed for
    that spread and biases the level up by repeating rows.

    `models` MAY be a `{name: model}` mapping, which then keys the `model`
    axis. Otherwise `repr` does, and an sklearn `Pipeline`'s embeds an address.
    Those keys MUST be distinct, or one model would overwrite another.
    """

    named = (dict(models) if isinstance(models, Mapping)
             else {repr(model): model for model in models})
    if len(named) != len(models):
        raise ValueError(
            "models must have distinct repr(). The results array keys its model "
            "axis by repr, so duplicates would overwrite each other. Pass a "
            "{name: model} mapping to name them."
        )

    scores = np.full((len(fractions), len(named), n_draws), np.nan)

    for draw in range(n_draws):
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_fraction, rng)

        training_sets_indices = [
            rng.choice(len(train_X), size=round(len(train_X) * fraction), replace=False)
            for fraction in fractions
        ]

        for m, model in enumerate(named.values()):
            for f, indices in enumerate(training_sets_indices):
                model.fit(train_X[indices], train_y[indices])
                scores[f, m, draw] = score(model.predict(test_X), test_y)

    return xr.DataArray(
        scores,
        dims=("fraction", "model", "draw"),
        coords={"fraction": fractions, "model": list(named), "draw": np.arange(n_draws)},
    )

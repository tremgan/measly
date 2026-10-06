"""Measure performance across a grid of training-set sizes."""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import xarray as xr

from measly.interfaces import Score, Model

__all__ = ["downsample", "train_test_split", "sweep", "SweepResults"]


def downsample(
    X,
    y,
    fraction: float,
    rng: np.random.Generator,
    indices_only=False,
    with_replacement: bool = True,
):

    assert len(X) == len(y)
    assert 0 < fraction <= 1

    n_samples = int(round(len(X) * fraction))
    indices = rng.choice(len(X), size=n_samples, replace=with_replacement)

    if indices_only:
        return indices

    return X[indices], y[indices]


def train_test_split(X, y, test_fraction: float, rng: np.random.Generator) -> tuple:

    order = rng.permutation(len(X))
    n_test = int(round(len(y) * test_fraction))
    test, train = order[:n_test], order[n_test:]

    return X[train], X[test], y[train], y[test]


type SweepResults = xr.DataArray


def sweep(
    models: Mapping[str, Model] | Sequence[Model],
    X,
    y,
    fractions: list[float],
    rng: np.random.Generator,
    score: Score,
    n_draws: int = 1000,
    test_fraction: float = 0.25,
) -> SweepResults:
    """Score every model at every fraction of the training pool, `n_draws` times.

    Only the training pool is downsampled. The test set is fixed within a draw
    and redrawn between draws. Models MUST be scored on held-out rows. Scoring
    on the training rows makes the curve rise with n rather than fall.

    `models` MAY be a `{name: model}` mapping, which then keys the `model`
    axis. Otherwise `repr` does, and an sklearn `Pipeline`'s embeds an address.
    """

    named = (dict(models) if isinstance(models, Mapping)
             else {repr(model): model for model in models})

    results = xr.DataArray(
        data=np.full((len(fractions), len(named), n_draws), np.nan),
        dims=("fraction", "model", "draw"),
        coords={
            "fraction": fractions,
            "model": list(named),
            "draw": np.arange(n_draws),
        },
    )

    # to be parallelized later
    for draw in range(n_draws):
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_fraction, rng)

        training_sets_indices = {
            fraction: downsample(train_X, train_y, fraction=fraction, rng=rng, indices_only=True)
            for fraction in fractions
        }

        for name, model in named.items():
            for fraction, indices in training_sets_indices.items():
                model.fit(train_X[indices], train_y[indices])

                pred_y = model.predict(test_X)

                this_score = score(pred_y, test_y)

                results.loc[dict(fraction=fraction, model=name, draw=draw)] = (
                    this_score
                )

    return results

"""Measure performance across a grid of training-set sizes."""

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import xarray as xr

from measly.interfaces import Score, Model

__all__ = ["train_test_split", "sweep"]


def train_test_split(X, y, test_fraction: float, rng: np.random.Generator) -> tuple:

    order = rng.permutation(len(X))
    n_test = int(round(len(y) * test_fraction))
    test, train = order[:n_test], order[n_test:]

    return X[train], X[test], y[train], y[test]


def _score_draw(task) -> np.ndarray:
    """One draw's scores, `(fraction, model)`. Top level so a pool can pickle it."""
    models, score, train_X, train_y, test_X, test_y, subsets = task
    scores = np.empty((len(subsets), len(models)))
    for m, model in enumerate(models):
        for f, indices in enumerate(subsets):
            model.fit(train_X[indices], train_y[indices])
            scores[f, m] = score(model.predict(test_X), test_y)
    return scores


def sweep(
    models: Mapping[str, Model] | Sequence[Model],
    X,
    y,
    fractions: list[float],
    rng: np.random.Generator,
    score: Score,
    n_draws: int = 1000,
    test_fraction: float = 0.25,
    n_jobs: int = 1,
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

    `n_jobs` > 1 scores draws in that many processes, -1 in all cores. Results
    do not change. `score` and the models MUST be picklable, so no lambdas.
    Set `OMP_NUM_THREADS=1` or the workers' BLAS threads will contend.
    """

    named = (dict(models) if isinstance(models, Mapping)
             else {repr(model): model for model in models})
    if len(named) != len(models):
        raise ValueError(
            "models must have distinct repr(). The results array keys its model "
            "axis by repr, so duplicates would overwrite each other. Pass a "
            "{name: model} mapping to name them."
        )

    def tasks():
        for _ in range(n_draws):
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_fraction, rng)
            subsets = [
                rng.choice(len(train_X), size=round(len(train_X) * fraction), replace=False)
                for fraction in fractions
            ]
            yield list(named.values()), score, train_X, train_y, test_X, test_y, subsets

    if n_jobs == 1:
        per_draw = map(_score_draw, tasks())
    else:
        # ponytail: holds all draws' data at once. Ship indices if the pool is large.
        workers = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(workers) as pool:
            per_draw = list(pool.map(_score_draw, tasks(),
                                     chunksize=max(1, n_draws // (4 * workers))))
    scores = np.stack(list(per_draw), axis=-1)

    return xr.DataArray(
        scores,
        dims=("fraction", "model", "draw"),
        coords={"fraction": fractions, "model": list(named), "draw": np.arange(n_draws)},
    )

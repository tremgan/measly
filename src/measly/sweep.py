"""Measure performance across a grid of training-set sizes."""

import numpy as np
import xarray as xr

from measly.interfaces import Data, ProbabilisticModel, Score

__all__ = ["downsample", "train_test_split", "sweep", "SweepResults"]


def downsample(data: Data,
                    fraction: float,
                    rng: np.random.Generator,
                    with_replacement: bool = True):

    n_samples = int(round(len(data) * fraction))
    indices = rng.choice(len(data), size=n_samples, replace=with_replacement)

    return data[indices]


def train_test_split(data: Data,
                     test_fraction: float,
                     rng: np.random.Generator) -> tuple[Data, Data]:
    order = rng.permutation(len(data))
    n_test = int(round(len(data) * test_fraction))

    return data[order[n_test:]], data[order[:n_test]]


type SweepResults = xr.DataArray

def sweep(models: list[ProbabilisticModel],
        data: Data,
        fractions: list[float],
        rng: np.random.Generator,
        score: Score,
        n_draws: int=30,
        test_fraction: float=0.25
        ) -> SweepResults:
    """Score every model at every fraction of the training pool, `n_draws` times.

    The split is redrawn once per draw and held fixed across the fractions
    within it. Both halves of that matter. Resplitting per draw averages over
    test sets, so the test set's own sampling error does not survive as a fixed
    offset on `L_inf`. Holding it fixed within a draw keeps the points of one
    curve comparable: resplitting per fraction as well measurably worsens both
    `L_inf` and `alpha`, because differences along the curve then carry test
    noise that no longer cancels.

    Every model in a draw sees the same subsets and the same test set, which
    makes the comparison between models paired.
    """

    results = xr.DataArray(data=np.full((len(fractions), len(models), n_draws), np.nan),
                           dims=('fraction', 'model', 'draw'),
                           coords={'fraction': fractions,
                                   'model': [model.short_name for model in models],
                                   'draw': np.arange(n_draws)})

    # to be parallelized later
    for draw in range(n_draws):
        train, test = train_test_split(data, test_fraction, rng)
        subsets = {fraction: downsample(data=train, fraction=fraction, rng=rng)
                   for fraction in fractions}

        for model in models:
                for fraction, subset in subsets.items():

                        trained_model = model.condition(subset.X, subset.y)

                        this_score = score(trained_model.posterior_samples(test.X),
                                           test.y)

                        results.loc[dict(fraction=fraction,
                                         model=model.short_name,
                                         draw=draw)] = this_score


    return results

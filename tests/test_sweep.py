import numpy as np
import pytest

from conftest import FRACTIONS, Ridge, mse, run
from measly import analyse, sweep, train_test_split


def test_full_fraction_still_varies_across_draws(data):
    # The resplit alone must supply spread at 1.0. Without it the band collapses.
    results = run(data)
    assert (results.sel(fraction=1.0).std("draw") > 0).all()


def test_split_order_sizes_and_alignment(data):
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.25, np.random.default_rng(0))
    assert len(X_test) == len(y_test) == round(0.25 * len(X))
    assert len(X_train) + len(X_test) == len(X)
    rows = {tuple(r) for r in X_train}
    assert not any(tuple(r) in rows for r in X_test)


def test_sweep_shape_and_labels(data):
    results = run(data)
    assert results.dims == ("fraction", "model", "draw")
    assert results.shape == (len(FRACTIONS), 2, 6)
    assert list(results.model.values) == ["Ridge(1)", "Ridge(300)"]


def test_every_cell_is_written(data):
    """The array is NaN-initialised, so a skipped cell stays NaN."""
    assert not np.isnan(run(data).values).any()


def test_parallel_sweep_matches_serial(data):
    X, y = data
    kwargs = dict(models=[Ridge(1.0), Ridge(300.0)], X=X, y=y, fractions=FRACTIONS,
                  score=mse, n_draws=6)
    serial = sweep(rng=np.random.default_rng(0), **kwargs)
    parallel = sweep(rng=np.random.default_rng(0), n_jobs=2, **kwargs)
    assert np.array_equal(serial, parallel)


def test_sweep_is_reproducible(data):
    assert np.allclose(run(data, seed=3), run(data, seed=3))


def test_curve_falls_with_sample_size(data):
    """Held-out error improves with more data; on training rows it would not."""
    mean = run(data, n_draws=10).mean("draw")
    assert (mean.sel(fraction=1.0) < mean.sel(fraction=0.1)).all()


def test_spread_shrinks_with_sample_size(data):
    spread = run(data, n_draws=10).std("draw", ddof=1)
    assert (spread.sel(fraction=1.0) < spread.sel(fraction=0.1)).all()


def test_duplicate_model_labels_are_rejected(data):
    X, y = data
    with pytest.raises(ValueError, match="distinct repr"):
        analyse(models=[Ridge(1.0), Ridge(1.0)], X=X, y=y, rng=0, n_draws=3)

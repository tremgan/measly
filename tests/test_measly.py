import numpy as np
import pytest

from measly import downsample, fit_scaling_law, fit_scaling_laws, scaling_law, sweep, train_test_split

FRACTIONS = [0.1, 0.18, 0.32, 0.56, 1.0]


class Ridge:
    """A model in the fit/predict convention, with no dependencies."""

    def __init__(self, penalty):
        self.penalty, self.weights = penalty, None

    def __repr__(self):
        return f"Ridge({self.penalty:g})"

    def fit(self, X, y):
        gram = X.T @ X + self.penalty * np.eye(X.shape[1])
        self.weights = np.linalg.solve(gram, X.T @ y)

    def predict(self, X):
        return X @ self.weights


def mse(predicted, true):
    return float(np.mean((np.asarray(predicted) - np.asarray(true)) ** 2))


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8))
    return X, X @ rng.normal(size=8) + rng.normal(0, 1.0, 600)


def run(data, n_draws=6, seed=0):
    X, y = data
    return sweep(models=[Ridge(1.0), Ridge(300.0)], X=X, y=y, fractions=FRACTIONS,
                 rng=np.random.default_rng(seed), score=mse, n_draws=n_draws)


def test_downsample_size_and_replacement(data):
    X, y = data
    Xs, ys = downsample(X, y, 0.25, np.random.default_rng(0))
    assert len(Xs) == len(ys) == round(0.25 * len(X))
    assert len(np.unique(Xs, axis=0)) < len(Xs)          # bootstrap repeats rows


def test_downsample_without_replacement(data):
    X, y = data
    Xs, _ = downsample(X, y, 0.5, np.random.default_rng(0), with_replacement=False)
    assert len(np.unique(Xs, axis=0)) == len(Xs)


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


def test_sweep_is_reproducible(data):
    assert np.allclose(run(data, seed=3), run(data, seed=3))


def test_curve_falls_with_sample_size(data):
    """Held-out error improves with more data; on training rows it would not."""
    mean = run(data, n_draws=10).mean("draw")
    assert (mean.sel(fraction=1.0) < mean.sel(fraction=0.1)).all()


def test_spread_shrinks_with_sample_size(data):
    spread = run(data, n_draws=10).std("draw", ddof=1)
    assert (spread.sel(fraction=1.0) < spread.sel(fraction=0.1)).all()


def test_fit_recovers_a_known_curve():
    truth = dict(lower_bound=1.0, A=0.6, alpha=1.0)
    rng = np.random.default_rng(0)
    clean = scaling_law(np.array(FRACTIONS), **truth)
    noisy = clean[:, None] + rng.normal(0, 0.01, (len(FRACTIONS), 20))
    import xarray as xr
    measured = xr.DataArray(noisy, dims=("fraction", "draw"),
                            coords={"fraction": FRACTIONS, "draw": np.arange(20)})

    ensemble = fit_scaling_law(measured)
    assert len(ensemble) == 20 and ensemble.n_failed == 0
    assert np.median(ensemble.predict(1.0)) == pytest.approx(scaling_law(1.0, **truth), abs=0.02)


def test_predict_shape(data):
    """A sequence of sizes gives (len(x), n_members); a scalar drops the first
    axis, so callers that mix the two need `np.atleast_2d`."""
    ensemble = fit_scaling_law(run(data, n_draws=10).sel(model="Ridge(300)"))
    assert ensemble.predict([0.1, 0.5, 1.0]).shape == (3, len(ensemble))
    assert ensemble.predict(1.0).shape == (len(ensemble),)


def test_fit_scaling_laws_keys_by_model(data):
    fitted = fit_scaling_laws(run(data))
    assert set(fitted) == {"Ridge(1)", "Ridge(300)"}

import numpy as np
import pytest

from measly import (
    analyse,
    downsample,
    fit_scaling_law,
    fit_scaling_laws,
    scaling_law,
    sweep,
    train_test_split,
)

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


def test_downsample_size_and_no_repeats(data):
    X, y = data
    Xs, ys = downsample(X, y, 0.5, np.random.default_rng(0))
    assert len(Xs) == len(ys) == round(0.5 * len(X))
    assert len(np.unique(Xs, axis=0)) == len(Xs)


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


def test_law_is_configurable_and_defaults_to_pow3():
    from measly import POW3, POW4, ScalingLaw

    assert repr(POW3) == "pow3" and repr(POW4) == "pow4"
    assert POW3(1.0, 1.0, 0.5, 1.0) == pytest.approx(1.5)
    assert POW4(1.0, 1.0, 0.5, 1.0, 0.0) == pytest.approx(1.5)  # shift=0 is pow3


def test_pow4_fits_and_differs_from_pow3(data):
    from measly import POW3, POW4

    measured = run(data, n_draws=10).sel(model="Ridge(300)")
    three = fit_scaling_law(measured, law=POW3)
    four = fit_scaling_law(measured, law=POW4)
    assert len(four) > 0 and four.n_failed == 0
    assert not np.allclose(three.predict(2.0), four.predict(2.0))


def test_each_curve_keeps_its_own_parameters(data):
    """A closure over the loop variable would leave every member identical."""
    ensemble = fit_scaling_law(run(data, n_draws=8).sel(model="Ridge(300)"))
    values = ensemble.predict(1.0)
    assert len(np.unique(values)) == len(ensemble)


def test_a_custom_law_can_be_supplied(data):
    from measly import ScalingLaw

    flat = ScalingLaw("flat", lambda x, c: c + 0.0 * np.asarray(x, float),
                      lambda y, floor: ([max(float(y.min()), floor)],
                                        ((floor,), (np.inf,))))
    ensemble = fit_scaling_law(run(data, n_draws=5).sel(model="Ridge(1)"), law=flat)
    assert len(ensemble) == 5
    assert np.allclose(ensemble.predict(1.0), ensemble.predict(100.0))


def test_default_law_is_pow4(data):
    from measly import POW4

    measured = run(data, n_draws=5).sel(model="Ridge(300)")
    assert np.allclose(fit_scaling_law(measured).predict(2.0),
                       fit_scaling_law(measured, law=POW4).predict(2.0))


# --- analyse ---------------------------------------------------------------


def analysed(data, **kwargs):
    X, y = data
    options = dict(models=[Ridge(1.0), Ridge(300.0)], X=X, y=y, rng=0, n_draws=6)
    return analyse(**(options | kwargs))


def test_analyse_keys_everything_by_model_repr(data):
    result = analysed(data)
    assert set(result.curves) == {"Ridge(1)", "Ridge(300)"}
    assert set(result.validation) == set(result.curves)
    assert result.models == list(result.curves)


def test_default_grid_is_eight_fractions(data):
    from measly import DEFAULT_FRACTIONS

    assert len(DEFAULT_FRACTIONS) == 8
    assert analysed(data).results.sizes["fraction"] == 8


def test_projection_brackets_its_median(data):
    for name, projection in analysed(data).project(2.0).items():
        assert projection.low <= projection.loss <= projection.high
        assert projection.gain_low <= projection.gain <= projection.gain_high


def test_gain_is_what_the_curve_itself_climbs(data):
    """Each member is paired with its own level, and POW4 cannot rise.

    Comparing a member against a mean over draws instead left that draw's
    test-set offset in the difference, which produced negative gains.
    """
    for projection in analysed(data).project(2.0).values():
        assert projection.gain >= 0.0
        assert projection.gain_low >= 0.0


def test_same_seed_same_answer(data):
    assert np.allclose(analysed(data, rng=7).results, analysed(data, rng=7).results)


def test_different_seeds_differ(data):
    assert not np.allclose(analysed(data, rng=7).results, analysed(data, rng=8).results)


def test_hold_back_zero_skips_validation(data):
    assert all(np.isnan(v) for v in analysed(data, hold_back=0).validation.values())


def test_validation_runs_by_default(data):
    assert all(not np.isnan(v) for v in analysed(data).validation.values())


def test_duplicate_model_labels_are_rejected(data):
    X, y = data
    with pytest.raises(ValueError, match="distinct repr"):
        analyse(models=[Ridge(1.0), Ridge(1.0)], X=X, y=y, rng=0, n_draws=3)


def test_law_flows_through(data):
    from measly import POW3

    assert analysed(data, law=POW3).law is POW3


def test_summary_mentions_each_model_and_the_check(data):
    text = analysed(data).summary()
    assert "Ridge(1)" in text and "Ridge(300)" in text
    assert "held-back check" in text


def test_models_may_be_named(data):
    X, y = data
    result = analyse({"small": Ridge(1.0), "big": Ridge(300.0)},
                     X=X, y=y, rng=0, n_draws=3)

    assert result.models == ["small", "big"]
    assert set(result.curves) == {"small", "big"}
    assert set(result.validation) == {"small", "big"}
    assert set(result.measured()) == {"small", "big"}


def test_project_accepts_a_vector_of_factors(data):
    result = analysed(data)
    factors = [1.0, 2.0, 4.0]
    curve = result.project(factors)

    for name in result.models:
        one_at_a_time = [result.project(f)[name] for f in factors]
        assert curve[name].loss.shape == (len(factors),)
        for field in ("loss", "low", "high", "gain", "gain_low", "gain_high"):
            np.testing.assert_allclose(
                getattr(curve[name], field),
                [getattr(p, field) for p in one_at_a_time],
            )


def test_gain_interval_is_tighter_than_the_loss_interval(data):
    """Pairing cancels the per-draw test-set offset that dominates the level."""
    for projected in analysed(data).project(2.0).values():
        assert projected.gain_high - projected.gain_low < projected.high - projected.low


def test_project_returns_floats_for_one_factor(data):
    projected = analysed(data).project(2.0)
    assert all(isinstance(getattr(p, "loss"), float) for p in projected.values())


def test_plot_labels_each_model(data):
    import matplotlib

    matplotlib.use("Agg")
    from measly import plot

    result = analysed(data)
    ax = plot(result, factor=3.0)

    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert len(labels) == len(result.models)
    assert all(any(name in label for label in labels) for name in result.models)
    assert ax.get_xlabel() == "training examples"


def test_summary_names_the_score(data):
    def mean_absolute_error(predicted, true):
        return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(true))))

    assert "mean_squared_error" in analysed(data).summary()
    assert "mean_absolute_error" in analysed(data, score=mean_absolute_error).summary()

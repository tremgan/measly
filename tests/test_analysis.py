import numpy as np

from conftest import Ridge, analysed
from measly import DEFAULT_FRACTIONS, POW3, analyse


def test_analyse_keys_everything_by_model_repr(data):
    result = analysed(data)
    assert set(result.curves) == {"Ridge(1)", "Ridge(300)"}
    assert set(result.validation) == set(result.curves)
    assert result.models == list(result.curves)


def test_default_grid_is_eight_fractions(data):
    assert len(DEFAULT_FRACTIONS) == 8
    assert analysed(data).results.sizes["fraction"] == 8


def test_projection_brackets_its_median(data):
    for projection in analysed(data).project(2.0).values():
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


def test_law_flows_through(data):
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
    assert all(isinstance(p.loss, float) for p in projected.values())


def test_summary_names_the_score(data):
    def mean_absolute_error(predicted, true):
        return float(np.mean(np.abs(np.asarray(predicted) - np.asarray(true))))

    assert "mean_squared_error" in analysed(data).summary()
    assert "mean_absolute_error" in analysed(data, score=mean_absolute_error).summary()

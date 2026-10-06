import numpy as np
import pytest
import xarray as xr

from conftest import FRACTIONS, run
from measly import POW3, POW4, ScalingLaw, fit_scaling_law, fit_scaling_laws, pow3


def test_fit_recovers_a_known_curve():
    truth = dict(lower_bound=1.0, A=0.6, alpha=1.0)
    rng = np.random.default_rng(0)
    clean = pow3(np.array(FRACTIONS), **truth)
    noisy = clean[:, None] + rng.normal(0, 0.01, (len(FRACTIONS), 20))
    measured = xr.DataArray(noisy, dims=("fraction", "draw"),
                            coords={"fraction": FRACTIONS, "draw": np.arange(20)})

    ensemble = fit_scaling_law(measured)
    assert len(ensemble) == 20 and ensemble.n_failed == 0
    assert np.median(ensemble.predict(1.0)) == pytest.approx(pow3(1.0, **truth), abs=0.02)


def test_predict_shape(data):
    """A sequence of sizes gives (len(x), n_members); a scalar drops the first
    axis, so callers that mix the two need `np.atleast_2d`."""
    ensemble = fit_scaling_law(run(data, n_draws=10).sel(model="Ridge(300)"))
    assert ensemble.predict([0.1, 0.5, 1.0]).shape == (3, len(ensemble))
    assert ensemble.predict(1.0).shape == (len(ensemble),)


def test_fit_scaling_laws_keys_by_model(data):
    fitted = fit_scaling_laws(run(data))
    assert set(fitted) == {"Ridge(1)", "Ridge(300)"}


def test_law_names_and_zero_shift_is_pow3():
    assert repr(POW3) == "pow3" and repr(POW4) == "pow4"
    assert POW3(1.0, 1.0, 0.5, 1.0) == pytest.approx(1.5)
    assert POW4(1.0, 1.0, 0.5, 1.0, 0.0) == pytest.approx(1.5)  # shift=0 is pow3


def test_pow4_fits_and_differs_from_pow3(data):
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
    flat = ScalingLaw("flat", lambda x, c: c + 0.0 * np.asarray(x, float),
                      lambda y, floor: ([max(float(y.min()), floor)],
                                        ((floor,), (np.inf,))))
    ensemble = fit_scaling_law(run(data, n_draws=5).sel(model="Ridge(1)"), law=flat)
    assert len(ensemble) == 5
    assert np.allclose(ensemble.predict(1.0), ensemble.predict(100.0))


def test_default_law_is_pow4(data):
    measured = run(data, n_draws=5).sel(model="Ridge(300)")
    assert np.allclose(fit_scaling_law(measured).predict(2.0),
                       fit_scaling_law(measured, law=POW4).predict(2.0))

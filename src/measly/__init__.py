"""Estimate whether a model is data-limited, and whether more capacity would pay off."""

from measly.curve_fit import CurveEnsemble, fit_scaling_law, fit_scaling_laws, scaling_law
from measly.interfaces import Data, EnsembleModel, ProbabilisticModel, Score
from measly.sweep import SweepResults, downsample, sweep, train_test_split

__all__ = [
    "Data",
    "ProbabilisticModel",
    "EnsembleModel",
    "Score",
    "downsample",
    "train_test_split",
    "sweep",
    "SweepResults",
    "scaling_law",
    "CurveEnsemble",
    "fit_scaling_law",
    "fit_scaling_laws",
]

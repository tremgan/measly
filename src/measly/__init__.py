"""Estimate whether a model is data-limited, and whether more capacity would pay off."""

from measly.analysis import (
    DEFAULT_FRACTIONS,
    VALIDATION_WARN,
    Analysis,
    Projection,
    analyse,
    mean_squared_error,
)
from measly.fit import (
    POW3,
    POW4,
    CurveEnsemble,
    ScalingLaw,
    fit_scaling_law,
    fit_scaling_laws,
    pow3,
)
from measly.interfaces import Model, Score
from measly.plot import plot
from measly.sweep import sweep, train_test_split

__all__ = [
    "analyse",
    "Analysis",
    "Projection",
    "mean_squared_error",
    "plot",
    "DEFAULT_FRACTIONS",
    "VALIDATION_WARN",
    "Model",
    "Score",
    "train_test_split",
    "sweep",
    "pow3",
    "ScalingLaw",
    "POW3",
    "POW4",
    "CurveEnsemble",
    "fit_scaling_law",
    "fit_scaling_laws",
]

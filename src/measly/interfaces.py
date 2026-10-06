"""The contracts a user implements to run measly on their own problem."""

from typing import Any, Callable, Protocol, Sequence

__all__ = ["Score", "Model"]


type Score = Callable[[Sequence[float], Sequence[float]], float]
"""Reduce predictions and true targets to one number. Lower MUST be better.

It SHOULD be a mean over per-point terms: MSE, MAE, error rate. Set-level
statistics MUST NOT be used. F1, AUC and Spearman do not decompose per point,
so the power-law form does not describe them. The fit still returns a number.
"""


class Model(Protocol):
    """Anything with scikit-learn's fit/predict convention.

    `fit` MUST leave the model fully retrained. `sweep` refits one instance at
    every sample size. A model that carries state between calls, such as an
    sklearn estimator with `warm_start=True`, reports the same score at every
    size. That plots as a flat curve and raises nothing.

    `predict` MUST return one number per row of `X`, in order. Returns are
    `Any`: sklearn's `fit` gives back `self`, and `predict` an ndarray.
    """

    def fit(self, X, y) -> Any: ...

    def predict(self, X) -> Any: ...

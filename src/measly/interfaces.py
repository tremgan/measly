"""The contracts a user implements to run measly on their own problem."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol, Self, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["Score", "ProbabilisticModel", "EnsembleModel", "Data"]


type Score = Callable[[NDArray, Sequence[float]], float]
"""Reduce posterior samples and the true targets to one number.

Samples arrive as `(n_points, n_samples)`. Lower is better, and it should be a
mean over per-point terms: that is what keeps a power-law fit to the resulting
curve meaningful, and it rules out set-level statistics like F1 or Spearman.
CRPS is the natural choice, being a proper scoring rule for a distribution that
reduces exactly to absolute error for a single sample — which is what lets
deterministic and probabilistic models share one curve.
"""


class ProbabilisticModel(ABC):
    """A model measly can retrain and then ask for predictive samples."""

    @abstractmethod
    def posterior_samples(self, X) -> NDArray:
        """Draw from the posterior predictive at each row of X.

        Returns `(len(X), n_samples)`. The model chooses `n_samples`; a
        deterministic model returns a single column. Monte Carlo error on the
        resulting score falls as 1/sqrt(n_samples) — around 0.5% at 200 samples,
        well under the draw-to-draw spread the sweep is measuring.
        """

    @abstractmethod
    def condition(self, X, y) -> Self:
        """Fit on (X, y) and return a NEW model.

        Returning a new instance rather than mutating is what stops a fit
        leaking between cells of the sweep.
        """

    @property
    @abstractmethod
    def short_name(self) -> str:
        """Label for this model along the results array's `model` axis."""


@dataclass
class EnsembleModel(ProbabilisticModel):
    """Several models pooled into one predictive distribution.

    Pooling the members' samples *is* the equal-weight mixture of their
    posteriors, so there is nothing to combine beyond a concatenation.

    `condition` fits every member on the same data, so variation between members
    has to come from the members differing — by hyperparameter or by model
    class. An ensemble of identical deterministic members returns identical
    columns and quantifies nothing, and `condition` has no rng with which to
    bootstrap them apart.
    """

    models: list[ProbabilisticModel]

    @property
    def short_name(self) -> str:
        return "ensemble(" + "+".join(m.short_name for m in self.models) + ")"

    def condition(self, X, y) -> "EnsembleModel":
        return EnsembleModel([m.condition(X, y) for m in self.models])

    def posterior_samples(self, X) -> NDArray:
        return np.concatenate([m.posterior_samples(X) for m in self.models], axis=1)


class Data(Protocol):
    """Data or a pointer to data"""

    def __getitem__(self, index) -> Self:
        ...

    def __len__(self) -> int:
        ...

    @property
    def X(self):
        ...

    @property
    def y(self) -> Sequence[float]:
        ...

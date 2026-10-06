import numpy as np
import pytest

from measly import analyse, sweep

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


def analysed(data, **kwargs):
    X, y = data
    options = dict(models=[Ridge(1.0), Ridge(300.0)], X=X, y=y, rng=0, n_draws=6)
    return analyse(**(options | kwargs))

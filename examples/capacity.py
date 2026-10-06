"""More capacity is worse on a pilot and better once the data arrives.

The small model is misspecified: it sees a subset of the features, so it has a
floor above the noise. The large model is unbiased but pays variance early.
The curves therefore cross, and which model is "better" depends entirely on
how much data you have.

Because the data is synthetic the projection can be audited rather than
trusted. The extra rows are drawn from the same process and the models
refitted on them, so the figure carries both what measly predicted and what
actually happened.

    uv run --group demo python examples/capacity.py

Writes docs/learning-curve.png.
"""

import numpy as np
from matplotlib.lines import Line2D
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from measly import analyse, mean_squared_error, plot
from measly.plot import GUIDE, PALETTE

TOTAL, CAPACITY = 75, 20
PILOT, NOISE = 3000, 1.0
TAIL = 0.05
"""Weight on every feature the small model omits.

It alone sets where the curves cross, at `NOISE**2 / TAIL**2` examples. The
feature counts set how steep the large model's descent is and how wide the
final gap is, without moving the crossing.
"""

FACTORS = (1.25, 1.5, 1.75, 2.0)
"""Multiples of the measured pool to audit the projection at.

Capped at 2x on purpose. The held-back check fits on the smaller sizes and
predicts the largest two, a stretch of `1.0 / 0.5179`, so roughly 1.9x is as
far as the method carries evidence. Auditing past that measures nothing the
tool claims.
"""

REPEATS = 10
"""Independent collections at each size, so the audit carries its own spread."""


def ridge(capacity):
    """Ridge that sees only the first `capacity` features."""
    return Pipeline([
        ("capacity",
         ColumnTransformer([("keep", "passthrough", slice(0, capacity))])),
        ("ridge", Ridge()),
    ])


rng = np.random.default_rng(0)
weights = np.concatenate([rng.normal(size=4), np.full(TOTAL - 4, TAIL)])


def sample(n):
    X = rng.normal(size=(n, TOTAL))
    return X, X @ weights + rng.normal(0.0, NOISE, n)


# Named, because a Pipeline's repr embeds a memory address.
models = {f"ridge({capacity} features)": ridge(capacity)
          for capacity in (CAPACITY, TOTAL)}

X_pilot, y_pilot = sample(PILOT)
result = analyse(models, X_pilot, y_pilot, rng=0)
print(result.summary())
print(f"\ncurves cross near {round(NOISE ** 2 / TAIL ** 2)} examples\n")

# Go and collect the data measly was asked about, then check it was right.
# A lab keeps its pilot and adds to it, so the larger sets contain the smaller
# ones. Only the new rows are redrawn across repeats, because the pilot is not
# something that gets collected again.
X_check, y_check = sample(20_000)
sizes = [round(result.n_train * factor) for factor in FACTORS]
X_base, y_base = X_pilot[:result.n_train], y_pilot[:result.n_train]

scores = {name: np.empty((REPEATS, len(sizes))) for name in models}
for repeat in range(REPEATS):
    X_new, y_new = sample(max(sizes) - result.n_train)
    X_more = np.vstack([X_base, X_new])
    y_more = np.concatenate([y_base, y_new])
    for name, model in models.items():
        for column, n in enumerate(sizes):
            model.fit(X_more[:n], y_more[:n])
            scores[name][repeat, column] = mean_squared_error(
                model.predict(X_check), y_check)

actual = {name: value.mean(axis=0) for name, value in scores.items()}
spread = {name: value.std(axis=0, ddof=1) for name, value in scores.items()}

print(f"{'model':>20} {'examples':>9} {'actual':>17} {'projected':>10}"
      f" {'interval':>18} {'error':>7}")
for name in models:
    for factor, n, measured, sd in zip(FACTORS, sizes, actual[name], spread[name]):
        projected = result.project(factor)[name]
        print(f"{name:>20} {n:>9} {f'{measured:.4f} +-{sd:.4f}':>17}"
              f" {projected.loss:>10.4f}"
              f" {f'[{projected.low:.4f}, {projected.high:.4f}]':>18}"
              f" {abs(projected.loss - measured) / measured:>7.1%}"
              f"{'' if projected.low <= measured <= projected.high else '  OUT'}")

ax = plot(result, factor=max(FACTORS))
for index, name in enumerate(result.models):
    colour = PALETTE[index % len(PALETTE)]
    ax.errorbar(sizes, actual[name], yerr=spread[name], fmt="D", ms=5.5,
                mfc="white", mec=colour, mew=1.5, color=colour,
                elinewidth=1.1, capsize=0, zorder=6)

legend = ax.get_legend_handles_labels()[0] + [
    Line2D([], [], ls="none", marker="o", color=GUIDE, ms=5,
           label="measured by measly"),
    Line2D([], [], ls="none", marker="D", mfc="white", mec=GUIDE, mew=1.5,
           ms=5.5, label="actual, after collecting more"),
]
ax.legend(handles=legend, frameon=False, fontsize=9, loc="upper right",
          handlelength=1.6, borderaxespad=0)
ax.figure.tight_layout()
ax.figure.savefig("docs/learning-curve.png", dpi=120)
print("\nwrote docs/learning-curve.png")

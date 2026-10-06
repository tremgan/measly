# measly🍽️

More data is always better, but how much better?

measly tries to estimate whether your model is in a data-limited regime or not (i.e. would measuring more samples actually improve performance), for applications where data acquisition is slow and expensive, including but not limited to the life sciences.

## Problem Definition 

measly fits many many models to bootstrapped datasets at different fractions of the orginial datset to create an ensemble model.

Each ensemble member is then fitted with a scaling law. The default is `pow4`:

```
L(n) = L∞ + A · (n + d)^(−α)
```

`pow3`, the three-parameter form most of the literature quotes, is the same
thing with `d` fixed at zero:

```
L(n) = L∞ + A · n^(−α)
```


## Quickstart

```sh
pip install git+https://github.com/tremgan/measly
```

Supply a model with `fit`/`predict` and your data. Everything else has a default.

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from measly import analyse, plot


def ridge(capacity):
    """Ridge that sees only the first `capacity` features."""
    return Pipeline([
        ("capacity",
         ColumnTransformer([("keep", "passthrough", slice(0, capacity))])),
        ("ridge", Ridge()),
    ])


rng = np.random.default_rng(0)
weights = np.concatenate([rng.normal(size=4), np.full(71, 0.05)])
X = rng.normal(size=(3000, 75))
y = X @ weights + rng.normal(0, 1.0, 3000)

# Named, because a Pipeline's repr embeds a memory address.
result = analyse({"ridge(20 features)": ridge(20),
                  "ridge(75 features)": ridge(75)}, X, y, rng=0)
print(result.summary())
plot(result, factor=4.0)
```

```
measly: 2 model(s), 2250 training examples, 30 draws
        pow4, mean_squared_error (lower is better)

  ridge(20 features)
    now (2250 examples)      1.1868
    at 4500 examples         1.1902 [1.1012, 1.3016]
    gain from getting there  -0.0034 [-0.1148, +0.0856]
    held-back check          0.4% error

  ridge(75 features)
    now (2250 examples)      1.0763
    at 4500 examples         1.0851 [0.9886, 1.1469]
    gain from getting there  -0.0087 [-0.0706, +0.0877]
    held-back check          1.9% error
```

![a learning curve for two model capacities](docs/learning-curve.png)

The capacity question in one figure. At 225 examples the 75-feature model is
20% worse. At 2250 it is 9% better. The curves cross near 400.

That crossover is what a pilot study hides. Measured only at 225 examples you
would have picked the small model, and been wrong about every later round. A
model's rank at one sample size does not give you its rank at the next.

Both gains come out indistinguishable from zero, and their intervals say so.
That is the answer: by 2250 examples the capacity decision has already paid
off, and more data buys little.

## Does it work?

The process above is synthetic, so the projection can be audited instead of
trusted. [`examples/capacity.py`](examples/capacity.py) draws the extra rows
from the same process, refits at four sizes past the measured range, and scores
on 20,000 held-out rows — the open diamonds in the figure. The larger sets
extend the pilot rather than replacing it, because that is what collecting
more data means. Only the new rows are redrawn across the ten repeats.

```
               model  examples            actual  projected           interval   error
  ridge(20 features)      2812   1.1419 +-0.0019     1.1909   [1.1012, 1.3059]    4.3%
  ridge(20 features)      3375   1.1397 +-0.0020     1.1905   [1.1012, 1.3038]    4.5%
  ridge(20 features)      3938   1.1379 +-0.0023     1.1903   [1.1012, 1.3025]    4.6%
  ridge(20 features)      4500   1.1364 +-0.0020     1.1902   [1.1012, 1.3016]    4.7%
  ridge(75 features)      2812   1.0256 +-0.0031     1.0871   [0.9987, 1.1549]    6.0%
  ridge(75 features)      3375   1.0204 +-0.0031     1.0859   [0.9936, 1.1513]    6.4%
  ridge(75 features)      3938   1.0162 +-0.0033     1.0854   [0.9906, 1.1488]    6.8%
  ridge(75 features)      4500   1.0133 +-0.0028     1.0851   [0.9886, 1.1469]    7.1%
```

All eight land inside their intervals, never more than 7.1% from the median,
and every one of them errs pessimistic.

Read the projected loss as conservative rather than central, and the more so
the further the model is from its floor. Over eight pilots the projection at
2x sat above what actually happened in 8 of 8 for the steeply descending model
and 6 of 8 for the nearly flat one. It is not a guaranteed bound, but measly
is most pessimistic exactly when the answer is "collect more": for that steep
model it understated the gain every single time.

The audit stops at twice the measured pool. The held-back check fits on the
smaller sizes and predicts the largest two, a stretch of about 1.9x, so that
is as far as the method carries evidence. Projecting ten times out would
return a number and support none of it.

The diamonds sit a visible step below the curve. That is expected, and it has
two causes. measly trains on bootstrap resamples, so a 2250-example fit sees
about 1422 unique rows — `N(1 - e^-1)` — and scores worse than a model handed
2250 real ones. That part never changes sign. measly also scores on a slice of
your own pilot rather than on the population, which over six pilots shifted the
level by -1.2% to +4.6%. The first is the bias, the second is the scatter.

Their error bars are smaller than the markers that carry them. Ten independent
collections at one size vary by 0.0019 to 0.0033, while the band
spans about 0.2. The band is conservative by roughly two orders of magnitude, which is the
safe direction to be wrong in but worth knowing. Draws are bootstrap, so a "2250-example" fit sees about
63% unique rows and scores worse than a real one would. measly therefore
understates what more data buys rather than overstating it.

`analyse` always checks itself. It refits the curves on the smaller sample
sizes, predicts the largest ones you already measured, and reports the error.
Above 15% the projection MUST NOT be relied on: the shape fits your data but
does not predict it.

`analyse(...).results` is an `xarray.DataArray` over `(fraction, model, draw)`
if you want the raw measurements.

## References

- [The Shape of Learning Curves: a Review](https://arxiv.org/abs/2103.10948). Viering & Loog, 2021.
- [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409). Hestness et al., 2017.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). Kaplan et al., 2020.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). Hoffmann et al., 2022.
- [Revisiting Neural Scaling Laws in Language and Vision](https://arxiv.org/abs/2209.06640). Alabdulmohsin et al., 2022.
- [Broken Neural Scaling Laws](https://github.com/ethancaballero/broken_neural_scaling_laws). Caballero et al., 2022.
- [How Much More Data Do I Need?](https://research.nvidia.com/labs/toronto-ai/estimatingrequirements/). Mahmood et al., 2022.
- [Estimation of Predictive Performance in High-Dimensional Data Settings using Learning Curves](https://arxiv.org/abs/2206.03825). Goedhart et al., 2022.
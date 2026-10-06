# measly🍽️

More data is always better, but how much better?

measly tries to estimate whether your model is in a data-limited regime or not (i.e. would measuring more samples actually improve performance), for applications where data acquisition is slow and expensive, including but not limited to the life sciences.

## Problem Definition 

measly fits many many models to random subsets of the dataset at different fractions of its size to create an ensemble model.

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
uv add git+https://github.com/tremgan/measly
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
X = rng.normal(size=(1000, 75))
y = X @ weights + rng.normal(0, 1.0, 1000)

# Named, because a Pipeline's repr embeds a memory address.
result = analyse({"ridge(20 features)": ridge(20),
                  "ridge(75 features)": ridge(75)}, X, y, rng=0)
print(result.summary())
plot(result, factor=4.0)
```

```
measly: 2 model(s), 750 training examples, 100 draws
        pow4, mean_squared_error (lower is better)

  ridge(20 features)
    now (750 examples)       1.0882
    at 1500 examples         1.0717 [0.9312, 1.2301]
    gain from getting there  +0.0062 [+0.0001, +0.0367]
    held-back check          2.1% error

  ridge(75 features)
    now (750 examples)       1.0787
    at 1500 examples         1.0961 [0.9040, 1.3084]
    gain from getting there  +0.0087 [+0.0009, +0.0391]
    held-back check          4.9% error
```

![a learning curve for two model capacities](docs/learning-curve.png)

The capacity question in one figure. At 75 examples the 75-feature model is
more than three times worse. At 750 it is 1% better. The curves cross near 400.

That crossover is what a pilot study hides. Measured only at 75 examples you
would have picked the small model, and been wrong about every later round. A
model's rank at one sample size does not give you its rank at the next.

Both gains are small but positive, and the larger model gains about 40%
more. That is the answer to "should we collect more": a little, and it is the
high-capacity model that benefits.

The gain interval is far tighter than the loss interval above it, which looks
inconsistent until you see why. Every size in a draw is scored on the same test
set, so that draw's whole curve shifts bodily up or down. That offset dominates
the loss band, and cancels in the gain, because each curve is compared against
its own level rather than an average over draws.

## Does it work?

The process above is synthetic, so the projection can be audited instead of
trusted. [`examples/capacity.py`](examples/capacity.py) draws the extra rows
from the same process, refits at four sizes past the measured range, and scores
on 20,000 held-out rows — the open diamonds in the figure. The larger sets
extend the pilot rather than replacing it, because that is what collecting
more data means. Only the new rows are redrawn across the ten repeats.

```
               model  examples            actual  projected           interval   error
  ridge(20 features)       938   1.1592 +-0.0027     1.0795   [0.9534, 1.2388]    6.9%
  ridge(20 features)      1125   1.1539 +-0.0047     1.0762   [0.9357, 1.2321]    6.7%
  ridge(20 features)      1312   1.1519 +-0.0047     1.0737   [0.9322, 1.2308]    6.8%
  ridge(20 features)      1500   1.1499 +-0.0042     1.0717   [0.9312, 1.2301]    6.8%
  ridge(75 features)       938   1.0736 +-0.0091     1.1035   [0.9201, 1.3113]    2.8%
  ridge(75 features)      1125   1.0594 +-0.0090     1.1006   [0.9139, 1.3095]    3.9%
  ridge(75 features)      1312   1.0515 +-0.0091     1.0963   [0.9079, 1.3087]    4.3%
  ridge(75 features)      1500   1.0466 +-0.0057     1.0961   [0.9040, 1.3084]    4.7%
```

All eight land inside their intervals, never more than 6.9% from the median.

Do not read the projected loss as biased either way. Over eight pilots the
projection at 2x sat above what actually happened in 3 of 8 for the nearly flat
model and 5 of 8 for the steeply descending one. The mean error was +0.4% and
+2.4%, and all sixteen intervals held the truth. The scatter comes from
scoring on a slice of your own pilot, not from a systematic lean.

The audit stops at twice the measured pool. The held-back check fits on the
smaller sizes and predicts the largest two, a stretch of about 1.9x, so that
is as far as the method carries evidence. Projecting ten times out would
return a number and support none of it.

Their error bars are smaller than the markers that carry them. Ten independent
collections at one size vary by 0.0027 to 0.0091, while the band spans about
0.3. The band is conservative by roughly two orders of magnitude, which is the
safe direction to be wrong in but worth knowing. Most of that width is test-set
noise, not uncertainty about the curve.

Subsets are drawn without replacement. Each draw resplits the pool, so even the
full-size fit varies across draws. A bootstrap would add little to that spread
and would bias the level up, because a resample repeats rows.

`analyse` always checks itself. It refits the curves on the smaller sample
sizes, predicts the largest ones you already measured, and reports the error.
Above 15% the projection MUST NOT be relied on: the shape fits your data but
does not predict it.

`analyse(...).results` is an `xarray.DataArray` over `(fraction, model, draw)`
if you want the raw measurements.

## Previous work in this area/Inspiration

- [The Shape of Learning Curves: a Review](https://arxiv.org/abs/2103.10948). Viering & Loog, 2021.
- [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409). Hestness et al., 2017.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). Kaplan et al., 2020.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). Hoffmann et al., 2022.
- [Revisiting Neural Scaling Laws in Language and Vision](https://arxiv.org/abs/2209.06640). Alabdulmohsin et al., 2022.
- [Broken Neural Scaling Laws](https://github.com/ethancaballero/broken_neural_scaling_laws). Caballero et al., 2022.
- [How Much More Data Do I Need?](https://research.nvidia.com/labs/toronto-ai/estimatingrequirements/). Mahmood et al., 2022.
- [Estimation of Predictive Performance in High-Dimensional Data Settings using Learning Curves](https://arxiv.org/abs/2206.03825). Goedhart et al., 2022.

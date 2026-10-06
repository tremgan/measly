# measly🍽️

[![PyPI](https://img.shields.io/pypi/v/measly)](https://pypi.org/project/measly/)

More data helps. measly estimates how much.

measly estimates whether a model is data-limited, meaning whether measuring more samples would improve its performance. It is built for fields where each measurement is slow and expensive, such as the life sciences.

Note that measly's sweep trains `'n_draws' * 'n_models' * 'n_fractions'` models, which can be extremely computationally expensive. The premise is that this is (should be) still cheaper than collecting new data. If this is not the case, measly isn't appropriate for your use case.

## Problem Definition 

measly fits a model to many random subsets of the dataset at different fractions of its size. Each draw gives one learning curve, and each curve is fitted with a scaling law. The default is `pow4`:

$$L(n) = L_\infty + A\,(n + d)^{-\alpha}$$

`pow3`, the three-parameter form most of the literature quotes, is `pow4`
with $d$ fixed at zero:

$$L(n) = L_\infty + A\,n^{-\alpha}$$


![a learning curve for two model capacities](docs/learning-curve.png)

At 75 examples the 75-feature model is more than three times worse. At 750 it
is 1% better. The curves cross near 400.

A pilot study hides that crossover. Measured only at 75 examples, you would
have picked the small model and been wrong about every later round. A model's
rank at one sample size says little about its rank at the next.

Both gains are small but positive, and the larger model gains about 40% more.
So collecting more data helps a little, and mostly the high-capacity model.

The gain interval is far tighter than the loss interval. Every size
in a draw is scored on the same test set, so that draw's whole curve shifts up
or down together. The shift dominates the loss band and cancels in the gain,
because each curve is compared against its own level, not an average over
draws.

## Does it work?

The example data is synthetic, so the projection can be checked against the
truth. [`examples/capacity.py`](examples/capacity.py) draws extra rows from the
same process, refits at four sizes past the measured range, and scores on
20,000 held-out rows. These are the open diamonds in the figure. The larger
sets extend the pilot, as collecting more data would. Only the new rows are
redrawn across the ten repeats.

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

The projected loss shows no consistent bias. Over eight pilots the projection
at 2x sat above what actually happened in 3 of 8 for the nearly flat model and
5 of 8 for the steeply descending one. The mean error was +0.4% and +2.4%, and
all sixteen intervals held the truth. The scatter comes from scoring on a slice
of your own pilot.

The audit stops at twice the measured pool. The held-back check fits on the
smaller sizes and predicts the largest two, a stretch of about 1.9x, so that
is as far as the method carries evidence. Projecting ten times out would
return a number and support none of it.

The error bars on the diamonds are smaller than the markers. Ten independent
collections at one size vary by 0.0027 to 0.0091, while the band spans about
0.3, so the band is 30 to 100 times wider. That is the safe direction to err
in. Most of the width is test-set noise, not uncertainty about the curve.

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

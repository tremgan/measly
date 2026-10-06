# measly🍽️

[![PyPI](https://img.shields.io/pypi/v/measly)](https://pypi.org/project/measly/)

> [!WARNING]
> **Work in progress.** This project is under active development. Not really ready for external use.

More data helps is always better, but how much better?

measly estimates whether a model is data-limited, meaning whether measuring more samples would improve its performance. It is built for fields where each measurement is slow and expensive, such as the life sciences.

Note that measly's sweep trains `'n_draws' * 'n_models' * 'n_fractions'` models, which can be extremely computationally expensive. The premise is that this is (should be) still cheaper than collecting new data. If this is not the case, measly isn't appropriate for your use case.


## Problem Definition 

measly fits a model to many random subsets of the dataset at different fractions of its size. Each draw gives one learning curve, and each curve is fitted with a scaling law. The default is `pow4`:

$$L(n) = L_\infty + A\,(n + d)^{-\alpha}$$

`pow3`, the three-parameter form most of the literature quotes, is `pow4`
with $d$ fixed at zero:

$$L(n) = L_\infty + A\,n^{-\alpha}$$

## Example

<img width="991" height="662" alt="image" src="https://github.com/user-attachments/assets/2e0892b5-ade2-450f-a241-8a807b61e2e2" />

Here's a complete work-in-progress graph to get a feel for what it's actually doing

## Previous work in this area/Inspiration

- [The Shape of Learning Curves: a Review](https://arxiv.org/abs/2103.10948). Viering & Loog, 2021.
- [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409). Hestness et al., 2017.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). Kaplan et al., 2020.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). Hoffmann et al., 2022.
- [Revisiting Neural Scaling Laws in Language and Vision](https://arxiv.org/abs/2209.06640). Alabdulmohsin et al., 2022.
- [Broken Neural Scaling Laws](https://github.com/ethancaballero/broken_neural_scaling_laws). Caballero et al., 2022.
- [How Much More Data Do I Need?](https://research.nvidia.com/labs/toronto-ai/estimatingrequirements/). Mahmood et al., 2022.
- [Estimation of Predictive Performance in High-Dimensional Data Settings using Learning Curves](https://arxiv.org/abs/2206.03825). Goedhart et al., 2022.

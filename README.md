# measly🍽️

More data is always better, but how much better?

measly tries to estimate whether your model is in a data-limited regime or not (i.e. would measuring more samples actually improve performance), for applications where data acquisition is slow and expensive, including but not limited to the life sciences.

## Problem Definition 

measly fits many many models to bootstrapped datasets at different fractions of the orginial datset to create an ensemble model.

```
L(n) = L∞ + A · n^(−α)
```

## References

- [The Shape of Learning Curves: a Review](https://arxiv.org/abs/2103.10948). Viering & Loog, 2021.
- [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409). Hestness et al., 2017.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361). Kaplan et al., 2020.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). Hoffmann et al., 2022.
- [Revisiting Neural Scaling Laws in Language and Vision](https://arxiv.org/abs/2209.06640). Alabdulmohsin et al., 2022.
- [Broken Neural Scaling Laws](https://github.com/ethancaballero/broken_neural_scaling_laws). Caballero et al., 2022.
- [How Much More Data Do I Need?](https://research.nvidia.com/labs/toronto-ai/estimatingrequirements/). Mahmood et al., 2022.
- [Estimation of Predictive Performance in High-Dimensional Data Settings using Learning Curves](https://arxiv.org/abs/2206.03825). Goedhart et al., 2022.

Helpers related to (multi)normal and a small van der Corput low-discrepancy generator.
"""
import numpy as np


def mvn_rvs(mean, cov, size=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    mean = np.asarray(mean)
    return rng.multivariate_normal(mean, cov, size=size)


def mvn_logpdf(x, mean, cov):
    x = np.atleast_2d(x)
    mean = np.asarray(mean)
    d = mean.size
    xc = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    quad = np.sum(xc @ inv * xc, axis=1)
    const = -0.5 * (d * np.log(2 * np.pi) + logdet)
    return const - 0.5 * quad


def van_der_corput(n: int, base: int = 2) -> float:
    """Return nth element of van der Corput sequence in [0,1).
    Simple single-value implementation commonly used for quasi-random draws.
    """
    v = 0
    denom = 1
    while n > 0:
        denom *= base
        n, remainder = divmod(n, base)
        v += remainder / denom
    return v

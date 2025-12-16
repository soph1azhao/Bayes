
Power calculations and small wrapper API expected by final codebase:
- power: Monte Carlo Bayesian power (alias of simulate_bayes_power)
- power_1d: same as power but accepts a single-element list or scalar
- constraint / constraint_1d: return boolean whether power >= threshold
"""
from typing import Union
import math
import numpy as np
from .rndgenerator import get_rng


def _simulate_bayes_power(n: int,
                          delta: float,
                          sigma: float = 1.0,
                          prior_var: float = 1.0,
                          threshold: float = 0.975,
                          nsim: int = 2000,
                          seed: int = None) -> float:
    rng = get_rng(seed)
    post_var = 1.0 / (n / (sigma**2) + 1.0 / prior_var)
    post_sd = math.sqrt(post_var)
    successes = 0
    for _ in range(nsim):
        sample_mean = rng.normal(loc=delta, scale=sigma / math.sqrt(n))
        post_mean = post_var * ((n * sample_mean) / (sigma**2) + 0.0 / prior_var)
        z = post_mean / post_sd
        p_gt0 = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        if p_gt0 > threshold:
            successes += 1
    return successes / nsim


# Public API functions expected by your downstream code

def power(n: int, delta: float, **kwargs) -> float:
    """Alias: compute Bayesian power for sample size n."""
    return _simulate_bayes_power(int(n), float(delta), **kwargs)


def power_1d(x: Union[int, float, list], delta: float, **kwargs) -> float:
    """Wrapper that accepts scalar or single-element list like [n]."""
    if isinstance(x, (list, tuple, np.ndarray)):
        n = int(x[0])
    else:
        n = int(x)
    return power(n, delta, **kwargs)


def constraint(n: int, delta: float, threshold: float = 0.975, **kwargs) -> bool:
    """Return True if power >= threshold (useful as a constraint function)."""
    p = power(int(n), float(delta), threshold=threshold, **kwargs)
    return float(p) >= threshold


def constraint_1d(x: Union[int, float, list], delta: float, threshold: float = 0.975, **kwargs) -> bool:
    if isinstance(x, (list, tuple, np.ndarray)):
        n = int(x[0])
    else:
        n = int(x)
    return constraint(n, delta, threshold=threshold, **kwargs)

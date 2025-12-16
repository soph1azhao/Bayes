
"""
"""
from typing import Callable, Optional
import numpy as np

class BayesSampler:
    """Minimal Metropolis-Hastings sampler exposing a simple API.

    Usage:
        sampler = BayesSampler(logp, dim)
        samples = sampler.sample(x0, n_samples=1000, burn_in=100, thin=1)
    """
    def __init__(self, logp: Callable[[np.ndarray], float], dim: int, step_scale: float = 0.5, rng: Optional[np.random.Generator] = None):
        self.logp = logp
        self.dim = int(dim)
        self.step_scale = float(step_scale)
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample(self, x0: np.ndarray, n_samples: int, burn_in: int = 0, thin: int = 1) -> np.ndarray:
        x = np.asarray(x0, dtype=float).copy()
        current_logp = float(self.logp(x))
        samples = []
        total_iters = burn_in + n_samples * thin
        for i in range(total_iters):
            prop = x + self.rng.normal(scale=self.step_scale, size=self.dim)
            prop_logp = float(self.logp(prop))
            # acceptance
            try:
                a = np.exp(prop_logp - current_logp)
            except OverflowError:
                a = float('inf') if (prop_logp > current_logp) else 0.0
            if self.rng.random() < min(1.0, a):
                x = prop
                current_logp = prop_logp
            if i >= burn_in and ((i - burn_in) % thin == 0):
                samples.append(x.copy())
        return np.asarray(samples)

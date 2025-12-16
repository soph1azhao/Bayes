
"""Random generator utilities (wraps numpy Generator) and exported module-level helpers.
"""
from typing import Optional
import numpy as np


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def randint(low: int, high: int, size=1, seed: Optional[int] = None):
    rng = get_rng(seed)
    return rng.integers(low, high, size=size)


def normal(loc=0.0, scale=1.0, size=None, seed: Optional[int] = None):
    rng = get_rng(seed)
    return rng.normal(loc=loc, scale=scale, size=size)

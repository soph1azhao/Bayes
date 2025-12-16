
"""Provide ZSQP and ZSQP_1d classes (simple wrappers around grid search or scipy when available).
The names mimic the interface your downstream code expects.
"""
from typing import Callable, Sequence, Tuple
import numpy as np


class _BaseOptimizer:
    def __init__(self, objective: Callable[[Sequence[float]], float]):
        self.objective = objective

class ZSQP(_BaseOptimizer):
    """Optimizer for integer/continuous designs. Provides maximize_over_integer_n.
    Uses grid search for robustness on clusters without SciPy.
    """
    def maximize_over_integer_n(self, n_min: int, n_max: int, step: int = 1) -> Tuple[int, float]:
        best_n = n_min
        best_val = -float('inf')
        for n in range(n_min, n_max + 1, step):
            try:
                val = self.objective([n])
            except Exception:
                val = -float('inf')
            if val > best_val:
                best_val = val
                best_n = n
        return best_n, best_val

class ZSQP_1d(ZSQP):
    """1d-specialized alias (keeps same API)."""
    pass

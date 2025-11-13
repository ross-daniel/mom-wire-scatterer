import numpy as np
from typing import Callable, Tuple


def numerical_integral(f: Callable[[np.ndarray], np.ndarray], degree: int, bounds: Tuple[float, float]) -> float:
    a, b = bounds[0], bounds[1]
    x, w = np.polynomial.legendre.leggauss(degree)
    t = 0.5 * (x + 1) * (b - a) + a
    return np.sum(w * f(t)) * 0.5 * (b - a)

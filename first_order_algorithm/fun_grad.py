from numba import njit
import numpy as np


@njit()
def f(x):
    return (11 - x[0] - x[1]) ** 2 + (1 + x[0] + 10 * x[1] - x[0] * x[1]) ** 2


@njit()
def df(x):
    res = np.zeros(2)
    res[0] = -2 * (11 - x[0] - x[1]) + 2 * (1 + x[0] + 10 * x[1] - x[0] * x[1]) * (1 - x[1])
    res[1] = -2 * (11 - x[0] - x[1]) + 2 * (1 + x[0] + 10 * x[1] - x[0] * x[1]) * (10 - x[0])
    return res

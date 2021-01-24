"""
step1: choose a starting point x0 and set k = 0
step2: determine a descent direction d_k
step3: determine step size by line search, choose step size tau_k > 0
step4: update x[k + 1] = x[k] + tau_k * d_k, k += 1
repeat until stopping criterion is satisfied

Linear approximation:
Suppose that f is differentiable and \partial f(x) \neq 0
Then we have linear approximation for all delta x \neq 0: f(x + \delta x) \approx f(x) + \partial f(x)^T \
"""

import numpy as np
from numba import njit
import sympy as sym


def f(x):
    return (x[0] - 3) ** 2 + (x[1] - 2) ** 2


def df(x):
    return np.array([2 * (x[0] - 3), 2 * (x[1] - 2)])


def f(x):
    return (11 - x[0] - x[1]) ** 2 + (1 + x[0] + 10 * x[1] - x[0] * x[1]) ** 2


def df(x):
    return np.array([-2 * (11 - x[0] - x[1]) + 2 * (1 + x[0] + 10 * x[1] - x[0] * x[1]) * (1 - x[1]),
                     -2 * (11 - x[0] - x[1]) + 2 * (1 + x[0] + 10 * x[1] - x[0] * x[1]) * (10 - x[0])])
# def symbolic function
# x1, x2 = sym.symbols('x1, x2')
# f = (x1 - 3) ** 2 + (x2 - 3) ** 2
# df1 = sym.diff(f, x1)
# df2 = sym.diff(f, x2)

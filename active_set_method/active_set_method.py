"""
Active set method  solves linearly  constrained  general quadratic program
"""
import numpy as np


def get_active_set(x):
    return np.where(x == 0)[0]


def qp_econ(inv_q, p, a, b):
    """
    solves equality constrained quadratic program
    :param inv_q:
    :param p:
    :param a:
    :param b:
    :return: solution x and multiplier u
    """
    large_factor = np.linalg.inv(np.dot(np.dot(a, inv_q), a.T))
    rhs = b + np.dot(np.dot(a, inv_q), p)
    x = -np.dot(inv_q, p) + np.dot(np.dot(np.dot(inv_q, a.T), large_factor), rhs)
    u = -np.dot(large_factor, rhs)
    return x, u

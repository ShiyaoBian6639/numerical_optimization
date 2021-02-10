import numpy as np
from numba import njit


@njit()
def householder_reflection(v):
    n = len(v)
    identity = np.eye(n)
    beta = 2 / np.dot(v.T, v)
    p = identity - beta * v.dot(v.T)
    return p


@njit()
def unit_transformation(x):
    """
    transforms any given vector x into e1, where e1 = [gamma, 0, 0, 0, ..., 0]
    :param x: a vector with shape n * 1
    :return: householder reflection with n * n
    """
    n = len(x)
    target_vec = np.zeros((n, 1))
    target_vec[0] = 1
    v = x - np.linalg.norm(x) * target_vec
    return householder_reflection(v)

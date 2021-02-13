import numpy as np
from numba import njit
from math import sqrt


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


@njit()
def givens_rotation(a, b):
    if b == 0:
        c = 1
        s = 0
    elif abs(b) > abs(a):
        tau = - a / b
        s = 1 / sqrt(1 + tau * tau)
        c = s * tau
    else:
        tau = - b / a
        c = 1 / sqrt(1 + tau * tau)
        s = c * tau
    res = np.empty((2, 2))
    res[0, 0] = c
    res[0, 1] = -s
    res[1, 0] = s
    res[1, 1] = c
    return res


@njit()
def apply_givens_rotation(A, row, col):
    a = A[row - 1, col]
    b = A[row, col]
    givens = givens_rotation(a, b)
    temp = A[(row - 1): (row + 1), :]
    A[(row - 1): (row + 1), :] = givens.dot(temp)

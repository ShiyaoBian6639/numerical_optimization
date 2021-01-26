"""
Active set method  solves linearly  constrained  general quadratic program
"""
import numpy as np


def get_active_set(x):
    return np.where(x == 0)[0]


def get_econ(a, x):
    num_var = len(x)
    active_set = get_active_set(x)
    new_con = np.zeros((len(active_set), num_var))
    for ind, val in enumerate(active_set):
        new_con[ind, val] = 1
    equal_con = np.vstack((a, new_con))
    num_con = len(a) + len(active_set)
    b = np.zeros(num_con)

    return active_set, equal_con, b


def qp_econ(inv_q, p, a, b):
    """
    solves equality constrained quadratic program
    :param inv_q:
    :param p:
    :param a:
    :param b:
    :return: direction d and multiplier u
    """
    large_factor = np.linalg.inv(np.dot(np.dot(a, inv_q), a.T))
    rhs = b + np.dot(np.dot(a, inv_q), p)
    d = -np.dot(inv_q, p) + np.dot(np.dot(np.dot(inv_q, a.T), large_factor), rhs)
    u = -np.dot(large_factor, rhs)
    return d, u

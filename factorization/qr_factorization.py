"""
A = QR
Q: orthogonal matrix
R: upper triangular matrix
"""
import numpy as np
from matrix_computations.householder_reflection import unit_transformation
from numba import njit


# m, n = 10, 14
# A = np.random.random((m, n))


def qr_fact(a):
    m, n = a.shape
    if m > n:
        a = a.T
        m, n = a.shape
    q, r = np.linalg.qr(a.T, mode='complete')
    # np.allclose(q.dot(r), a.T)  # q dot r restores the matrix A.T
    y = q[:, :m]
    z = q[:, m:]
    r = r[:m, :]
    return y, z, r


@njit()
def qr_add_row(q1, q2, r, a):
    n, m = q1.shape
    q1t_a = np.dot(q1.T, a)
    q2t_a = np.dot(q2.T, a)
    gamma = np.linalg.norm(q2t_a)
    _q = unit_transformation(q2t_a)
    r_update = np.empty((m + 1, m + 1))
    r_update[:m, :m] = r
    r_update[:m, m] = q1t_a[:, 0]
    r_update[m, m] = gamma
    r_update[m, :m] = 0
    q2_update = q2.dot(_q.T)
    # move the first column in q2 to the last column of q1 (change of null space)
    q1_update = np.zeros((n, m + 1))
    q1_update[:, :m] = q1
    q1_update[:, m] = q2_update[:, 0]
    return q1_update, q2_update[:, 1:(n - m)], r_update


def qr_drop_row():
    pass

"""
A = QR
Q: orthogonal matrix
R: upper triangular matrix
"""
import numpy as np

# m, n = 10, 14
# A = np.random.random((m, n))


def qr_fact(a):
    m, n = a.shape
    if m > n:
        a = a.T
        m, n = a.shape
    q, r = np.linalg.qr(a.T, mode='complete')
    np.allclose(q.dot(r), a.T)  # a1 dot a2 restores the matrix A.T
    y = q[:, :m]
    z = q[:, m:]
    r = r[:m, :]
    return y, z, r


def qr_update():
    pass

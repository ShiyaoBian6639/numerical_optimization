import numpy as np


def cholesky_solve(a, b):
    return np.dot(np.linalg.inv(a), b)

import numpy as np
from scipy.sparse import csr_matrix, tril


def cholesky_solve(a, b):
    return np.dot(np.linalg.inv(a), b)


def chol_md():
    # minimum degree ordering heuristic
    pass



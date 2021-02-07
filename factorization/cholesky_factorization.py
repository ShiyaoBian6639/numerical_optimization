import numpy as np
from scipy.sparse import csr_matrix, tril


def cholesky_solve(a, b):
    return np.dot(np.linalg.inv(a), b)


A = csr_matrix([[1, 2, 0, 0, 3], [4, 5, 0, 6, 7], [0, 0, 8, 9, 0]], dtype='int32')
lower = tril(A).tocsr()
print(lower.toarray())
indptr = lower.indptr
indices = lower.indices

from factorization.qr_factorization import qr_fact, qr_add_row, qr_drop_row
from matrix_computations.orthogonalization import apply_givens_rotation
import numpy as np

m, n = 5, 14
A = np.zeros((m, n))
A[0, :] = 1
for i in range(1, m):
    A[i, i - 1] = 1
q, q1, q2, r = qr_fact(A)
# test qr update
a = np.zeros((n, 1))
a[6, 0] = 1
_A = np.vstack((A, a.T))
q_ben, q1_ben, q2_ben, r_ben = qr_fact(_A)

# update procedure
q1_add, q2_add, r_add = qr_add_row(q1, q2, r, a)
print(np.allclose(q1_ben, q1_add))
print(np.allclose(q2_ben, q2_add))
print(np.allclose(r_ben, r_add))

# %timeit q1_ben, q2_ben, r_ben = qr_fact(_A)
# %timeit q1_add, q2_add, r_add = qr_add_row(q1, q2, r, a)

# delete procedure
_A = np.delete(A, 1, axis=0)
q_ben, q1_ben, q2_ben, r_ben = qr_fact(_A)
h = np.dot(q.T, _A.T)
r_drop = qr_drop_row(q, _A, 1)
# apply givens rotation on h
# # qr factorization
# A = np.array([
#     [0, -1, 1],
#     [4, 2, 0],
#     [3, 4, 0],
# ], dtype=float)

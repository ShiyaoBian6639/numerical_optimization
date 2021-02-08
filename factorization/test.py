from factorization.qr_factorization import qr_fact, qr_update_add, qr_update_remove
import numpy as np

m, n = 5, 14
A = np.zeros((m, n))
A[0, :] = 1
for i in range(1, m):
    A[i, i - 1] = 1
q1, q2, r = qr_fact(A)
# test qr update
a = np.zeros((1, n))
a[0, 4] = 1
_A = np.vstack((A, a))
q1_ben, q2_ben, r_ben = qr_fact(_A)

# update procedure
q1at = np.dot(q1.T, a.T)  # m * 1
q2at = np.dot(q2.T, a.T)  # (n - m) * 1
gamma = np.linalg.norm(q2at)

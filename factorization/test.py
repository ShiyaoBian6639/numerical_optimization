from factorization.qr_factorization import qr_fact, qr_add_row, qr_drop_row
import numpy as np

m, n = 5, 14
A = np.zeros((m, n))
A[0, :] = 1
for i in range(1, m):
    A[i, i - 1] = 1
q1, q2, r = qr_fact(A)
# test qr update
a = np.zeros((n, 1))
a[4, 0] = 1
_A = np.vstack((A, a.T))
q1_ben, q2_ben, r_ben = qr_fact(_A)

# update procedure
q1_add, q2_add, r_add = qr_add_row(q1, q2, r, a)
print(np.allclose(q1_ben, q1_add))
print(np.allclose(q2_ben, q2_add))
print(np.allclose(r_ben, r_add))

%timeit q1_ben, q2_ben, r_ben = qr_fact(_A)
%timeit q1_add, q2_add, r_add = qr_add_row(q1, q2, r, a)
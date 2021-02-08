import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from linear_solve.utils import lower_triangular_solve
# method 1: build compressed sparse column
row = np.array([0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4])
col = np.array([0, 0, 1, 0, 2, 1, 2, 3, 1, 3, 4])
data = np.array([1, 2, 4, 5, 6, 4, 8, 5, 2, 8, 3])
A = csr_matrix((data, (row, col)), shape=(5, 5))
print(A.toarray())
b = np.array([1, 2, 3, 4, 5])


ind_ptr = A.indptr
indices = A.indices
data = A.data

# sparse decomposition

# %timeit np.dot(np.linalg.inv(A.toarray()), b)

# %timeit lower_triangular_solve(indices, ind_ptr, data, b)

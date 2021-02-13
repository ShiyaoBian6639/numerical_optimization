from matrix_computations.orthogonalization import householder_reflection, unit_transformation
import numpy as np

n = 100
# v = np.random.random((n, 1))
# p = householder_reflection(v)

# transform any x to multiples of e
identity = np.eye(n)
target_vec = identity[:, 0][np.newaxis].T
x = np.random.random((n, 1))
v = x - np.linalg.norm(x) * target_vec
p = householder_reflection(v)
s = p.dot(x)


p = unit_transformation(x)
s = p.dot(x)  # s is [r, 0, 0, 0, ..., 0]


import numpy as np

A = np.array([
    [8, 6, 1, 1, 0, 0],
    [4, 2, 1.5, 0, 1, 0],
    [2, 1.5, 0.5, 0, 0, 1],
])

b = np.array([48.0, 20.0, 8.0])
c = np.array([60, 30, 20, 0, 0, 0])
x = np.zeros(6)
x[3:6] = np.array([48.0, 20.0, 8.0])
X = np.zeros((6, 6))
np.fill_diagonal(X, x)

lmd = np.zeros(3)
s = c.copy()
S = np.zeros((6, 6))
np.fill_diagonal(S, s)
e = np.ones(6)


def naive_newton_step(A):
    m, n = A.shape
    I = np.eye(n)
    jacobian_dim = 2 * n + m
    jacobian = np.zeros((jacobian_dim, jacobian_dim))
    jacobian[n: (n + m), : n] = A
    jacobian[(m + n):, :n] = S
    jacobian[:n, n:(m + n)] = A.T
    jacobian[:n, (m + n):] = I
    jacobian[(m + n):, (m + n):] = X
    inv_jacobian = np.linalg.inv(jacobian)

    rb = A.dot(x) - b
    rc = np.dot(A.T, lmd) + s - c
    comp_slack = np.dot(X.dot(S), e)
    f = np.zeros(jacobian_dim)
    f[:m] = -rb
    f[m:(m + n)] = -rc
    f[(m + n):] = -comp_slack

    step = inv_jacobian.dot(f)


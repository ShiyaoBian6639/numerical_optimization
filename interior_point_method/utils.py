import numpy as np


def dakota_instance():
    A = np.array([
        [8, 6, 1, 1, 0, 0],
        [4, 2, 1.5, 0, 1, 0],
        [2, 1.5, 0.5, 0, 0, 1],
    ])

    b = np.array([48.0, 20.0, 8.0])
    c = np.array([60, 30, 20, 0, 0, 0])
    var_ind = 3
    return A, b, c, var_ind


def ipm_initial_feasible(a, b, c):
    """
    recover the dual solution from primal solution using complementary conditions
    :param a: constraint coefficient
    :param b:  rhs
    :param c: objective coefficient
    :return: x, s, X, S, e, mu, lmd
    """
    m, n = a.shape
    inv_aat = np.linalg.inv(a.dot(a.T))
    _x = np.dot(np.dot(a.T, inv_aat), b)
    lmd = inv_aat.dot(a).dot(c)
    _s = c - np.dot(a.T, lmd)
    x_adj = max(-3 / 2 * min(_x), 0)
    s_adj = max(-3 / 2 * min(_s), 0)

    __x = 1 / 2 * _x.dot(_s) / _s.sum()
    __s = 1 / 2 * _x.dot(_s) / _x.sum()
    x = _x + x_adj + __x
    s = _s + s_adj + __s
    X = np.zeros((n, n))
    np.fill_diagonal(X, x)
    S = np.zeros((n, n))
    np.fill_diagonal(S, s)
    e = np.ones(n)
    mu = x.dot(s) / n

    return x, s, X, S, e, mu, lmd


def predictor_step(A, b, c, x, s, lmd, X, S, e, sigma):
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

    duality_measure = x.dot(s) / n
    rb = A.dot(x) - b
    rc = np.dot(A.T, lmd) + s - c
    comp_slack = np.dot(X.dot(S), e)  # - sigma * duality_measure * e
    f = np.zeros(jacobian_dim)
    f[:m] = -rb
    f[m:(m + n)] = -rc
    f[(m + n):] = -comp_slack

    step = inv_jacobian.dot(f)
    delta_x = step[:n]
    delta_lmd = step[n:(m + n)]
    delta_s = step[(m + n):]
    return delta_x, delta_lmd, delta_s, inv_jacobian


def corrector_step(delta_x, delta_s, inv_jacobian, ):
    jacobian_dim = inv_jacobian

    pass


def step_length(x, s, delta_x, delta_s):
    n = len(x)
    max_primal_step_len = np.inf
    max_dual_step_len = np.inf
    for i in range(n):
        if delta_x[i] < 0:
            primal_step_len = - x[i] / delta_x[i]
            if primal_step_len < max_primal_step_len:
                max_primal_step_len = primal_step_len

        if delta_s[i] < 0:
            dual_step_len = - s[i] / delta_s[i]
            if dual_step_len < max_dual_step_len:
                max_dual_step_len = dual_step_len
    return min(max_primal_step_len, 1), min(1, max_dual_step_len)

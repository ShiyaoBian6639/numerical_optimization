import numpy as np
from interior_point_method.utils import ipm_initial_feasible, dakota_instance, predictor_step, step_length

A, b, c, var_ind = dakota_instance()  # generate instance
m, n = A.shape
x_var = np.array([1, 1, 1])  # provide initial feasible x_var

x, s, X, S, e, mu, lmd = ipm_initial_feasible(A, b, c)

sigma = 0.5  # duality decrement rate

delta_x, delta_lmd, delta_s = predictor_step(A, b, c, x, s, lmd, X, S, e, sigma)

alpha = step_length(x, s, delta_x, delta_s) / 2
# update
x += alpha * delta_x
lmd += alpha * delta_lmd
s += alpha * delta_s
np.fill_diagonal(X, x)
np.fill_diagonal(S, s)
# sigma *= 0.9

from active_set_method.utils import active_set_method, generate_random_instance
import numpy as np
from gurobipy import Model, GRB

# q = np.eye(3) * 2  # quadratic coefficient
# p = np.array([-2.0, 2.0, 0.0])  # linear coefficient
# x = np.array([100.0, 100.0, -199.0])  # initial feasible solution
# a = np.array([[1.0, 1.0, 1.0]])
# active_set_method(q, p, x, a)

# large instance test

n = 10
q, p, x = generate_random_instance(n)
q = q + 2 * np.eye(n)

x = x / x.sum()
a = np.ones((1, n))
x, cnt, = active_set_method(q, p, x, a)

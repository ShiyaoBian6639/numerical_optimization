from active_set_method.utils import active_set_method, generate_random_instance
import numpy as np
from active_set_method.utils import plot_obj
from gurobipy import Model, GRB

# q = np.eye(3) * 2  # quadratic coefficient
# p = np.array([-2.0, 2.0, 0.0])  # linear coefficient
# x = np.array([100.0, 100.0, -199.0])  # initial feasible solution
# a = np.array([[1.0, 1.0, 1.0]])
# active_set_method(q, p, x, a)

# large instance test

n = 1000
q, p, x = generate_random_instance(n)
q = q + 2 * np.eye(n)
x = np.random.random(n)
x = x / x.sum()
a = np.ones((1, n))
x, cnt, obj_list = active_set_method(q, p, x, a)

plot_obj(obj_list)
# for i in range(n):
#     if abs(x[i] > 1e-8):
#         print(f"x[{i}] = {x[i]}")

from active_set_method.utils import active_set_method, generate_random_instance
import numpy as np
from active_set_method.utils import plot_obj
from gurobipy import Model, GRB

# q = np.eye(3) * 2  # quadratic coefficient
# p = np.array([-2.0, 2.0, 0.0])  # linear coefficient
# x = np.array([100.0, 100.0, -199.0])  # initial feasible solution
# a = np.array([[1.0, 1.0, 1.0]])
# active_set_method(q, p, x, a)

# s.Wright example
x = np.zeros(3)
a = np.array([[1, 0, 1], [0, 1, 1]])
b = np.array([3, 0])
c = np.array([-8, -3, -3])
G = np.array([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
# large instance test

n = 100
q, p, y = generate_random_instance(n)
q = q + 2 * np.eye(n)
y = np.random.random(n)
y = y / y.sum()
a = np.ones((1, n))
y, cnt, obj_list = active_set_method(q, p, y, a)
print(obj_list[-1])
plot_obj(obj_list)
# for i in range(n):
#     if abs(x[i] > 1e-8):
#         print(f"x[{i}] = {x[i]}")

from active_set_method.utils import active_set_method
import numpy as np

# q = np.eye(3) * 2  # quadratic coefficient
# p = np.array([-2.0, 2.0, 0.0])  # linear coefficient
# x = np.array([100.0, 100.0, -199.0])  # initial feasible solution
# a = np.array([[1.0, 1.0, 1.0]])
# active_set_method(q, p, x, a)

# large instance test

n = 100
temp = 10 * np.random.random((n, n))
q = temp + temp.T
p = np.random.random(n)
x = np.random.random(n)
x = x / x.sum()
a = np.ones((1, n))
active_set_method(q, p, x, a)

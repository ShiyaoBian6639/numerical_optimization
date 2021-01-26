from active_set_method.utils import active_set_method
import numpy as np

q = np.eye(3) * 2  # quadratic coefficient
p = np.array([-2.0, 2.0, 0.0])  # linear coefficient
inv_q = np.linalg.inv(q)

x = np.array([0.0, 1.0, 0.0])  # initial solution
a = np.array([[1.0, 1.0, 1.0]])
active_set_method(q, p, x, a)


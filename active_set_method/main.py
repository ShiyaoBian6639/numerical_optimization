from active_set_method.utils import qp_econ, get_econ
import numpy as np

Q = np.eye(3) * 2  # quadratic coefficient
p = np.array([-2.0, 2.0, 0.0])  # linear coefficient
inv_q = np.linalg.inv(Q)

x = np.array([0.0, 1.0, 0.0])  # initial solution
a = np.array([[1.0, 1.0, 1.0]])
active_set, a, b = get_econ(a, x)  # active set of current solution


sub_p = p + np.dot(Q, x)  # solve sub problem
d, u = qp_econ(inv_q, sub_p, a, b)

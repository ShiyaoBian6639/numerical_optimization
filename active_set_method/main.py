import active_set_method.active_set_method as asm
import numpy as np

Q = np.eye(3) * 2
p = np.array([-2.0, 2.0, 0.0])
inv_q = np.linalg.inv(Q)
a = np.array([
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0]
])
b = np.array([0.0, 0.0, 0.0])

import numpy as np
from active_set_method.utils import get_econ, get_active_set_bool, qp_econ, TOLERANCE, get_max_multiplier, \
    drop_active_set, append_active_set, compute_step_length


# @njit()
def naive_active_set_method(q, p, x, a):
    eig_val, eig_mat = np.linalg.eig(q)
    if min(eig_val) < 0:
        print("problem is not positive semi definite")
        return 0, 0, 0
    original_constraint = a.copy()
    inv_q = np.linalg.inv(q)
    num_con = len(a)
    active_set, a, b = get_econ(original_constraint, x)  # active set of current solution
    active_set_bool = get_active_set_bool(active_set, x)
    step_count = 0
    obj_list = []
    while True:
        sub_p = p + np.dot(q, x)  # solve sub problem
        d, u = qp_econ(inv_q, sub_p, a, b)
        # d, u = qp_econ_null_space(a, x, b, q, sub_p)
        step_count += 1
        # print(f"step_count is {step_count}")
        direction_norm = np.linalg.norm(d)
        # print(f"direction norm: {direction_norm}")
        if direction_norm < TOLERANCE:
            max_multiplier = get_max_multiplier(u, num_con, active_set, active_set_bool)
            if max_multiplier < 0:
                return x, step_count, np.array(obj_list)
            else:  # update the active set
                active_set, a, b, active_set_bool = drop_active_set(active_set, a, b, max_multiplier, num_con,
                                                                    active_set_bool)
        else:  # compute the step length
            step_length, append_index = compute_step_length(x, d, active_set_bool)
            # print(f"step length is {step_length}")
            x += step_length * d
            minx = min(x)
            if append_index >= 0:  # append new element "append_index" to the current active set
                active_set, a, b, active_set_bool = append_active_set(active_set, a, b, append_index, active_set_bool)

            obj = 0.5 * np.dot(np.dot(x.T, q), x) + np.dot(p, x)
            obj_list.append(obj)

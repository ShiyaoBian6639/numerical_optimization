import numpy as np
from active_set_method.utils import get_econ, get_active_set_bool, qp_econ_null_space, TOLERANCE, get_max_multiplier, \
    drop_active_set, append_active_set, compute_step_length, qp_econ_factor
from factorization.qr_factorization import qr_fact, qr_add_row, qr_drop_row


# @njit()
def efficient_active_set_method(q, p, x, a):
    eig_val, eig_mat = np.linalg.eig(q)
    if min(eig_val) < 0:
        print("problem is not positive semi definite")
        return 0, 0, 0
    original_constraint = a.copy()
    num_con = len(a)
    active_set, a, b = get_econ(original_constraint, x)  # active set of current solution
    active_set_bool = get_active_set_bool(active_set, x)
    step_count = 0
    obj_list = []
    while True:
        sub_p = p + np.dot(q, x)  # solve sub problem
        d, u = qp_econ_null_space(a, x, b, q, sub_p)
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
            print(obj)
            obj_list.append(obj)
        print(f"sum x: {x.sum()}")


def active_set_method(q, p, x, a):
    eig_val, eig_mat = np.linalg.eig(q)
    if min(eig_val) < 0:
        print("problem is not positive semi definite")
        return 0, 0, 0
    original_constraint = a.copy()
    num_con = len(a)
    active_set, a, b = get_econ(original_constraint, x)  # active set of current solution
    active_set_bool = get_active_set_bool(active_set, x)

    q, y, z, r = qr_fact(a)  # qr factorization of matrix a
    x, step_count, obj_list = active_set_update(q, p, x, a, y, z, r, b, num_con, active_set, active_set_bool)
    return x, step_count, obj_list

def active_set_update(q, p, x, a, y, z, r, b, num_con, active_set, active_set_bool):
    step_count = 0
    obj_list = []
    while True:
        sub_p = p + np.dot(q, x)  # solve sub problem
        # d, u = qp_econ_null_space(a, x, b, q, sub_p)
        d, u = qp_econ_factor(a, y, z, r, x, b, q, sub_p)
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
                q_pre = np.hstack((y, z))
                r = qr_drop_row(q_pre, a, max_multiplier)
                n_col = y.shape[1] - 1
                y = q_pre[:, :n_col]
                z = q_pre[:, n_col:]
        else:  # compute the step length
            step_length, append_index = compute_step_length(x, d, active_set_bool)
            # print(f"step length is {step_length}")
            x += step_length * d
            minx = min(x)
            if append_index >= 0:  # append new element "append_index" to the current active set
                active_set, a, b, active_set_bool, append_column = append_active_set(active_set, a, b, append_index,
                                                                                     active_set_bool)
                y, z, r = qr_add_row(y, z, r, append_column)
            obj = 0.5 * np.dot(np.dot(x.T, q), x) + np.dot(p, x)
            print(obj)
            obj_list.append(obj)
        print(f"sum x: {x.sum()}")

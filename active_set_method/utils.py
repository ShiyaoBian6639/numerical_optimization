"""
Active set method  solves linearly  constrained  general quadratic program
"""
import numpy as np
from numba import njit

TOLERANCE = 1e-4


# @njit()
def get_active_set(x, tol):
    return np.where(np.abs(x) < tol)[0]


# @njit()
def get_active_set_bool(active_set, x):
    n = len(x)
    active_set_bool = np.zeros(n)
    active_set_bool[active_set] = 1
    return active_set_bool


# @njit()
def get_econ(a, x):
    num_var = len(x)
    active_set = get_active_set(x, TOLERANCE)
    new_con = np.zeros((len(active_set), num_var))
    for ind, val in enumerate(active_set):
        new_con[ind, val] = 1
    equal_con = np.vstack((a, new_con))
    num_con = len(a) + len(active_set)
    b = np.zeros(num_con)

    return active_set, equal_con, b


# @njit()
def qp_econ(inv_q, p, a, b):
    """
    solves equality constrained quadratic program (naive method)
    :param inv_q:
    :param p:
    :param a:
    :param b:
    :return: direction d and multiplier u
    """
    large_factor = np.linalg.inv(np.dot(np.dot(a, inv_q), a.T))
    rhs = b + np.dot(np.dot(a, inv_q), p)
    d = -np.dot(inv_q, p) + np.dot(np.dot(np.dot(inv_q, a.T), large_factor), rhs)
    u = -np.dot(large_factor, rhs)
    return d, u


# @njit()
def get_max_multiplier(u, num_con):
    """
    :param u: multiplier
    :param num_con: number of constraints
    :return: the index of the largest multiplier, -1 indicates the active set algorithm to terminate
    """
    max_multiplier = -np.inf
    max_index = -1
    for i in range(num_con, len(u)):
        val = u[i]
        if val > TOLERANCE and val > max_multiplier:
            max_multiplier = val
            max_index = i
    return max_index - num_con


# @njit()
def drop_active_set(active_set, a, b, index, num_cons, active_set_bool):
    if len(active_set) > 0:
        active_set = np.delete(active_set, index)
    a = np.delete(a, index + num_cons, 0)
    b = np.delete(b, index + num_cons)
    active_set_bool[index] = 0
    return active_set, a, b, active_set_bool


# @njit()
def compute_step_length(x, d, active_set_bool):
    step_length = np.inf
    for i in range(len(d)):
        if active_set_bool[i] == 0 and d[i] < -TOLERANCE and x[i] > TOLERANCE:
            ratio = - x[i] / d[i]
            if ratio < step_length:
                step_length = ratio
    if step_length > 1.0:
        return 1.0
    return my_floor(step_length, 12)


def my_floor(a, precision=0):
    """
    helper function to round 3.1415926 down to 3.1415, differs with np.round
    :param a:
    :param precision:
    :return:
    """
    return np.round(a - 0.5 * 10 ** (-precision), precision)


# @njit()
def active_set_method(q, p, x, a):
    original_constraint = a.copy()
    inv_q = np.linalg.inv(q)
    num_con = len(a)
    active_set, a, b = get_econ(original_constraint, x)  # active set of current solution
    active_set_bool = get_active_set_bool(active_set, x)
    sub_p = p + np.dot(q, x)  # solve sub problem
    d, u = qp_econ(inv_q, sub_p, a, b)
    max_multiplier = get_max_multiplier(u, num_con)
    step_count = 0
    step_length = 0
    x_list = [x.copy()]
    step_list = [0]
    d_list = [d.copy()]
    single_d_list = []
    while True:
        step_count += 1
        # print(f"step_count is {step_count}")
        direction_norm = np.linalg.norm(d)
        print(f"direction norm: {direction_norm}")
        if direction_norm < TOLERANCE:
            if max_multiplier < 0:
                return x, step_count, np.array(x_list), np.array(step_list), np.array(d_list), np.array(single_d_list)
            else:  # update the active set
                active_set, a, b, active_set_bool = drop_active_set(active_set, a, b, max_multiplier, num_con,
                                                                    active_set_bool)
                if len(active_set) != len(b) - num_con:  # check the correspondence between the active set and rhs
                    print("length of active set does not agree with rhs")
                    return x, step_count, np.array(x_list), np.array(step_list), np.array(d_list), np.array(single_d_list)
                d, u = qp_econ(inv_q, sub_p, a, b)  # recompute the direction and multiplier
                single_d_list.append(step_count)
                max_multiplier = get_max_multiplier(u, num_con)
        else:  # compute the step length
            step_length = compute_step_length(x, d, active_set_bool)
            print(f"step length is {step_length}")
            x += step_length * d
            # if min(x) < 0:
            #     print("min value is negative")
            #     return x, step_count, np.array(x_list), np.array(step_list), np.array(d_list)
            obj = np.dot(np.dot(x.T, q), x) + np.dot(p, x)
            print(obj)
            active_set, a, b = get_econ(original_constraint, x)  # active set of current solution
            active_set_bool = get_active_set_bool(active_set, x)
            sub_p = p + np.dot(q, x)  # solve sub problem
            d, u = qp_econ(inv_q, sub_p, a, b)
            max_multiplier = get_max_multiplier(u, num_con)

        x_list.append(x.copy())
        step_list.append(step_length)
        d_list.append(d.copy())

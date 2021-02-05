"""
Active set method  solves linearly  constrained  general quadratic program
"""
import numpy as np
from numba import njit, int64
from matplotlib import pyplot as plt
from factorization.qr_factorization import qr_fact, qr_update
from factorization.cholesky_factorization import cholesky_solve
TOLERANCE = 1e-10


def generate_random_instance(n):
    q = np.cov(10 * np.random.random((n, 2)))
    p = np.random.random(n)
    x = np.random.random(n)
    # np.savetxt('../data/q.txt', q)
    # np.savetxt('../data/p.txt', p)
    # np.savetxt('../data/x.txt', x)
    return q, p, x


def read_qp_instance():
    q = np.loadtxt('./data/q.txt')
    p = np.loadtxt('./data/p.txt')
    x = np.loadtxt('./data/x.txt')
    return q, p, x


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


def qp_econ_null_space(a, x, b, G, c):
    """
    solves equality constrained quadratic program using the null space method
    :param a: Ax = b, equality constraint coefficient
    :param x: current solution
    :param b: Ax = b, rhs of equality constraint
    :param G: positive semi-definite matrix in the qp objective function
    :param c: linear objective
    :return: p the search direction and lmd the lagrangian multiplier
    """
    h = a.dot(x) - b
    g = c + G.dot(x)
    y, z, r = qr_fact(a)
    inv_ay = np.linalg.inv(a.dot(y))
    py = -np.dot(inv_ay, h)
    pz_rhs = -np.dot(np.dot(np.dot(z.T, G), y), py) - np.dot(z.T, g)
    pz_lhs = np.dot(np.dot(z.T, G), z)
    pz = cholesky_solve(pz_lhs, pz_rhs)
    p = y.dot(py) + z.dot(pz)
    lmd_left = inv_ay.dot(y.T)
    lmd_right = g + G.dot(p)
    lmd = lmd_left.dot(lmd_right)
    return p, lmd


# @njit()
def get_max_multiplier(u, num_con, active_set, active_set_bool):
    """
    :param u: multiplier
    :param num_con: number of constraints
    :param active_set: active set
    :param active_set_bool: boolean array representation of active set
    :return: the index of the largest multiplier, -1 indicates the active set algorithm to terminate
    """
    # print(f"u is {u}")
    max_multiplier = -np.inf
    max_index = -1
    for i in range(num_con, len(u)):
        val = u[i]
        if val > 0 and val > max_multiplier and active_set_bool[active_set[i - num_con]]:
            max_multiplier = val
            max_index = i
    return max_index - num_con


# @njit()
def drop_active_set(active_set, a, b, index, num_cons, active_set_bool):
    print(index)
    len_active_set = len(active_set)
    if len_active_set > 0:
        temp = active_set[index]
        active_set = drop_element(active_set, index)
        active_set_bool[temp] = 0
    a = drop_element(a, index + num_cons)
    b = drop_element(b, index + num_cons)
    return active_set, a, b, active_set_bool


# @njit()
def drop_element(arr, ind):
    """
    remove ind-th element from arr, depending on the shape of arr
    if arr has dim 1, the ind-th element is removed
    if arr has dim 2, the ind-th row is removed
    :param arr: input array of dim 1 or 2
    :param ind: the index to be removed
    :return: new arr with ind-th element removed from arr
    """
    n = len(arr)
    if arr.ndim == 1:
        new_arr = np.zeros(n - 1, dtype=int)
        count = 0
        for i in range(n):
            if i != ind:
                new_arr[count] = arr[i]
                count += 1
        return new_arr

    if arr.ndim == 2:
        m = arr.shape[1]
        new_arr = np.zeros((n - 1, m))
        count = 0
        for i in range(n):
            if i != ind:
                new_arr[count, :] = arr[i]
                count += 1
        return new_arr


# @njit()
def append_active_set(active_set, a, b, index, active_set_bool):
    if len(active_set) > 0:
        len_active_set = len(active_set)
        new_active_set = np.zeros(len_active_set + 1, dtype=int)
        new_active_set[:len_active_set] = active_set
        new_active_set[len_active_set] = index
    else:
        new_active_set = index * np.ones(1, dtype=int)
    active_set_bool[index] = 1
    temp = np.zeros(a.shape[1])
    temp[index] = 1
    a = np.vstack((a, temp))
    b = np.zeros(len(b) + 1)
    return new_active_set, a, b, active_set_bool


# @njit()
def compute_step_length(x, d, active_set_bool):
    step_length = np.inf
    j = -1
    for i in range(len(d)):
        if active_set_bool[i] == 0 and d[i] < 0 and x[i] > 0:
            ratio = - x[i] / d[i]
            if ratio < step_length:
                step_length = ratio
                j = i
    if step_length > 1.0:
        return 1.0, j
    return my_floor(step_length, 12), j


# @njit()
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
    eig_val, eig_mat = np.linalg.eig(q)
    if min(eig_val) < 0:
        print("problem is not positive semidefinite")
        return 0, 0
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
        # d, u = qp_econ_null_space(a, x, b, q, p)
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
            if append_index >= 0:  # append new element "append_index" to the current active set
                active_set, a, b, active_set_bool = append_active_set(active_set, a, b, append_index, active_set_bool)

            obj = 0.5 * np.dot(np.dot(x.T, q), x) + np.dot(p, x)
            print(obj)
            obj_list.append(obj)


def plot_obj(arr):
    plt.plot(arr)
    plt.show()

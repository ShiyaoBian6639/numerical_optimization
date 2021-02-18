import numpy as np
from numba import njit


# solves linear system exploiting sparsity
@njit()
def sparse_lower_triangular_solve(indices, ind_ptr, data, b):
    """
    solves lx = b, where l is a sparse lower triangular matrix
    :param indices: column indices of l
    :param ind_ptr: row indices of l
    :param data: value of l
    :param b: right hand side
    :return: x
    """
    x = np.zeros(len(b))
    count = 0
    row_count = 0
    for row_indices in range(len(ind_ptr) - 1):
        remain = b[row_count]
        for row_index in range(ind_ptr[row_indices], ind_ptr[row_indices + 1]):
            col_index = indices[count]
            if row_count == col_index:
                x[row_count] = remain / data[count]
            else:
                remain -= data[count] * x[col_index]
            count += 1
        row_count += 1
    return x


def dense_upper_triangular_solve(a, b):
    """
    solves ax = b
    :param a: upper triangular matrix
    :param b: right-hand-side
    :return:
    """
    pass

"""
step1: choose a starting point x0 and set k = 0
step2: determine a descent direction d_k
step3: determine step size by line search, choose step size tau_k > 0
step4: update x[k + 1] = x[k] + tau_k * d_k, k += 1
repeat until stopping criterion is satisfied

Linear approximation:
Suppose that f is differentiable and \partial f(x) \neq 0
Then we have linear approximation for all delta x \neq 0: f(x + \delta x) \approx f(x) + \partial f(x)^T \
"""
from first_order_algorithm.line_search import armijo_step
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from first_order_algorithm.fun_grad import f, df


def steepest_descent(x, eps, sigma, beta):
    d = -df(x)  # compute the gradient of f at initial point x
    descent_iter = 0
    step_size_iter = 0
    x_list = [x.copy()]
    y_list = [f(x)]
    while np.linalg.norm(d) > eps:
        descent_iter += 1
        step_size, armi_iter = armijo_step(x, d, sigma, beta)
        step_size_iter += armi_iter
        x += step_size * d
        x_list.append(x.copy())
        y_list.append(f(x))
        d = -df(x)
    return x, descent_iter, step_size_iter, x_list, y_list


def contour_plot(delta, x_range, y_range, x_arr, levels, title):
    x1 = np.arange(x_range[0], x_range[1], delta)
    x2 = np.arange(y_range[0], y_range[1], delta)
    X1, X2 = np.meshgrid(x1, x2)
    Y = evaluate_mesh(X1, X2)
    fig, ax = plt.subplots()
    CS = ax.contour(X1, X2, Y, levels=levels)
    ax.scatter(x_arr[:, 0], x_arr[:, 1], s=1)
    ax.plot(x_arr[:, 0], x_arr[:, 1], color='red')
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_title(title)
    fig.show()


@njit()
def evaluate_mesh(X1, X2):
    n, m = X1.shape
    Y = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            Y[i, j] = f([X1[i, j], X2[i, j]])
    return Y

import numpy as np
from utils import f, df
from line_search import armijo_step

x = np.array([190.0, 270.0])
x = np.array([0.0, 0.0])
eps, sigma, beta = 1e-4, 0.5, 0.5


def steepest_descent(x, eps, sigma, beta):
    d = -df(x)  # compute the gradient of f at initial point x
    descent_iter = 0
    step_size_iter = 0
    while np.linalg.norm(d) > eps:
        descent_iter += 1
        step_size, armi_iter = armijo_step(x, d, sigma, beta)
        print(step_size)
        step_size_iter += armi_iter
        x += step_size * d
        d = -df(x)
    return x, descent_iter, step_size_iter


res, descent, step = steepest_descent(x, 0.0001, 0.5, 0.5)

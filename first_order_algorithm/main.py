import numpy as np
from first_order_algorithm.utils import contour_plot, steepest_descent

x = np.array([10.0, 10.0])
# x = np.array([0.0, 0.0])
eps, sigma, beta = 1e-4, 0.5, 0.5

res, descent_step, line_search_step, x_list, y_list = steepest_descent(x, 0.0001, 0.5, 0.5)
x_arr = np.array(x_list)
x_range = [8, 15]
y_range = [2, 12]
levels = np.arange(0, 20, 2) ** 2
title = 'Start from [10, 10], armijo takes 35 steps to converge within 1e-4'
contour_plot(0.5, x_range, y_range, x_arr, levels, title)

y_arr = np.array(y_list)
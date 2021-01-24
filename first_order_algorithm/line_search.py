from utils import f, df
import numpy as np


def armijo_step(x, d, sigma, beta):
    """
    determine step size
    :return:
    """
    count = 0
    tau = 1
    diff = f(x + tau * d) - f(x)
    while diff > sigma * tau * np.dot(df(x), d):
        count += 1
        tau *= beta
        diff = f(x + tau * d) - f(x)
    return tau, count


#!/usr/bin/env python3.7

# Copyright 2020, Gurobi Optimization, LLC

# Portfolio selection: given a sum of money to invest, one must decide how to
# spend it amongst a portfolio of financial securities.  Our approach is due
# to Markowitz (1959) and looks to minimize the risk associated with the
# investment while realizing a target expected return.  By varying the target,
# one can compute an 'efficient frontier', which defines the optimal portfolio
# for a given expected return.
#
# Note that this example reads historical return data from a comma-separated
# file (../data/portfolio.csv).  As a result, it must be run from the Gurobi
# examples/python directory.
#
# This example requires the pandas (>= 0.20.3), NumPy, and Matplotlib
# Python packages, which are part of the SciPy ecosystem for
# mathematics, science, and engineering (http://scipy.org).  These
# packages aren't included in all Python distributions, but are
# included by default with Anaconda Python.

from active_set_method.utils import generate_random_instance
from gurobipy import GRB, Model
import numpy as np

n = 5000
q, p, x = generate_random_instance(n)

m = Model('portfolio')
A = np.ones(n)
x = m.addMVar(n, lb=0.0)
m.setObjective(x @ q @ x + 2 * p @ x, GRB.MINIMIZE)
m.addConstr(A @ x == 1)
m.optimize()
count = 0
temp = np.zeros(n)
for i in range(n):
    if x[i].x > 1e-4:
        temp[i] = x[i].x
x = temp.copy()

shit = np.vstack((x, y))

gurobi_obj = .5 * x @ q @ x + p @ x
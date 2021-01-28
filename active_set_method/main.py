from active_set_method.utils import active_set_method
import numpy as np
from gurobipy import Model, GRB

# q = np.eye(3) * 2  # quadratic coefficient
# p = np.array([-2.0, 2.0, 0.0])  # linear coefficient
# x = np.array([100.0, 100.0, -199.0])  # initial feasible solution
# a = np.array([[1.0, 1.0, 1.0]])
# active_set_method(q, p, x, a)

# large instance test

n = 10
temp = 10 * np.random.random((n, n))
q = temp + temp.T
p = np.random.random(n)
x = np.random.random(n)
x = x / x.sum()
a = np.ones((1, n))
x, cnt = active_set_method(q, p, x, a)
print(min(x))

m = Model('portfolio')
vars = m.addVars(n)
temp = 10 * np.random.random((n, n))
sigma = temp + temp.T
portfolio_risk = sigma.dot(vars).dot(vars)
m.setObjective(portfolio_risk, GRB.MINIMIZE)
m.addConstr(vars.sum() == 1, 'budget')
m.optimize()
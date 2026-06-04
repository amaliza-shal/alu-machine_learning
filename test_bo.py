import numpy as np
BayesianOptimization = __import__('5-bayes_opt').BayesianOptimization

def f(x):
    return np.sin(5*x) + 2*np.sin(-2*x)

X_init = np.array([2.03085276, 3.59890832, 4.55210364, 5.89850049, -3.14159265]).reshape(-1, 1)
Y_init = f(X_init)

bo = BayesianOptimization(f, X_init, Y_init, (-np.pi, np.pi), 50, l=0.6, sigma_f=2, xsi=0.05)
X_opt, Y_opt = bo.optimize(iterations=15)

print(X_opt)
print(Y_opt)
print(bo.gp.X)

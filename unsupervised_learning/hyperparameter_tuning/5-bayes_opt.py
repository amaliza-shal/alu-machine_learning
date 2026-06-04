#!/usr/bin/env python3
"""
Bayes Optimization
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
    Bayes Optimization using Gaussian Process
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01,
                 minimize=True):
        """
        * f is the black-box function to be optimized
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        xmin, xmax = bounds
        self.X_s = np.linspace(xmin, xmax, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        * Uses the Expected Improvement acquisition function
        Returns: X_next, EI
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            mu_sample_opt = np.min(self.gp.Y)
            imp = (mu_sample_opt - mu - self.xsi)
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = (mu - mu_sample_opt - self.xsi)

        with np.errstate(divide='warn'):
            Z = np.where(sigma > 0, imp / sigma, 0)
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimize method
        """
        for _ in range(iterations):
            x_next, _ = self.acquisition()
            # If the next proposed point has already been sampled, stop early
            if np.any(np.isclose(self.gp.X, x_next)):
                break
            y_next = self.f(x_next)
            self.gp.update(x_next, y_next)

        if self.minimize is True:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)

        x_opt = self.gp.X[idx]
        y_opt = self.gp.Y[idx]

        return x_opt, y_opt

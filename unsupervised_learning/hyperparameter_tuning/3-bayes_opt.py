#!/usr/bin/env python3
"""
Initialize Bayesian Optimization
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor
        Args:
            f: black-box function to be optimized
            X_init: numpy.ndarray of shape (t, 1) representing the inputs
                    already sampled
            Y_init: numpy.ndarray of shape (t, 1) representing the outputs
                    for each input in X_init
            bounds: tuple of (min, max) representing the bounds of the space
            ac_samples: number of samples that should be analyzed during acquisition
            l: length parameter for the kernel
            sigma_f: standard deviation for the output
            xsi: exploration-exploitation factor for acquisition
            minimize: bool determining whether to minimize (True) or maximize (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

#!/usr/bin/env python3
"""
Normalization Constants
"""
import numpy as np


def normalization_constants(X):
    """
    calculates the normalization (standardization) constants of a matrix:
    X is the numpy.ndarray of shape (m, nx) to normalize
    Returns: the mean and standard deviation of each feature, respectively
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std

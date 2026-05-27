#!/usr/bin/env python3
"""
Shuffle Data
"""
import numpy as np


def shuffle_data(X, Y):
    """
    shuffles the data points in two matrices the same way:
    X is the first numpy.ndarray of shape (m, nx) to shuffle
    Y is the second numpy.ndarray of shape (m, ny) to shuffle
    Returns: the shuffled X and Y matrices
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]

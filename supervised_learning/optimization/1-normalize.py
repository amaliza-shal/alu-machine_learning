#!/usr/bin/env python3
"""
Normalize
"""
import numpy as np


def normalize(X, m, s):
    """
    normalizes (standardizes) a matrix:
    X is the numpy.ndarray of shape (d, nx) to normalize
    m is the mean of all features of X
    s is the standard deviation of all features of X
    Returns: The normalized X matrix
    """
    return (X - m) / s

#!/usr/bin/env python3
"""
Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    calculates the batch normalization of a tensor in numpy:
    Z is a numpy.ndarray of shape (m, n) to be normalized
    gamma is a numpy.ndarray of shape (1, n) containing the scaling factors
        for batch normalization
    beta is a numpy.ndarray of shape (1, n) containing the offsets for
        batch normalization
    epsilon is a small number used to avoid division by zero
    Returns: the normalized Z
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    out = gamma * Z_norm + beta
    return out

#!/usr/bin/env python3
"""One-hot encoding function"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Args:
        Y: numpy.ndarray with shape (m,) containing numeric class labels
        classes: The maximum number of classes found in Y

    Returns:
        A one-hot encoding of Y with shape (classes, m), or None on failure
    """
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None

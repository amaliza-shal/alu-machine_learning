#!/usr/bin/env python3
"""One-hot decoding function"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels

    Args:
        one_hot: A one-hot encoded numpy.ndarray with shape (classes, m)

    Returns:
        A numpy.ndarray with shape (m,) containing numeric labels for
        each example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None

#!/usr/bin/env python3
"""
Calculates the F1 score of a confusion matrix
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix
    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
    Returns:
        numpy.ndarray of shape (classes,) containing the F1 score
    """
    s = sensitivity(confusion)
    p = precision(confusion)
    return 2 * (p * s) / (p + s)

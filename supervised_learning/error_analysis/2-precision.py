#!/usr/bin/env python3
"""
Calculates the precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
    Returns:
        numpy.ndarray of shape (classes,) containing the precision
    """
    # Precision = TP / (TP + FP)
    # TP + FP is the sum of the column (predicted positives)
    tp = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)
    return tp / predicted_positives

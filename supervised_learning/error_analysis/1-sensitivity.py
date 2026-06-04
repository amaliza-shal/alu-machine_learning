#!/usr/bin/env python3
"""
Calculates the sensitivity for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
    Returns:
        numpy.ndarray of shape (classes,) containing the sensitivity
    """
    # Sensitivity (Recall) = TP / (TP + FN)
    # TP + FN is the sum of the row (actual positives)
    tp = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)
    return tp / actual_positives

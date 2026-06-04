#!/usr/bin/env python3
"""
Calculates the specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
    Returns:
        numpy.ndarray of shape (classes,) containing the specificity
    """
    # Specificity = TN / (TN + FP)
    # TN = Total - (ActualPos + PredictedPos - TP)
    # Actual Pos = sum(axis=1), Predicted Pos = sum(axis=0)
    # FP = PredictedPos - TP
    # TN + FP = Total - ActualPos
    tp = np.diag(confusion)
    actual_pos = np.sum(confusion, axis=1)
    predicted_pos = np.sum(confusion, axis=0)
    total = np.sum(confusion)

    # TN = total - (actual_pos + predicted_pos - tp)
    # TN + FP = total - actual_pos (All actual negatives)
    # Specificity = TN / (total - actual_pos)
    tn = total - (actual_pos + predicted_pos - tp)
    actual_neg = total - actual_pos
    return tn / actual_neg

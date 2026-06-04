#!/usr/bin/env python3
"""
Creates a confusion matrix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix
    Args:
        labels: one-hot numpy.ndarray of shape (m, classes)
        logits: one-hot numpy.ndarray of shape (m, classes)
    Returns:
        confusion matrix of shape (classes, classes)
    """
    return np.dot(labels.T, logits)

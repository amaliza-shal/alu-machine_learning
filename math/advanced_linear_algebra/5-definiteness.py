#!/usr/bin/env python3
"""
Module for calculating the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if valid matrix (square)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check for symmetry
    # Check if A == A.T
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Calculate eigenvalues
        w, _ = np.linalg.eig(matrix)

        if np.all(w > 0):
            return "Positive definite"
        if np.all(w >= 0):
            return "Positive semi-definite"
        if np.all(w < 0):
            return "Negative definite"
        if np.all(w <= 0):
            return "Negative semi-definite"

        return "Indefinite"

    except Exception:

        return None

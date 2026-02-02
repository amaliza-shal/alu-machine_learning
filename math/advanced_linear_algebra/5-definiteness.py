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
        
        # Check if there are both positive and negative eigenvalues
        # Because floating point, check > epsilon?
        # Assuming exact enough or handled by numpy
        # Given "Indefinite" is an option
        # Indefinite means some positive and some negative.
        # It's implied by falling through the above checks if it didn't fit semi-def cases.
        # But wait. Indefinite matrix: xT A x takes both positive and negative values.
        # Equivalent to having both positive and negative eigenvalues.
        return "Indefinite"
        
    except Exception:
        return None

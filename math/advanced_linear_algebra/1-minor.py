#!/usr/bin/env python3
"""
Module for calculating the minor matrix
"""


def get_determinant(matrix):
    """
    Helper function to calculate determinant of a matrix
    """
    # Assuming valid square matrix here as this is a helper
    if matrix == [[]]:
        return 1
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(n):
        sub_matrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * get_determinant(sub_matrix)
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    minor_matrix = []
    for r in range(n):
        row_res = []
        for c in range(n):
            sub_matrix = [row[:c] + row[c+1:]
                          for i, row in enumerate(matrix) if i != r]
            det = get_determinant(sub_matrix)
            row_res.append(det)
        minor_matrix.append(row_res)

    return minor_matrix

#!/usr/bin/env python3
"""
Module for calculating the determinant of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        return 1

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(n):
        sub_matrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub_matrix)

    return det

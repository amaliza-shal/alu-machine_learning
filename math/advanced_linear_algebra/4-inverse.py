#!/usr/bin/env python3
"""
Module for calculating the inverse of a matrix
"""


def get_determinant(matrix):
    """
    Helper function to calculate determinant of a matrix
    """
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


def adjugate(matrix):
    """
    Helper function to calculate adjugate matrix
    """
    n = len(matrix)
    if n == 1:
        return [[1]]

    cofactor_matrix = []
    for r in range(n):
        row_res = []
        for c in range(n):
            sub_matrix = [row[:c] + row[c+1:]
                          for i, row in enumerate(matrix) if i != r]
            det = get_determinant(sub_matrix)
            sign = (-1) ** (r + c)
            row_res.append(sign * det)
        cofactor_matrix.append(row_res)

    adjugate_matrix = []
    for c in range(n):
        new_row = []
        for r in range(n):
            new_row.append(cofactor_matrix[r][c])
        adjugate_matrix.append(new_row)
    return adjugate_matrix


def inverse(matrix):
    """
    Calculates the inverse of a matrix
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

    det = get_determinant(matrix)

    if det == 0:
        return None

    adj = adjugate(matrix)
    inv_matrix = []
    for r in range(n):
        new_row = []
        for c in range(n):
            new_row.append(adj[r][c] / det)
        inv_matrix.append(new_row)

    return inv_matrix

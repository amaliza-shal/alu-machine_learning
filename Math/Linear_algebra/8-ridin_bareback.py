#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    """Performs matrix multiplication on two 2D matrices"""
    # Check if matrices can be multiplied
    # Number of columns in mat1 must equal number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None
    
    # Get dimensions
    rows_mat1 = len(mat1)
    cols_mat1 = len(mat1[0])
    cols_mat2 = len(mat2[0])
    
    # Initialize result matrix with zeros
    result = []
    
    # Perform matrix multiplication
    for i in range(rows_mat1):
        row = []
        for j in range(cols_mat2):
            # Calculate dot product of row i from mat1 and column j from mat2
            element = 0
            for k in range(cols_mat1):
                element += mat1[i][k] * mat2[k][j]
            row.append(element)
        result.append(row)
    
    return result
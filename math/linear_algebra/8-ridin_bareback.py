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
    

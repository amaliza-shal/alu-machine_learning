#!/usr/bin/env python3
"""
Forward Propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout:
    X is a numpy.ndarray of shape (nx, m) containing the input data
    weights is a dictionary of the weights and biases of the neural network
    L is the number of layers in the network
    keep_prob is the probability that a node will be kept
    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function
    Returns: a dictionary containing the outputs of each layer and the
        dropout mask used on each layer
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i == L:
            # Softmax
            exp_Z = np.exp(Z)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            # Tanh
            A = np.tanh(Z)
            # Apply Dropout
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            A = (A * D) / keep_prob
            cache['D' + str(i)] = D

        cache['A' + str(i)] = A

    return cache

#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using gradient
        descent with L2 regularization:
    Y is a one-hot numpy.ndarray of shape (classes, m) that contains
        the correct labels for the data
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs of each layer of the neural network
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network
    The neural network uses tanh activations on each layer except the last,
        which uses a softmax activation
    The weights and biases of the network should be updated in place
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dw = (1 / m) * np.matmul(dz, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if i > 1:
            # tanh derivative: 1 - A^2
            dz = np.matmul(W.T, dz) * (1 - A_prev ** 2)

        weights['W' + str(i)] = W - alpha * dw
        weights['b' + str(i)] = b - alpha * db

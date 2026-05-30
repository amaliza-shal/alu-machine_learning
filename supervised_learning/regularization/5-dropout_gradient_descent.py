#!/usr/bin/env python3
"""
Gradient Descent with Dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    updates the weights of a neural network with Dropout regularization
        using gradient descent:
    Y is a one-hot numpy.ndarray of shape (classes, m) containing the labels
    weights is a dictionary of the weights and biases of the neural network
    cache is a dictionary of the outputs and dropout masks of each layer
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network
    All layers use tanh activation except last, which uses softmax
    The weights of the network should be updated in place
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dw = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if i > 1:
            # Backprop Through Dropout and Tanh
            # dA = W.T * dz
            # dA = (dA * D) / keep_prob
            # dz = dA * (1 - A_prev^2)
            D = cache['D' + str(i - 1)]
            dz = np.matmul(W.T, dz)
            dz = (dz * D) / keep_prob
            dz = dz * (1 - A_prev ** 2)

        weights['W' + str(i)] = W - alpha * dw
        weights['b' + str(i)] = b - alpha * db

#!/usr/bin/env python3
"""DeepNeuralNetwork class for binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize the deep neural network

        Args:
            nx: Number of input features
            layers: List representing the number of nodes in each layer

        Raises:
            TypeError: If nx is not an integer or layers is not a list
            ValueError: If nx is less than 1 or layers is empty or
                contains non-positive integers
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # Validate all nodes in layers without loops
        valid = all(isinstance(node, int) and node >= 1 for node in layers)
        if not valid:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Initialize weights using He initialization
        prev_nodes = nx
        for i in range(1, self.L + 1):
            nodes = layers[i - 1]
            He_factor = np.sqrt(2.0 / prev_nodes)
            self.weights[f'W{i}'] = np.random.normal(
                0, He_factor, (nodes, prev_nodes))
            self.weights[f'b{i}'] = np.zeros((nodes, 1))
            prev_nodes = nodes

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the deep neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data

        Returns:
            The output of the neural network and the cache
        """
        self.cache['A0'] = X
        return self._forward_layer(X, 1)

    def _forward_layer(self, A_prev, layer):
        """
        Recursively computes forward propagation through layers

        Args:
            A_prev: Output from previous layer
            layer: Current layer number

        Returns:
            The output of the neural network
        """
        Z = np.matmul(self.weights[f'W{layer}'], A_prev) + self.weights[f'b{layer}']
        A = 1 / (1 + np.exp(-Z))
        self.cache[f'A{layer}'] = A

        if layer == self.L:
            return A
        else:
            return self._forward_layer(A, layer + 1)

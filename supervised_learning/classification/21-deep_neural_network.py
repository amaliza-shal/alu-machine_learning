#!/usr/bin/env python3
"""DeepNeuralNetwork class with gradient descent"""
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
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for node in layers:
            if not isinstance(node, int) or node < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights using He initialization
        prev_nodes = nx
        for i in range(1, self.__L + 1):
            nodes = layers[i - 1]
            He_factor = np.sqrt(2.0 / prev_nodes)
            self.__weights['W{}'.format(i)] = np.random.normal(
                0, He_factor, (nodes, prev_nodes))
            self.__weights['b{}'.format(i)] = np.zeros((nodes, 1))
            prev_nodes = nodes

    @property
    def L(self):
        """Get the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Get the cache"""
        return self.__cache

    @property
    def weights(self):
        """Get the weights"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data

        Returns:
            The output of the neural network and the cache
        """
        self.__cache['A0'] = X
        A = X

        for i in range(1, self.__L + 1):
            Z = np.matmul(self.__weights['W{}'.format(i)], A) + self.__weights['b{}'.format(i)]
            A = 1 / (1 + np.exp(-Z))
            self.__cache['A{}'.format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) containing correct labels
            A: numpy.ndarray with shape (1, m) containing activated output

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
            Y: numpy.ndarray with shape (1, m) containing correct labels

        Returns:
            The neural network's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y: numpy.ndarray with shape (1, m) containing correct labels
            cache: Dictionary containing all intermediary values of the network
            alpha: The learning rate
        """
        m = Y.shape[1]
        dZ = cache['A{}'.format(self.__L)] - Y
        self._backward_layer(Y, cache, dZ, m, alpha, self.__L)

    def _backward_layer(self, Y, cache, dZ, m, alpha, layer):
        """
        Recursively computes backpropagation through layers

        Args:
            Y: numpy.ndarray with shape (1, m) containing correct labels
            cache: Dictionary containing all intermediary values
            dZ: Gradient of Z
            m: Number of examples
            alpha: The learning rate
            layer: Current layer number
        """
        dW = np.matmul(dZ, cache['A{}'.format(layer-1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        self.__weights['W{}'.format(layer)] = self.__weights['W{}'.format(layer)] - alpha * dW
        self.__weights['b{}'.format(layer)] = self.__weights['b{}'.format(layer)] - alpha * db

        if layer > 1:
            dA = np.matmul(self.__weights['W{}'.format(layer)].T, dZ)
            dZ = dA * cache['A{}'.format(layer-1)] * (1 - cache['A{}'.format(layer-1)])
            self._backward_layer(Y, cache, dZ, m, alpha, layer - 1)

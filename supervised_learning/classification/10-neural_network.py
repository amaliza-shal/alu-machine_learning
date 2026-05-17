#!/usr/bin/env python3
"""NeuralNetwork class with forward propagation"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        """
        Initialize the neural network

        Args:
            nx: Number of input features
            nodes: Number of nodes in the hidden layer

        Raises:
            TypeError: If nx or nodes is not an integer
            ValueError: If nx or nodes is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Get the weights of the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """Get the bias of the hidden layer"""
        return self.__b1

    @property
    def A1(self):
        """Get the activated output of the hidden layer"""
        return self.__A1

    @property
    def W2(self):
        """Get the weights of the output layer"""
        return self.__W2

    @property
    def b2(self):
        """Get the bias of the output layer"""
        return self.__b2

    @property
    def A2(self):
        """Get the activated output of the output layer"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data

        Returns:
            The private attributes __A1 and __A2
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

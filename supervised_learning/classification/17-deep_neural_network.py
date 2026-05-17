#!/usr/bin/env python3
"""DeepNeuralNetwork class with private attributes"""
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
            ValueError: If nx is less than 1 or layers is empty or contains non-positive integers
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
            self.__weights[f'W{i}'] = np.random.normal(0, np.sqrt(2.0 / prev_nodes), (nodes, prev_nodes))
            self.__weights[f'b{i}'] = np.zeros((nodes, 1))
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

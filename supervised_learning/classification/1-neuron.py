#!/usr/bin/env python3
"""Neuron class for binary classification with private attributes"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize the neuron
        
        Args:
            nx: Number of input features to the neuron
            
        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Get the weights"""
        return self.__W

    @property
    def b(self):
        """Get the bias"""
        return self.__b

    @property
    def A(self):
        """Get the activated output"""
        return self.__A

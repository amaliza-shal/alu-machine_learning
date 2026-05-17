#!/usr/bin/env python3
"""Neuron class for binary classification"""
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
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0

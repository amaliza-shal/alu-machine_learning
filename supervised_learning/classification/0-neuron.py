#!/usr/bin/env python3
"""Module that defines a single neuron performing binary classification"""
import numpy as np


class Neuron:
    """Class that defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Class constructor

        Args:
            nx (int): number of input features to the neuron
        """
        if not isinstance(nx, int) or isinstance(nx, bool):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
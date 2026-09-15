#!/usr/bin/env python3
"""
Task 2: Forward Propagation
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x: the placeholder for the input data
    layer_sizes: a list containing the number of nodes in each layer
    activations: a list containing the activation functions for each layer
    Returns: the prediction of the network in tensor form
    """
    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev

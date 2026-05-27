#!/usr/bin/env python3
"""
Task 1: Layers
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev: the tensor output of the previous layer
    n: the number of nodes in the layer to create
    activation: the activation function that the layer should use
    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    to implement He et. al initialization for the layer weights
    each layer should be given the name layer
    Returns: the tensor output of the layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
    )
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name='layer'
    )
    return layer(prev)

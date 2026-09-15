#!/usr/bin/env python3
"""
Batch Norm Upgraded
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer for a neural network in tensorflow:
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used on the output
        of the layer
    Returns: a tensor of the activated output for the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = layer(prev)

    gamma = tf.Variable(tf.ones([n]), name="gamma")
    beta = tf.Variable(tf.zeros([n]), name="beta")
    epsilon = 1e-8

    m, s = tf.nn.moments(Z, axes=[0])
    normalized = tf.nn.batch_normalization(Z, m, s, beta, gamma, epsilon)

    return activation(normalized)

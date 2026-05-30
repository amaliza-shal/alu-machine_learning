#!/usr/bin/env python3
"""
L2 Regularization Cost using TensorFlow
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    calculates the cost of a neural network with L2 regularization:
    cost is a tensor containing the cost of the network without
        L2 regularization
    Returns: a tensor containing the cost of the network accounting
        for L2 regularization
    """
    # tf.losses.get_regularization_losses() returns a list of tensors
    # for each layer that has a kernel_regularizer
    l2_losses = tf.losses.get_regularization_losses()
    return cost + tf.add_n(l2_losses)

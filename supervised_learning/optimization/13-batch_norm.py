#!/usr/bin/env python3
"""
Batch Normalization
"""
import tensorflow as tf


def batch_norm(Z, gamma, beta, epsilon):
    """
    creates a batch normalization layer for a neural network in tensorflow:
    Z is the variable to be normalized
    gamma is a tensor containing the scaling factors for batch normalization
    beta is a tensor containing the offsets for batch normalization
    epsilon is a small number used to avoid division by zero
    Returns: a tensor of the normalized Z
    """
    m, s = tf.nn.moments(Z, axes=[0])
    return tf.nn.batch_normalization(Z, m, s, beta, gamma, epsilon)

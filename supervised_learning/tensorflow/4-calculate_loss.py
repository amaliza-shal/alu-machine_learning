#!/usr/bin/env python3
"""
Task 4: Loss
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network’s predictions
    Returns: a tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss

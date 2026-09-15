#!/usr/bin/env python3
"""
Task 5: Train_Op
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    loss: the loss of the network’s prediction
    alpha: the learning rate
    Returns: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op

#!/usr/bin/env python3
"""
Task 3: Accuracy
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    y: placeholder for the labels of the input data
    y_pred: tensor containing the network’s predictions
    Returns: a tensor containing the decimal accuracy of the prediction
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

#!/usr/bin/env python3
"""
Moving Average
"""
import numpy as np


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set:
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Returns: a list containing the moving averages of data
    """
    v = 0
    ma = []
    for x in data:
        v = v * beta + (1 - beta) * x
        ma.append(v)
    return ma

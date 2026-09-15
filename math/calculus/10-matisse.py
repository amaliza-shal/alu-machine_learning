#!/usr/bin/env python3
"""Module for calculating derivative of polynomial"""


def poly_derivative(poly):
    """Calculates derivative of polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    derivative = []
    for i in range(1, len(poly)):
        if type(poly[i]) not in [int, float]:
            return None
        derivative.append(poly[i] * i)

    if len(derivative) == 0:
        return [0]
    return derivative

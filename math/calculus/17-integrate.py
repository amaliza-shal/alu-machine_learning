#!/usr/bin/env python3
"""Module for calculating integral of polynomial"""


def poly_integral(poly, C=0):
    """Calculates integral of polynomial"""
    if type(poly) is not list or len(poly) == 0:
        return None
    if type(C) not in [int, float]:
        return None
        
    integral = [C]
    for i in range(len(poly)):
        if type(poly[i]) not in [int, float]:
            return None
        
        res = poly[i] / (i + 1)
        if res.is_integer():
            res = int(res)
        integral.append(res)
             
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
        
    return integral

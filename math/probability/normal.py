#!/usr/bin/env python3
"""
Normal distribution class
"""


class Normal:
    """
    Represents a Normal distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initializes the Normal distribution
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            variance = sum([(x - self.mean) ** 2 for x in data]) / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        pi = 3.1415926536
        e = 2.7182818285
        coeff = 1 / (self.stddev * ((2 * pi) ** 0.5))
        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2
        return coeff * (e ** exponent)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        pi = 3.1415926536
        k = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = (k - (k ** 3) / 3 + (k ** 5) / 10 -
               (k ** 7) / 42 + (k ** 9) / 216)
        erf *= 2 / (pi ** 0.5)
        return 0.5 * (1 + erf)

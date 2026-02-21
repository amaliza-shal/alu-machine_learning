#!/usr/bin/env python3
"""
Binomial distribution class
"""


class Binomial:
    """
    Represents a Binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum([(x - mean) ** 2 for x in data]) / len(data)

            p_est = 1 - (variance / mean)
            self.n = int(round(mean / p_est))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0

        n_fact = 1
        for i in range(1, self.n + 1):
            n_fact *= i

        k_fact = 1
        for i in range(1, k + 1):
            k_fact *= i

        nk_fact = 1
        for i in range(1, self.n - k + 1):
            nk_fact *= i

        comb = n_fact / (k_fact * nk_fact)

        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)

        return cdf_value

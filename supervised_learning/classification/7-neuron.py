#!/usr/bin/env python3
"""Neuron class with advanced training"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize the neuron

        Args:
            nx: Number of input features to the neuron

        Raises:
            TypeError: If nx is not an integer
            ValueError: If nx is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Get the weights"""
        return self.__W

    @property
    def b(self):
        """Get the bias"""
        return self.__b

    @property
    def A(self):
        """Get the activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data

        Returns:
            The private attribute __A
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (1, m) containing correct labels
            A: numpy.ndarray with shape (1, m) containing activated output

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
            Y: numpy.ndarray with shape (1, m) containing correct labels

        Returns:
            The neuron's prediction and the cost of the network
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
            Y: numpy.ndarray with shape (1, m) containing correct labels
            A: numpy.ndarray with shape (1, m) containing activated output
            alpha: The learning rate
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = np.matmul(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
            Y: numpy.ndarray with shape (1, m) containing correct labels
            iterations: The number of iterations to train over
            alpha: The learning rate
            verbose: Boolean to print information about training
            graph: Boolean to graph information about training
            step: Number of iterations between prints/graphs

        Returns:
            The evaluation of the training data after iterations
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iterations_list = []

        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A)

            if i % step == 0:
                costs.append(cost)
                iterations_list.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

            if i < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)

        if graph:
            plt.figure()
            plt.plot(iterations_list, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

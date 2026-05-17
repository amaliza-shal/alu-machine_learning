#!/usr/bin/env python3
"""DeepNeuralNetwork class with different activation functions"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing multiclass classification"""

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize the deep neural network

        Args:
            nx: Number of input features
            layers: List representing the number of nodes in each layer
            activation: Type of activation function ('sig' or 'tanh')

        Raises:
            TypeError: If nx is not an integer or layers is not a list
            ValueError: If nx is less than 1 or layers is empty or
                contains non-positive integers
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for node in layers:
            if not isinstance(node, int) or node < 1:
                raise TypeError("layers must be a list of positive integers")

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

        # Initialize weights using He initialization
        prev_nodes = nx
        for i in range(1, self.__L + 1):
            nodes = layers[i - 1]
            He_factor = np.sqrt(2.0 / prev_nodes)
            self.__weights['W{}'.format(i)] = np.random.normal(
                0, He_factor, (nodes, prev_nodes))
            self.__weights['b{}'.format(i)] = np.zeros((nodes, 1))
            prev_nodes = nodes

    @property
    def L(self):
        """Get the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Get the cache"""
        return self.__cache

    @property
    def weights(self):
        """Get the weights"""
        return self.__weights

    @property
    def activation(self):
        """Get the activation function"""
        return self.__activation

    def _sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def _tanh(self, Z):
        """Tanh activation function"""
        return np.tanh(Z)

    def _sigmoid_derivative(self, A):
        """Derivative of sigmoid"""
        return A * (1 - A)

    def _tanh_derivative(self, A):
        """Derivative of tanh"""
        return 1 - A ** 2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data

        Returns:
            The output of the neural network and the cache
        """
        self.__cache['A0'] = X
        A = X

        for i in range(1, self.__L + 1):
            Z = np.matmul(self.__weights[f'W{i}'], A) + self.__weights[f'b{i}']

            # Use specified activation for hidden layers, sigmoid for output
            if i == self.__L:
                A = self._sigmoid(Z)
            else:
                if self.__activation == 'sig':
                    A = self._sigmoid(Z)
                else:  # tanh
                    A = self._tanh(Z)

            self.__cache[f'A{i}'] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression

        Args:
            Y: numpy.ndarray with shape (classes, m) containing labels
            A: numpy.ndarray with shape (classes, m) containing output

        Returns:
            The cost
        """
        m = Y.shape[1]
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
            Y: numpy.ndarray with shape (classes, m) containing labels

        Returns:
            The neural network's prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        # Return one-hot encoded predictions
        prediction = np.zeros_like(A)
        prediction[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network

        Args:
            Y: numpy.ndarray with shape (classes, m) containing labels
            cache: Dictionary containing all intermediary values
            alpha: The learning rate
        """
        m = Y.shape[1]

        # Backpropagation
        dZ = cache[f'A{self.__L}'] - Y

        for i in range(self.__L, 0, -1):
            dW = np.matmul(dZ, cache[f'A{i-1}'].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                dA = np.matmul(self.__weights[f'W{i}'].T, dZ)
                # Use specified activation derivative for hidden layers
                if self.__activation == 'sig':
                    dZ = dA * self._sigmoid_derivative(cache[f'A{i-1}'])
                else:  # tanh
                    dZ = dA * self._tanh_derivative(cache[f'A{i-1}'])

            self.__weights[f'W{i}'] = self.__weights[f'W{i}'] - alpha * dW
            self.__weights[f'b{i}'] = self.__weights[f'b{i}'] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network

        Args:
            X: numpy.ndarray with shape (nx, m) containing the input data
            Y: numpy.ndarray with shape (classes, m) one-hot labels
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
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i % step == 0:
                costs.append(cost)
                iterations_list.append(i)
                if verbose:
                    print("Cost after {} iterations: {}".format(i, cost))

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.figure()
            plt.plot(iterations_list, costs, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format

        Args:
            filename: The file to which the object should be saved
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object

        Args:
            filename: The file from which the object should be loaded

        Returns:
            The loaded object, or None if filename doesn't exist
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

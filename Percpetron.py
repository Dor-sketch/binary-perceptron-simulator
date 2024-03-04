"""
This program implements multiple perceptrons to classify 21 digits
binary numbers into 3 classes:
1. more that 15 ones
2. less than 15 ones
3. else
"""

import random
import time
import matplotlib.pyplot as plt
import numpy as np

class InputNeuron:
    def __init__(self, n_inputs=21):
        random.seed(time.time())
        self.units = [random.choice([0, 1]) for _ in range(n_inputs)]

    def __len__(self):
        return len(self.units)


class Percpetron:
    def __init__(self, InputNeuron, threshold=0.999):
        """
        Initialize the perceptron

        :param InputNeuron: the input neuron
        :param threshold: the threshold for the perceptron.
        """
        self.input_neuron = InputNeuron
        self.weights = np.random.rand(len(self.input_neuron))
        self.threshold = threshold  # see readme for more info
        self.last_input = np.zeros(len(self.input_neuron))

    def dot_product(self, inputs=None):
        """
        Calculate the dot product of the inputs and the weights

        :param inputs: list of 21 digits binary numbers, each list is a dot (input) represented a binary number
        :return: the dot product of the inputs and the weights
        """
        if inputs is None:
            inputs = self.last_input
        return np.dot(self.weights, inputs)

    def predict(self, inputs):
        """
        Predict the label of the given dot

        :param inputs: list of 21 digits binary numbers, each list is a dot (input) represented a binary number
        :return: the predicted label of the dot
        """
        if isinstance(inputs, str):
            inputs = np.fromstring(inputs, sep=' ')
        print(f'inputs: {inputs}')
        print(f'weights: {self.weights}')
        print(f'dot: {np.dot(self.weights, inputs)}')
        self.last_input = inputs
        if np.dot(self.weights, inputs) >= self.threshold:
            return 1
        else:
            return -1

    def train(self, inputs, labels, learning_rate=0.01):
        """
        Train the perceptron

        :param inputs: list of 21 digits binary numbers, each list is a dot (input) represented a binary number
        :param labels: the labels of the dots
        """
        # initialize the weights to be zeros vector
        self.weights = np.zeros(len(self.input_neuron))

        # dictionary to keep track of incorrect predictions
        incorrect_predictions = {i: True for i in range(len(inputs))}

        while incorrect_predictions:
            for i in list(incorrect_predictions.keys()):
                prediction = self.predict(inputs[i])
                if prediction != labels[i]:
                    self.weights += learning_rate * \
                        (labels[i] - prediction) * inputs[i]
                else:
                    # remove correct predictions from the dictionary
                    incorrect_predictions.pop(i)

        print(f'weights: {self.weights}')

    def plot(self, dots, labels):
        """
        Plot the perceptron's decision boundary

        :param dots: list of 21 digits binary numbers (list of lists), each list is a dot (input) represented a binary number
        :param labels: list of labels for each dot, 1 if the dot has more than 15 ones, -1 otherwise
        """
        # transfer the dots to points in the 2D space where the x-axis is the number of ones and the y-axis is the number of zeros
        # use PCA to reduce the dimensionality of the data
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        dots = pca.fit_transform(dots)

        # plot the dots
        plt.figure()
        plt.scatter(dots[:, 0], dots[:, 1], c=labels)

        # plot the decision boundary using pca
        weights = pca.transform([self.weights])[0]
        x = np.linspace(-1, 1, 100)
        y = (weights[0] * x) / -weights[1]

        plt.plot(x, y, 'k-')
        plt.show()

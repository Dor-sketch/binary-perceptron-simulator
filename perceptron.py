"""
This program implements multiple perceptrons to classify 21 digits
binary numbers into 3 classes:
1. more that 15 ones
2. less than 15 ones
3. else
"""

import random
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np


class InputNeuron:
    def __init__(self, n_inputs=21):
        self.units = [random.choice([0, 1]) for _ in range(n_inputs)]

    def __len__(self):
        return len(self.units)


class Percpetron:
    def __init__(self, InputNeuron, threshold=1):
        """
        Initialize the perceptron

        :param InputNeuron: the input neuron
        :param threshold: the threshold for the perceptron.
        """
        self.input_neuron = InputNeuron
        self.weights = np.random.rand(len(self.input_neuron))
        for i in range(len(self.input_neuron)):
            self.weights[i] = 1
        self.threshold = threshold  # see readme for more info

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


class CombinedPercpetron:
    def __init__(self, n_inputs=21):
        self.one_perceptron = Percpetron(InputNeuron(n_inputs))
        for i in range(len(self.one_perceptron.weights)):
            self.one_perceptron.weights[i] = 1/16
        self.zero_perceptron = Percpetron(InputNeuron(n_inputs))
        for i in range(len(self.zero_perceptron.weights)):
            self.zero_perceptron.weights[i] = 1/6

    def train(self, numbers=None, dots=[], ones_labels=[], zeros_labels=[]):
        """
        Train the perceptron to classify the dots into 3 classes:
        1. more than 15 ones
        2. more than 15 zeros
        3. Else

        :param dots: list of 21 digits binary numbers (list of lists), each list is a dot (input) represented a binary number
        :param labels: list of labels for each dot, 1 if the dot has more than 15 ones, -1 if the dot has more than 15 zeros, 0 otherwise
        """
        if not numbers:
            numbers = []
            for _ in range(100000):
                number = np.array([random.randint(0, 1) for _ in range(21)])
                numbers.append(number)
                num_sum = sum(number)
                if num_sum >= 15:
                    ones_labels.append(1)
                    zeros_labels.append(-1)
                elif num_sum <= 6:
                    zeros_labels.append(1)
                    ones_labels.append(-1)
                else:
                    zeros_labels.append(-1)
                    ones_labels.append(-1)

        dots = numbers
        # train the perceptrons
        self.one_perceptron.train(dots, ones_labels)
        self.zero_perceptron.train(dots, zeros_labels)

    def predict(self, number):
        """
        Predict the label of the given dot

        :param dot: list of 21 digits binary numbers, each list is a dot (input) represented a binary number
        :return: the predicted label of the dot
        """
        print(f'number: {number}')
        print(f'one: {self.one_perceptron.predict(number)}')
        print(f'zero: {self.zero_perceptron.predict(number)}')
        if self.one_perceptron.predict(number) == 1:
            return 1
        elif self.zero_perceptron.predict(number) == -1:
            return -1
        else:
            return 0

    def plot(self):
        """
        Plot the perceptron's weights - no need for input
        """
        # plot just weights
        plt.figure()
        plt.scatter(range(21), self.one_perceptron.weights, c='r')
        plt.scatter(range(21), self.zero_perceptron.weights, c='b')
        plt.plot([0, 21], [0, 0], 'k-')
        plt.show()

    def plot2(self):
        """
        Plot the perceptron's decision boundary - no need for input
        """
        # create the dots
        numbers = []
        for _ in range(100):
            number = np.array([random.randint(0, 1) for _ in range(21)])
            numbers.append(number)

        # create the labels
        labels = []
        for number in numbers:
            if self.one_perceptron.predict(number) == 1:
                labels.append(1)
            elif self.zero_perceptron.predict(number) == -1:
                labels.append(-1)
            else:
                labels.append(0)

        # plot the dots as numbers for easier visualization
        plt.figure()
        plt.scatter(range(100), labels, c=labels)

        plt.show()

        # plot the decision boundary
        self.one_perceptron.plot(numbers, labels)


class TkGUI:
    def __init__(self, master, perceptron):
        self.master = master
        self.master.title('Perceptrons')
        self.master.geometry('300x300')
        self.perceptron = perceptron
        self.label = '0'
        self.label_var = tk.StringVar()
        self.label_var.set(self.label)
        self.label = tk.Label(self.master, textvariable=self.label_var)
        self.label.pack()
        self.entry = tk.Entry(self.master)
        self.entry.pack()
        self.button = tk.Button(
            self.master, text='Predict', command=self.predict)
        self.button.pack()
        self.button = tk.Button(self.master, text='Quit',
                                command=self.master.quit)
        self.button.pack()
        self.button = tk.Button(self.master, text='Plot', command=self.plot)
        self.button.pack()
        self.button = tk.Button(self.master, text='Train', command=self.train)
        self.button.pack()
        self.button = tk.Button(self.master, text='Plot2',
                                command=self.perceptron.plot2)
        self.button.pack()

    def predict(self):
        binary_number = self.entry.get()
        # turn to vector of 21 digits binary number
        binary_number = np.array([int(digit) for digit in binary_number])
        # pad with zeros to get 21 digits
        binary_number = np.pad(
            binary_number, (21 - len(binary_number), 0), 'constant')
        label = self.perceptron.predict(binary_number)
        if label == 1:
            label = 'More than 15 ones'
        elif label == -1:
            label = 'More than 15 zeros'
        else:
            label = 'Else'
        self.label_var.set(label)

    def plot(self):
        self.perceptron.plot()

    def train(self):
        self.perceptron.train()


def main():
    # create the perceptron
    perceptron = CombinedPercpetron()
    # train the perceptron

    # create the GUI
    root = tk.Tk()
    gui = TkGUI(root, perceptron)
    root.mainloop()


if __name__ == '__main__':
    main()

"""
This file contains the CombinedPercpetron class, which is a
combination of two perceptrons, one for classifying the dots
with more than 15 ones and the other for classifying the dots
with more than 15 zeros. The class also contains a method to
plot the perceptron's weights change over time.
"""

from Percpetron import Percpetron, InputNeuron
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from sklearn.decomposition import PCA


class CombinedPercpetron:
    def __init__(self, n_inputs=21, debug=False):
        self.one_perceptron = Percpetron(InputNeuron(n_inputs))
        self.zero_perceptron = Percpetron(InputNeuron(n_inputs))
        if debug:
            for i in range(n_inputs):
                self.one_perceptron.weights[i] = 1 / 16
                self.zero_perceptron.weights[i] = 1 / 6

    def weights(self):
        return self.one_perceptron.weights, self.zero_perceptron.weights

    def train(
        self, numbers=None, dots=[], ones_labels=[], zeros_labels=[], cycles=1000
    ):
        """
        Train the perceptron to classify the dots into 3 classes:
        1. more than 15 ones
        2. more than 15 zeros
        3. Else

        :param dots: list of 21 digits binary numbers (list of lists), each list is a dot (input) represented a binary number
        :param labels: list of labels for each dot, 1 if the dot has more than 15 ones, -1 if the dot has more than 15 zeros, 0 otherwise
        """
        random.seed(time.time())

        # set for ran
        if not numbers:
            numbers = []
            # plant seed for reproducibility
            for _ in range(cycles):
                number = np.array([random.randint(0, 1) for _ in range(21)])
                numbers.append(number)
                num_sum = sum(number)
                if num_sum > 15:
                    ones_labels.append(1)
                    zeros_labels.append(1)
                elif num_sum < 6:
                    ones_labels.append(-1)
                    zeros_labels.append(-1)
                else:
                    ones_labels.append(-1)
                    zeros_labels.append(1)

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
        print(f"number: {number}")
        print(f"one: {self.one_perceptron.predict(number)}")
        print(f"zero: {self.zero_perceptron.predict(number)}")
        print(
            f"The total number of ones: {sum(number)} and the total number of zeros: {21 - sum(number)}"
        )
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
        plt.scatter(range(21), self.one_perceptron.weights, c="r")
        plt.scatter(range(21), self.zero_perceptron.weights, c="b")
        plt.plot([0, 21], [0, 0], "k-")
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

        # add decimal representation of the numbers next to the dots
        for i in range(100):
            plt.text(i, labels[i], f"{int(''.join(map(str, numbers[i])), 2)}")

        plt.show()
        # plot the decision boundary
        self.one_perceptron.plot(numbers, labels)

    def get_uniform_numbers(self, samples=300):
        """
        Get uniform numbers for the 3 classes

        :param samples: the number of samples for each class
        :return: list of 21 digits binary numbers (list of lists), each list is a dot (input) represented a binary number
        """
        numbers = []
        first_class_count = 0
        second_class_count = 0
        while len(numbers) < samples:
            number = np.array([random.randint(0, 1) for _ in range(21)])
            numbers.append(number)
            num_sum = sum(number)
            if num_sum > 15 and first_class_count < samples / 3:
                first_class_count += 1
            elif num_sum < 6 and second_class_count < samples / 3:
                second_class_count += 1
            else:
                if first_class_count < samples / 3:
                    numbers.pop()
                    numbers.append(np.array([1 for _ in range(21)])
                    )
                    first_class_count += 1
                elif second_class_count < samples / 3:
                    numbers.pop()
                    numbers.append(np.array([0 for _ in range(21)])
                    )
                    second_class_count += 1
        return numbers

    def get_labels(self, numbers):
        """
        Get labels for the given numbers

        :param numbers: list of 21 digits binary numbers (list of lists), each list is a dot (input) represented a binary number
        :return: list of labels for each dot, 1 if the dot has more than 15 ones, -1 if the dot has more than 15 zeros, 0 otherwise
        """
        # create the labels
        labels = []
        for number in numbers:
            if self.one_perceptron.predict(number) == 1:
                labels.append(1)
            elif self.zero_perceptron.predict(number) == -1:
                labels.append(-1)
            else:
                labels.append(0)

        return labels

    def plot_decision_boundary(self, samples=300):
        """
        plot 3D decision boundary for the 3 classes
        """
        # create the dots randomly but make sure to have samples/3 dots for each class
        numbers = self.get_uniform_numbers(samples)
        labels = self.get_labels(numbers)

        # use 3D PCA to reduce the dimensionality of the data
        pca = PCA(n_components=3)
        dots = pca.fit_transform(numbers)

        # plot the dots
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(dots[:, 0], dots[:, 1], dots[:, 2], c=labels)

        # add text - sum of ones in each dot
        for i in range(samples):
            ax.text(dots[i, 0], dots[i, 1], dots[i, 2], f"{sum(numbers[i])}")

        # plot the decision boundary
        ones_weights = pca.transform([self.one_perceptron.weights])[0]
        weight_zero = pca.transform([self.zero_perceptron.weights])[0]
        x = np.linspace(-1, 1, samples)
        y = np.linspace(-1, 1, samples)
        x, y = np.meshgrid(x, y)

        z_one = (-ones_weights[0]*x - ones_weights[1]*y) / ones_weights[2]
        ax.plot_surface(x+1, y, -z_one, color='k', alpha=0.5)

        z_zero = (-weight_zero[0]*x - weight_zero[1]*y) / weight_zero[2]
        ax.plot_surface(x - 1, y, z_zero, color='b', alpha=0.5)

        plt.show()

    def animate_learning(self):
        self.train(cycles=5000)
        ones_history = self.one_perceptron.weights_history
        zeros_history = self.zero_perceptron.weights_history

        # trim the history when converging
        for i in range(len(ones_history) - 1, 0, -1):
            if np.array_equal(ones_history[i], ones_history[i - 1]):
                ones_history = ones_history[:i]
                break
        for i in range(len(zeros_history) - 1, 0, -1):
            if np.array_equal(zeros_history[i], zeros_history[i - 1]):
                zeros_history = zeros_history[:i]
                break

        # make sure the history has the same length
        min_len = min(len(ones_history), len(zeros_history))
        ones_history = ones_history[:min_len]
        zeros_history = zeros_history[:min_len]
        # use 3D PCA to reduce the dimensionality of the data and see the weights change over time 3d
        pca = PCA(n_components=3)
        ones_history = pca.fit_transform(ones_history)
        zeros_history = pca.transform(zeros_history)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        def get_surface(weights, x, y):
            z = (-weights[0] * x - weights[1] * y) / weights[2]
            return z

        def update_ani(frame, ones_history, zeros_history):
            ax.clear()
            numbers = self.get_uniform_numbers(300)
            labels = self.get_labels(numbers)
            pca = PCA(n_components=3)
            dots = pca.fit_transform(numbers)
            ax.scatter(dots[:, 0], dots[:, 1], dots[:, 2], c=labels)

            x = np.linspace(-2, 2, 2)
            y = np.linspace(-2, 2, 2)
            x, y = np.meshgrid(x, y)
            z_ones = get_surface(ones_history[frame], x, y)
            z_zeros = get_surface(zeros_history[frame], x, y)
            ax.plot_surface(x+1, y, -z_ones, color='b', rstride=1, cstride=1, alpha=0.5, linewidth=0, antialiased=False)
            ax.plot_surface(x-1, y, z_zeros, color='r', rstride=1, cstride=1, alpha=0.5, linewidth=0, antialiased=False)


        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        ani = animation.FuncAnimation(
            fig, update_ani, frames=len(ones_history), fargs=(ones_history, zeros_history), blit=False, interval=100, repeat=True
        )
        plt.show()

        # store the animation
        ani.save("weights_change.gif", writer="pillow")


# p = CombinedPercpetron(debug=True)
# p.plot_decision_boundary()
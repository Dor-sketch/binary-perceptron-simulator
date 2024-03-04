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


class CombinedPercpetron:
    def __init__(self, n_inputs=21):
        self.one_perceptron = Percpetron(InputNeuron(n_inputs))
        self.zero_perceptron = Percpetron(InputNeuron(n_inputs))

    def weights(self):
        return self.one_perceptron.weights, self.zero_perceptron.weights

    def train(
        self, numbers=None, dots=[], ones_labels=[], zeros_labels=[], cycles=20000
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

        # plot the decision boundary
        self.one_perceptron.plot(numbers, labels)

    def plot_real_time_training(self):
        """
        Plot the perceptron's weights change over time
        """
        ones_weights = []
        zeros_weights = []
        random.seed(0)
        # train the perceptrons
        for _ in range(5000):
            number = np.array([random.randint(0, 1) for _ in range(21)])
            num_sum = sum(number)
            if num_sum > 15:
                ones_labels = 1
                zeros_labels = 1
            elif num_sum < 6:
                zeros_labels = -1
                ones_labels = -1
            else:
                zeros_labels = 1
                ones_labels = -1
            self.one_perceptron.train([number], [ones_labels])
            self.zero_perceptron.train([number], [zeros_labels])
            ones_weights.append(self.one_perceptron.weights)
            zeros_weights.append(self.zero_perceptron.weights)

        return ones_weights, zeros_weights

    def plot_lines(self, ones_weights, zeros_weights):
        """
        Plot the perceptron's weights change over time
        """
        ones_weights, zeros_weights = self.plot_real_time_training()

        # create the figure and lines
        fig, axs = plt.subplots(len(ones_weights[0]), 2)
        axs = axs.flatten()  # flatten axs
        lines = [[ax.plot([], [], c="r")[0], ax.plot([], [], c="b")[0]] for ax in axs]

        # initialization function
        def init():
            for ax in axs:
                ax.set_xlim(0, 1000)
                ax.set_ylim(-1, 1)
            # flatten lines
            return [line for sublist in lines for line in sublist]

        # animation function
        def update(i):
            for weight_index in range(len(ones_weights[0])):
                lines[weight_index * 2][0].set_data(
                    range(i), [weight[weight_index] for weight in ones_weights[:i]]
                )
                lines[weight_index * 2 + 1][0].set_data(
                    range(i), [weight[weight_index] for weight in zeros_weights[:i]]
                )
            # flatten lines
            return [line for sublist in lines for line in sublist]

        # create animation
        ani = animation.FuncAnimation(
            fig, update, frames=range(1000), init_func=init, blit=True
        )

        plt.show()


#     def plot_real_time_training(self):
#         """
#         Plot the perceptron's weights change over time
#         """
#         ones_weights = []
#         zeros_weights = []

#         # train the perceptrons
#         for _ in range(1000):
#             number = np.array([random.randint(0, 1) for _ in range(21)])
#             num_sum = sum(number)
#             if num_sum > 15:
#                 ones_labels = 1
#                 zeros_labels = 1
#             elif num_sum < 6:
#                 zeros_labels = -1
#                 ones_labels = -1
#             else:
#                 zeros_labels = 1
#                 ones_labels = -1
#             self.one_perceptron.train([number], [ones_labels])
#             self.zero_perceptron.train([number], [zeros_labels])
#             ones_weights.append(self.one_perceptron.weights)
#             zeros_weights.append(self.zero_perceptron.weights)

#         # create the figure and lines
#         fig, ax = plt.subplots()
#         line1, = ax.plot([], [], c='r')
#         line2, = ax.plot([], [], c='b')

#         # initialization function
#         def init():
#             ax.set_xlim(0, 1000)
#             ax.set_ylim(-1, 1)
#             return line1, line2,

#         # animation function
#         # animation function
#         def update(i):
#             for weight_index in range(len(ones_weights[0])):
#                 line1.set_data(range(i), [weight[weight_index] for weight in ones_weights[:i]])
#                 line2.set_data(range(i), [weight[weight_index] for weight in zeros_weights[:i]])
#             return line1, line2,

#         # create animation
#         ani = animation.FuncAnimation(fig, update, frames=range(1000), init_func=init, blit=True, interval=10)

#         plt.show()
# # create the perceptron

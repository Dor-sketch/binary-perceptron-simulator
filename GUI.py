"""
This file contains the Tkinter GUI class for the perceptron.
"""

import random
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation


class TkGUI:
    def __init__(self, master, perceptron):
        self.master = master
        self.master.title("Perceptrons")
        self.master.geometry("920x670")
        self.perceptron = perceptron

        self.label_var = tk.StringVar()
        self.label = tk.Label(self.master, textvariable=self.label_var)
        self.label.pack()

        self.entry = tk.Entry(self.master)
        self.entry.pack()

        self.predict_button = tk.Button(
            self.master, text="Predict", command=self.predict
        )
        self.predict_button.pack()

        self.train_button = tk.Button(
            self.master, text="Train", command=self.train)
        self.train_button.pack()

        self.plot_button = tk.Button(
            self.master, text="Plot Weights", command=self.plot)
        self.plot_button.pack()

        self.plot_3d_button = tk.Button(
            self.master, text="Plot Linear Separation (2-D Plae)", command=self.plot_3d)
        self.plot_3d_button.pack()

        self.fix_weights_button = tk.Button(
            self.master, text="Fix weights", command=self.fix_weights)
        self.fix_weights_button.pack()

        self.animate_learning_button = tk.Button(
            self.master, text="Animate Learning", command=self.animate_learning)
        self.animate_learning_button.pack()

        # Canvas setup
        self.figure = Figure(figsize=(50, 5), facecolor="black")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH)

        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        toolbar.update()
        toolbar.pack(side=tk.TOP, expand=True, anchor="s")
        self.ax = self.figure.add_subplot(111, facecolor="black")

        # Initialize animation data
        self.data = np.random.rand(
            len(self.perceptron.one_perceptron.weights), 50)
        self.lines = [
            self.ax.plot(np.arange(50), i + 0.5 * self.data[i] * w, c="w")[0]
            for i, w in enumerate(self.perceptron.one_perceptron.weights)
        ]

        # Start the animation
        self.ani = animation.FuncAnimation(
            self.figure, self.update_ani, blit=True, interval=100, repeat=True
        )

    def update_ani(self, *args):
        """ Update the data and lines for the animation """
        for i in range(len(self.perceptron.one_perceptron.weights)):
            self.data[:, 1:] = self.data[:, :-1]
            frequency = (
                i / self.perceptron.one_perceptron.weights[i] * 10.0
            )  # Frequency of the wave
            if frequency == 0:
                frequency = 1

            phase_shift = i * np.pi / 10.0  # Unique phase shift for each wave
            self.data[:, 0] = (
                2
                * self.perceptron.one_perceptron.dot_product()
                * np.sin(
                    frequency
                    * (
                        self.perceptron.one_perceptron.weights[i]
                        * self.perceptron.one_perceptron.last_input[i]
                        + self.perceptron.one_perceptron.threshold
                    )
                    * np.pi
                    / 180
                    + phase_shift
                )
                * random.choice([-1, 1])
            )

            self.lines[i].set_ydata(i + 0.5 * self.data[i])
        return self.lines

    def predict(self):
        binary_number = self.entry.get()
        binary_vector = np.array(list(map(int, binary_number.zfill(21))))[:21]
        label = self.perceptron.predict(binary_vector)
        self.label_var.set(
            "More than 15 ones"
            if label == 1
            else "More than 15 zeros" if label == -1 else "Else"
        )
        self.canvas.draw()

    def plot(self):
        self.perceptron.plot()

    def train(self):
        self.perceptron.train()

    def plot_3d(self):
        self.perceptron.plot_decision_boundary()

    def fix_weights(self):
        self.perceptron.fix_weights()
        self.canvas.draw()

    def animate_learning(self):
        self.perceptron.animate_learning()
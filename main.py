"""
This is the main file for the project. It creates the perceptron and the GUI and runs the program.
"""

from CombinedPercpetron import CombinedPercpetron
from GUI import TkGUI
import tkinter as tk


def main():
    # create the perceptron
    perceptron = CombinedPercpetron()
    # train the perceptron

    # create the GUI and update it every 100ms
    root = tk.Tk()
    gui = TkGUI(root, perceptron)
    root.mainloop()


if __name__ == '__main__':
    main()

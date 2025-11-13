#!/usr/bin/env python3
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

def plot_big_o(n: np.ndarray,
               O1: np.ndarray,
               Ologn: np.ndarray,
               On: np.ndarray,
               Onlogn: np.ndarray,
               On2: np.ndarray,
               On3: np.ndarray,
               O2n: np.ndarray) -> Tuple[Figure, Axes]:
    """
    Plots different Big O complexities on a logarithmic scale.

    :param n: Input sizes (1D array)
    :param O1: O(1) complexity values
    :param Ologn: O(log n) values
    :param On: O(n) values
    :param Onlogn: O(n log n) values
    :param On2: O(n^2) values
    :param On3: O(n^3) values
    :param O2n: O(2^n) values
    :return: Tuple containing the Figure and Axes objects
    """

    # Use subplots() for clean type inference
    fig, ax = plt.subplots(figsize=(12, 8))  # fig: Figure, ax: Axes

    # Plot all curves; returns lists of Line2D
    lines: List[Line2D] = []
    lines += ax.plot(n, O1, label="O(1) - Constant")
    lines += ax.plot(n, Ologn, label="O(log n) - Logarithmic")
    lines += ax.plot(n, On, label="O(n) - Linear")
    lines += ax.plot(n, Onlogn, label="O(n log n) - Linearithmic")
    lines += ax.plot(n, On2, label="O(n²) - Quadratic")
    lines += ax.plot(n, On3, label="O(n³) - Cubic")
    lines += ax.plot(n, O2n, label="O(2^n) - Exponential")

    # Configure axes
    ax.set_yscale('log')
    ax.set_xlabel("Input size n")
    ax.set_ylabel("Operations (log scale)")
    ax.set_title("Big O Complexity Comparison")
    ax.legend()
    ax.grid(True, which="both", ls="--", lw=0.5)

    return fig, ax

# --------------------------
# Run script and display plot
# --------------------------
if __name__ == "__main__":
    n = np.arange(1, 21)  # Input sizes

    O1 = np.ones_like(n)
    Ologn = np.log2(n)
    On = n
    Onlogn = n * np.log2(n)
    On2 = n**2
    On3 = n**3
    O2n = 2**n

    fig, ax = plot_big_o(n, O1, Ologn, On, Onlogn, On2, On3, O2n)
    plt.show()  # Automatically display the figure

#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


#
def main():
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.grid()
    # distance to goal
    x1, x2 = 0, 20
    # reward
    y1, y2 = 1, 0
    n_pts = 1000
    slope = (y1 - y2) / (x1 - x2)

    z1 = np.linspace(0, x2, n_pts)
    z2 = slope * z1 + y1
    z3 = 1 - (z1 / x2) ** 0.4

    random_radius = 0.5
    n_cols = len(z1[z1 >= random_radius])
    low = np.zeros((1, n_cols))
    high = np.ones((1, n_pts - n_cols))
    z4 = np.hstack((high, low))
    z4 = z4.ravel()
    # DSRNN current
    ax.plot(z1, z2, label="Potential-Based Reward")
    # DSRNN proposed
    ax.plot(z1, z3, label="Shaped Reward")
    # other algos
    ax.plot(z1, z4, label="Sparse Reward")

    ax.set_ylabel("Reward")
    ax.set_xlabel("Distance to goal")
    ax.legend(loc="upper right")
    ax.set_title("Reward Comparison")
    # for axis in ax:
    #     axis.set_ylabel("Reward")
    #     axis.set_xlabel("Distance to goal")
    fig.set_figheight(5)
    fig.set_figwidth(8)
    plt.show()


if __name__ == "__main__":
    main()

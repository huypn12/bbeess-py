import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import experiments.results.bee_3 as res_3
import experiments.results.bee_5 as res_5
import experiments.results.bee_10 as res_10


def visualize_data(plot_name, data_hist, hist_label):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Number of survived bees.", fontsize=12)
    ax.set_ylabel("Number of samples.", fontsize=12)
    ax.bar(hist_label, data_hist, alpha=0.75, width=0.5)
    plt.savefig(plot_name)


def visualize_bee_3():
    visualize_data(
        "bee_3_data",
        np.array(res_3.synthetic_data),
        ["0", "1", "2", "3"],
    )


def visualize_bee_5():
    visualize_data(
        "bee_5_data",
        np.array(res_5.synthetic_data),
        ["0", "1", "2", "3", "4", "5"],
    )


def visualize_bee_10():
    visualize_data(
        "bee_10_data",
        np.array(res_10.synthetic_data),
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    )


def main():
    visualize_bee_3()
    visualize_bee_5()
    visualize_bee_10()


if __name__ == "__main__":
    main()
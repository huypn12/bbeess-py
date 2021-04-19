import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import experiments.results.bee_3 as res_3
import experiments.results.bee_5 as res_5


def visualize_data(plot_name, data_hist, hist_label):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_ylabel("Number of samples.", fontsize=12)
    plt.xticks(rotation=90)
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


def main():
    visualize_bee_3()
    visualize_bee_5()


if __name__ == "__main__":
    main()
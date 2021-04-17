import experiments.results.die as die_result

import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


def visualize_particles(plot_name, x, y):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("p", fontsize=12)
    ax.set_ylabel("q", fontsize=12)
    ax.grid(True, linestyle="-", color="0.75")
    ax.scatter(x, y, s=20, marker="o")
    plt.savefig(plot_name)


def main():
    x = [p[0] for p in die_result.particle_trace]
    y = [p[1] for p in die_result.particle_trace]
    visualize_particles("knuth_die_trueparams", x, y)


if __name__ == "__main__":
    main()
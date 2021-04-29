import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

zeroconf_runtime_total = [[6.083, 54.867], [9.50, 37.93]]
zeroconf_runtime_avg = [[0.32, 2.88], [0.5, 1.978]]

bees_runtime_total = [[5.917, 68.783], [29.517, 352.083], [3976.117, 581.833]]
bees_runtime_avg = [[0.312, 3.167], [1.553, 18.518], [209.237, 30.592]]

sir_runtime_total = [[19.567, 231.6], [165.033, 567.683], [965.7, 776.9]]
sir_runtime_avg = [[1.03, 11.838], [8.658, 28.962], [50.814, 37.945]]


def _visualize_runtime_zeroconf(fig_name, rf_y, sim_y, label_y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_labels = ["Zeroconf 4", "Zeroconf 10"]
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(label_y, fontsize=12)
    ax.plot(x_labels, rf_y, c="b", marker="o", label="RF-SMC")
    ax.plot(x_labels, sim_y, c="r", marker="o", label="SMC-ABC-SMC")
    ax.legend()
    plt.savefig(fig_name)


def visualize_runtime_zeroconf():
    rf_y = [p[0] for p in zeroconf_runtime_avg]
    sim_y = [p[1] for p in zeroconf_runtime_avg]
    label_y = "Average runtime of one iteration (in minutes)"
    _visualize_runtime_zeroconf("zeroconf_runtime_avg", rf_y, sim_y, label_y)
    rf_y = [p[0] for p in zeroconf_runtime_total]
    sim_y = [p[1] for p in zeroconf_runtime_total]
    label_y = "Total runtime (in minutes)"
    _visualize_runtime_zeroconf("zeroconf_runtime_total", rf_y, sim_y, label_y)


def _visualize_runtime_bees(fig_name, rf_y, sim_y, label_y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_labels = ["Bees 3", "Bees 5", "Bees 10"]
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(label_y, fontsize=12)
    ax.plot(x_labels, rf_y, c="b", marker="o", label="RF-SMC")
    ax.plot(x_labels, sim_y, c="r", marker="o", label="SMC-ABC-SMC")
    ax.legend()
    plt.savefig(fig_name)


def visualize_runtime_bees():
    rf_y = [p[0] for p in bees_runtime_avg]
    sim_y = [p[1] for p in bees_runtime_avg]
    label_y = "Average runtime of one iteration (in minutes)"
    _visualize_runtime_bees("bees_runtime_avg", rf_y, sim_y, label_y)
    rf_y = [p[0] for p in bees_runtime_total]
    sim_y = [p[1] for p in bees_runtime_total]
    label_y = "Total runtime (in minutes)"
    _visualize_runtime_bees("bees_runtime_total", rf_y, sim_y, label_y)


def _visualize_runtime_sir(fig_name, rf_y, sim_y, label_y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_labels = ["SIR(5,1,0)", "SIR(10,1,0)", "SIR(15,1,0)"]
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel(label_y, fontsize=12)
    ax.plot(x_labels, rf_y, c="b", marker="o", label="RF-SMC")
    ax.plot(x_labels, sim_y, c="r", marker="o", label="SMC-ABC-SMC")
    ax.legend()
    plt.savefig(fig_name)


def visualize_runtime_sir():
    rf_y = [p[0] for p in sir_runtime_avg]
    sim_y = [p[1] for p in sir_runtime_avg]
    label_y = "Average runtime of one iteration (in minutes)"
    _visualize_runtime_sir("sir_runtime_avg", rf_y, sim_y, label_y)
    rf_y = [p[0] for p in sir_runtime_total]
    sim_y = [p[1] for p in sir_runtime_total]
    label_y = "Total runtime (in minutes)"
    _visualize_runtime_sir("sir_runtime_total", rf_y, sim_y, label_y)


def main():
    visualize_runtime_zeroconf()
    visualize_runtime_bees()
    visualize_runtime_sir()


if __name__ == "__main__":
    main()
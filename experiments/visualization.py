import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import experiments.sir_trace as sir_trace


def str2array(s):
    # Remove space after [
    s = re.sub("\[ +", "[", s.strip())
    # Replace commas and spaces
    s = re.sub("[,\s]+", ", ", s)
    return np.array(ast.literal_eval(s))


def visualize_llh(plot_name, truep, estp, x, y, t):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("alpha", fontsize=12)
    ax.set_ylabel("beta", fontsize=12)
    ax.grid(True, linestyle="-", color="0.75")
    ax.plot(
        truep[0],
        truep[1],
        color="green",
        marker="x",
        markersize=12,
    )
    ax.plot(
        estp[0],
        estp[1],
        color="red",
        marker="x",
        markersize=12,
    )
    points = ax.scatter(x, y, s=20, c=t, marker="o", cmap=cm.jet)
    plt.colorbar(points, label="ln(P(S_obs|(alpha, beta)))")
    plt.savefig(plot_name)


def visualize_data(plot_name, data_hist, hist_label):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Distribution over BSCCs", fontsize=12)
    ax.set_ylabel("Number of samples.", fontsize=12)
    ax.bar(hist_label, data_hist, alpha=0.75, width=0.5)
    plt.savefig(plot_name)


def visualize_dist(plot_name, truep, estp, x, y, t):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("alpha", fontsize=12)
    ax.set_ylabel("beta", fontsize=12)
    ax.grid(True, linestyle="-", color="0.75")
    ax.plot(
        truep[0],
        truep[1],
        color="green",
        marker="x",
        markersize=12,
    )
    ax.plot(
        estp[0],
        estp[1],
        color="red",
        marker="x",
        markersize=12,
    )
    points = ax.scatter(x, y, s=20, c=t, marker="o", cmap=cm.jet)
    plt.colorbar(points, label="l2_dist(S_obs, S_sim)")
    plt.savefig(plot_name)


def visualize_sir510():
    visualize_data(
        "sir510_data",
        str2array("[ 421  834 1126 1362 1851 4406]"),
        [
            "(0,0,6)",
            "(1,0,5)",
            "(2,0,4)",
            "(3,0,3)",
            "(4,0,2)",
            "(5,0,1)",
        ],
    )

    true_p = str2array(sir_trace.sir510_true_p)
    particles = str2array(sir_trace.sir510_rf_particles)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = str2array(sir_trace.sir510_rf_particles_weight)
    estp = str2array(sir_trace.sir510_rf_particles_mean)
    visualize_llh("sir510_rfsmc", true_p, estp, alpha, beta, llh)

    particles = str2array(sir_trace.sir510_smc_particles)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = str2array(sir_trace.sir510_smc_particles_weight)
    llh = llh / np.sum(llh)
    estp = str2array(sir_trace.sir510_smc_particles_mean)
    visualize_dist("sir510_abcsmc", true_p, estp, alpha, beta, llh)


def main():
    visualize_sir510()


if __name__ == "__main__":
    main()
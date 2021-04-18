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


def visualize_data(plot_name, data_hist, hist_label):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Distribution over BSCCs", fontsize=12)
    ax.set_ylabel("Number of samples.", fontsize=12)
    plt.xticks(rotation=90)
    ax.bar(hist_label, data_hist, alpha=0.75, width=0.5)
    plt.savefig(plot_name)


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
    plt.colorbar(points, label="ln(P(D_obs|(alpha, beta)))")
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


def visualize_sir1510():
    visualize_data(
        "ZeroConf (4 tries) data",
        np.array([41, 9959]),
        [
            "Failed",
            "OK",
        ],
    )

    # No BSCC merging, rf
    true_p = np.array([0.10554747, 0.44965874])
    particles = np.array(zer)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = str2array(sir_trace.gSirRfPweight["15_1_0"])
    est_p = str2array(sir_trace.gSirRfPmean["15_1_0"])
    visualize_llh("sir1510_rfsmc", true_p, est_p, alpha, beta, llh)

    # No BSCC merging, sim
    particles = str2array(sir_trace.gSirSimPtrace["15_1_0"])
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = str2array(sir_trace.gSirSimPweight["15_1_0"])
    llh = 1 / llh
    est_p = str2array(sir_trace.gSirRfPmean["15_1_0"])
    visualize_dist("sir1510_abcsmc", true_p, est_p, alpha, beta, llh)

    # BSCC merging, rf
    particles = str2array(sir_trace.gSirRfPtrace["15_1_0_a_few"])
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = str2array(sir_trace.gSirRfPweight["15_1_0_a_few"])
    est_p = str2array(sir_trace.gSirRfPmean["15_1_0_a_few"])
    visualize_llh("sir1510_rfsmc_few", true_p, est_p, alpha, beta, llh)

    # BSCC merging, sim
    particles = str2array(sir_trace.gSirSimPtrace["15_1_0_a_few"])
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = str2array(sir_trace.gSirSimPweight["15_1_0_a_few"])
    llh = 1 / llh
    est_p = str2array(sir_trace.gSirSimPmean["15_1_0_a_few"])
    visualize_dist("sir1510_abcsmc_few", true_p, est_p, alpha, beta, llh)


def main():
    visualize_sir1510()


if __name__ == "__main__":
    main()
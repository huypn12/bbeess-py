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


def visualize_sir1510():
    visualize_data(
        "sir510_data",
        str2array(
            "[  50  181  302  455  539  567  582  566  541  553  574  528  512  586 875 2589]"
        ),
        [
            "bscc_0_0_16",
            "bscc_1_0_15",
            "bscc_2_0_14",
            "bscc_3_0_13",
            "bscc_4_0_12",
            "bscc_5_0_11",
            "bscc_6_0_10",
            "bscc_7_0_9",
            "bscc_8_0_8",
            "bscc_9_0_7",
            "bscc_10_0_6",
            "bscc_11_0_5",
            "bscc_12_0_4",
            "bscc_13_0_3",
            "bscc_14_0_2",
            "bscc_15_0_1",
        ],
    )
    visualize_data(
        "sir510_data_merged",
        str2array("[ 54 146]"),
        ["(*,0,r>8)", "(*,0,r<=8)"],
    )

    # No BSCC merging, rf
    true_p = str2array("[0.01149928 0.06211052]")
    particles = str2array(sir_trace.gSirRfPtrace["15_1_0"])
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


def visualize_510():
    pass


def visualize_1010():
    pass


def main():
    visualize_sir1510()


if __name__ == "__main__":
    main()
import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import experiments.results.sir_5_1_0 as sir_5
import experiments.results.sir_10_1_0 as sir_10
import experiments.results.sir_15_1_0 as sir_15
import experiments.results.sir_15_1_0_a_few_mod as sir_15_a_few_mod
import experiments.results.sir_15_1_0_a_few as sir_15_a_few


def visualize_data(plot_name, data_hist, hist_label):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Distribution over BSCCs", fontsize=12)
    ax.set_ylabel("Number of samples.", fontsize=12)
    plt.xticks(rotation=90)
    ax.bar(hist_label, data_hist, alpha=0.75, width=0.5)
    plt.tight_layout()
    plt.savefig(plot_name)


def visualize_llh(plot_name, truep, estp, x, y, t):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("alpha", fontsize=12)
    ax.set_ylabel("beta", fontsize=12)
    ax.grid(True, linestyle="-", color="0.75")
    (true_point,) = ax.plot(
        truep[0],
        truep[1],
        color="green",
        marker="x",
        markersize=12,
        linestyle="None",
    )
    (estimated_point,) = ax.plot(
        estp[0],
        estp[1],
        color="red",
        marker="x",
        markersize=12,
        linestyle="None",
    )
    plt.legend(
        [true_point, estimated_point],
        ["True parameter", "Estimated parameter"],
        frameon=True,
        loc="best",
        prop={"size": 8},
    )
    points = ax.scatter(x, y, s=20, c=t, marker="o", cmap=cm.jet)
    plt.colorbar(points, label="Log-likelihood to the observed data")
    plt.savefig(plot_name)


def visualize_dist(plot_name, truep, estp, x, y, t):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("alpha", fontsize=12)
    ax.set_ylabel("beta", fontsize=12)
    ax.grid(True, linestyle="-", color="0.75")
    (true_point,) = ax.plot(
        truep[0],
        truep[1],
        color="green",
        marker="x",
        markersize=12,
        linestyle="None",
    )
    (estimated_point,) = ax.plot(
        estp[0],
        estp[1],
        color="red",
        marker="x",
        markersize=12,
        linestyle="None",
    )
    plt.legend(
        [true_point, estimated_point],
        ["True parameter", "Estimated parameter"],
        frameon=True,
        loc="best",
        prop={"size": 8},
    )
    points = ax.scatter(x, y, s=20, c=t, marker="o", cmap=cm.jet)
    plt.colorbar(points, label="Euclidean distance between simulated and observed data")
    plt.savefig(plot_name)


def visualize_sir510():
    visualize_data(
        "sir510_data",
        np.array(sir_5.synthetic_data),
        [
            "(0,0,6)",
            "(1,0,5)",
            "(2,0,4)",
            "(3,0,3)",
            "(4,0,2)",
            "(5,0,1)",
        ],
    )

    # No BSCC merging, rf
    true_p = np.array(sir_5.true_param)
    particles = np.array(sir_5.particle_trace_rf)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(sir_5.particle_weight_rf)
    est_p = np.array(sir_5.particle_mean_rf)
    visualize_llh("sir510_rfsmc", true_p, est_p, alpha, beta, llh)

    # No BSCC merging, sim
    particles = np.array(sir_5.particle_trace_sim)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(sir_5.particle_weight_sim)
    # llh = 1 / llh
    est_p = np.array(sir_5.particle_mean_sim)
    visualize_dist("sir510_abcsmc", true_p, est_p, alpha, beta, llh)


def visualize_sir1010():
    visualize_data(
        "sir1010_data",
        np.array(sir_10.synthetic_data),
        [
            "(0,0,11)",
            "(1,0,10)",
            "(2,0,9)",
            "(3,0,8)",
            "(4,0,7)",
            "(5,0,6)",
            "(6,0,5)",
            "(7,0,4)",
            "(8,0,3)",
            "(9,0,2)",
            "(10,0,1)",
        ],
    )

    # No BSCC merging, rf
    true_p = np.array(sir_10.true_param)
    particles = np.array(sir_10.particle_trace_rf)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(sir_10.particle_weight_rf)
    est_p = np.array(sir_10.particle_mean_rf)
    visualize_llh("sir1010_rfsmc", true_p, est_p, alpha, beta, llh)

    # No BSCC merging, sim
    particles = np.array(sir_10.particle_trace_sim)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(sir_10.particle_weight_sim)
    # llh = 1 / llh
    est_p = np.array(sir_10.particle_mean_sim)
    visualize_dist("sir1010_abcsmc", true_p, est_p, alpha, beta, llh)


def visualize_sir1510():
    visualize_data(
        "sir1510_data",
        np.array(sir_15.synthetic_data),
        [
            "(0,0,16)",
            "(1,0,15)",
            "(2,0,14)",
            "(3,0,13)",
            "(4,0,12)",
            "(5,0,11)",
            "(6,0,10)",
            "(7,0,9)",
            "(8,0,8)",
            "(9,0,7)",
            "(10,0,6)",
            "(11,0,5)",
            "(12,0,4)",
            "(13,0,3)",
            "(14,0,2)",
            "(15,0,1)",
        ],
    )

    # No BSCC merging, rf
    true_p = np.array(sir_15.true_param)
    particles = np.array(sir_15.particle_trace_rf)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(sir_15.particle_weight_rf)
    est_p = np.array(sir_15.particle_mean_rf)
    visualize_llh("sir1510_rfsmc", true_p, est_p, alpha, beta, llh)

    # No BSCC merging, sim
    particles = np.array(sir_15.particle_trace_sim)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(sir_15.particle_weight_sim)
    # llh = 1 / llh
    est_p = np.array(sir_15.particle_mean_sim)
    visualize_dist("sir1510_abcsmc", true_p, est_p, alpha, beta, llh)


def manual():
    true_p = np.array(sir_15.true_param)
    particles = np.array(sir_15_a_few_mod.particle_trace_sim)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(sir_15_a_few_mod.particle_weight_sim)
    llh = 1 / llh
    est_p = np.array(sir_15_a_few_mod.particle_mean_sim)
    visualize_dist("sir1510_merged_bscc_abcsmc_mod", true_p, est_p, alpha, beta, llh)


def few():
    true_p = np.array(sir_15.true_param)
    particles = np.array(sir_15_a_few.particle_trace_sim)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(sir_15_a_few.particle_weight_sim)
    llh = 1 / llh
    est_p = np.array(sir_15_a_few.particle_mean_sim)
    visualize_dist("sir1510_merged_bscc_abcsmc", true_p, est_p, alpha, beta, llh)

    particles = np.array(sir_15_a_few.particle_trace_rf)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(sir_15_a_few.particle_weight_rf)
    est_p = np.array(sir_15_a_few.particle_mean_rf)
    visualize_llh("sir1510_merged_bscc_rfsmc", true_p, est_p, alpha, beta, llh)


def main():
    visualize_sir510()
    visualize_sir1010()
    visualize_sir1510()


if __name__ == "__main__":
    main()
    manual()
    few()
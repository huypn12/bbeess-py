import numpy as np
import re
import ast
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import experiments.results.zeroconf_4 as res_4
import experiments.results.zeroconf_10 as res_10


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
    plt.xticks(rotation=45)
    ax.bar(hist_label, data_hist, alpha=0.75, width=0.75)
    plt.tight_layout()
    plt.savefig(plot_name)


def visualize_llh(plot_name, truep, estp, x, y, t):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("p", fontsize=12)
    ax.set_ylabel("q", fontsize=12)
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
    points = ax.scatter(x, y, s=20, c=t, marker="o", cmap=cm.jet)
    plt.colorbar(points, label="Log-likelihood to the observed data")
    plt.legend(
        [true_point, estimated_point],
        ["True parameter", "Estimated parameter"],
        frameon=True,
        loc="best",
        prop={"size": 8},
    )
    plt.savefig(plot_name)


def visualize_dist(plot_name, truep, estp, x, y, t):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("p", fontsize=12)
    ax.set_ylabel("q", fontsize=12)
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


def visualize_zeroconf_4():
    visualize_data(
        "zeroconf4_data",
        np.array([41, 9959]),
        [
            "Failed",
            "OK",
        ],
    )

    # rf
    true_p = np.array([0.10554747, 0.44965874])
    particles = np.array(res_4.particle_trace_rf)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(res_4.particle_weight_rf)
    est_p = np.array(res_4.particle_mean_rf)
    visualize_llh("zeroconf4_rf", true_p, est_p, alpha, beta, llh)

    # sim
    particles = np.array(res_4.particle_trace_sim)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(res_4.particle_weight_sim)
    est_p = np.array(res_4.particle_mean_sim)
    visualize_dist("zeroconf4_sim", true_p, est_p, alpha, beta, llh)


def visualize_zeroconf_10():
    visualize_data(
        "zeroconf10_data",
        np.array(res_10.synthetic_data),
        [
            "Failed",
            "OK",
        ],
    )

    # rf
    true_p = np.array(res_10.true_param)
    particles = np.array(res_10.particle_trace_rf)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(res_10.particle_weight_rf)
    est_p = np.array(res_10.particle_mean_rf)
    visualize_llh("zeroconf10_rf", true_p, est_p, alpha, beta, llh)

    # sim
    particles = np.array(res_10.particle_trace_sim)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.array(res_10.particle_weight_sim)
    est_p = np.array(res_10.particle_mean_sim)
    visualize_dist("zeroconf10_sim", true_p, est_p, alpha, beta, llh)


def main():
    visualize_zeroconf_4()
    visualize_zeroconf_10()


if __name__ == "__main__":
    main()
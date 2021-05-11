import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


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
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("r0", fontsize=12)
    ax.set_ylabel("r1", fontsize=12)
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
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("r0", fontsize=12)
    ax.set_ylabel("r1", fontsize=12)
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


def visualize_jchain_full_data():
    visualize_data(
        "BSCC, Julia Chain, 2 params",
        np.array([313, 1277, 8410]),
        [
            "bscc_1",
            "bscc_2",
            "bscc_3",
        ],
    )

    # rf
    true_p = np.array([0.0333821, 0.12909493])
    particle_trace_rf = np.load(
        "/home/huypn12/Works/mcss/bbeess-py/temp-files/log/fm2021_1/14:19:35.370369_jchain2.pm_RationalFunction_trace.npy"
    )
    particles = np.array(particle_trace_rf)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.load(
        "/home/huypn12/Works/mcss/bbeess-py/temp-files/log/fm2021_1/14:19:35.371845_jchain2.pm_RationalFunction_weight.npy"
    )
    print(llh)
    particle_mean_rf = [0.49433744, 0.51080695]
    est_p = np.array(particle_mean_rf)
    visualize_llh("jchain2_full_rf", true_p, est_p, alpha, beta, llh)

    # sim
    particle_trace_sim = np.load(
        "/home/huypn12/Works/mcss/bbeess-py/temp-files/log/fm2021_1/15:35:11.611929_jchain2.pm_Simulation_trace.npy"
    )
    particles = np.array(particle_trace_sim)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.load(
        "/home/huypn12/Works/mcss/bbeess-py/temp-files/log/fm2021_1/15:35:11.613365_jchain2.pm_Simulation_weight.npy"
    )
    particle_mean_sim = [0.12303183, 0.06734973]
    est_p = np.array(particle_mean_sim)
    visualize_dist("jchain2_full_sim", true_p, est_p, alpha, beta, llh)


def visualize_jchain_few_data():
    visualize_data(
        "BSCC, Julia Chain, 2 params",
        np.array([7, 27, 166]),
        [
            "bscc_1",
            "bscc_2",
            "bscc_3",
        ],
    )
    # rf
    true_p = np.array([0.0333821, 0.12909493])
    particle_trace_rf = np.load(
        "/home/huypn12/Works/mcss/bbeess-py/temp-files/log/fm2021_1/14:21:38.276271_jchain2.pm_RationalFunction_trace.npy"
    )
    particles = np.array(particle_trace_rf)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.load(
        "/home/huypn12/Works/mcss/bbeess-py/temp-files/log/fm2021_1/14:21:38.277758_jchain2.pm_RationalFunction_weight.npy"
    )
    print(llh)
    particle_mean_rf = [0.49433744, 0.51080695]
    est_p = np.array(particle_mean_rf)
    visualize_llh("jchain2_few_rf", true_p, est_p, alpha, beta, llh)

    # sim
    particle_trace_sim = np.load(
        "/home/huypn12/Works/mcss/bbeess-py/temp-files/log/fm2021_1/16:01:19.682741_jchain2.pm_Simulation_trace.npy"
    )
    particles = np.array(particle_trace_sim)
    alpha = [p[0] for p in particles]
    beta = [p[1] for p in particles]
    llh = np.load(
        "//home/huypn12/Works/mcss/bbeess-py/temp-files/log/fm2021_1/16:01:19.684314_jchain2.pm_Simulation_weight.npy"
    )
    particle_mean_sim = [0.12303183, 0.06734973]
    est_p = np.array(particle_mean_sim)
    visualize_dist("jchain2_few_sim", true_p, est_p, alpha, beta, llh)


def main():
    visualize_jchain_full_data()
    visualize_jchain_few_data()


if __name__ == "__main__":
    main()
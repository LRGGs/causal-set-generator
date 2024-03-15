from collections import defaultdict
from itertools import chain

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis.an_utils import (
    PATH_NAMES,
    read_file,
    linear, label_map, calculate_reduced_chi2
)
from src.utils import nrange

# matplotlib.use("TkAgg")


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def mean_angular_deviations_per_path_per_n(graphs, d):
    plt.grid(which="major")
    plt.grid(which="minor")
    plt.tick_params(
        axis="both", labelsize=10, direction="in", top=True, right=True, which="both"
    )
    for path_name in PATH_NAMES:
        if path_name not in ["shortestx", "randomx", "greedy_mx"]:
            n_angles = defaultdict(list)
            for graph in graphs:
                angles = []
                path = graph["paths"][path_name]
                for i in range(len(path) - 2):
                    v1 = np.array(path[i + 1]) - np.array(path[i])
                    v2 = np.array(path[i + 2]) - np.array(path[i + 1])
                    ang = angle_between(v1, v2)
                    angles.append(ang)

                n_angles[graph["n"]].append(np.mean(angles))
                # n_angles[graph["n"]] += angles

            x_data = np.array(list(n_angles.keys()))
            y_data = np.array([np.mean(v) for v in n_angles.values()])
            y_err = np.array([np.std(v) / np.sqrt(len(v)) for v in n_angles.values()])

            plt.errorbar(
                x_data,
                y_data,
                yerr=y_err,
                ls="none",
                capsize=5,
                marker=".",
                label=label_map[path_name],
            )
            if path_name in ["longest", "greedy_e"]:
                popt, pcov = curve_fit(
                    f=linear, xdata=x_data, ydata=y_data, p0=[1], sigma=y_err
                )
                fit = 10
                y_fit = np.array([popt]*len(x_data))
                red_chi, pval = calculate_reduced_chi2(y_data[-fit:], y_fit[-fit:], y_err[-fit:])
                plt.plot(x_data, y_fit, label=f"{label_map[path_name]} fit: ${popt[0]:.2f}$\n $\chi^2_\\nu={red_chi:.3f}$, p value = {pval:.2f}")

    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 0.875))
    plt.xlabel("Number of Nodes")
    plt.ylabel("Angular Deviation Between Edges, Rad")
    # plt.ylim(0.5, 0.94)
    plt.savefig(
        f"images/Mean Angular Deviation per Path {d}D.png",
        bbox_inches="tight",
        # facecolor="#F2F2F2",
        transparent=True,
    )
    plt.clf()


if __name__ == "__main__":
    d = 2
    graphs = read_file(nrange(100, 10000, 100), 0.1, d, 100, extra="paths")
    mean_angular_deviations_per_path_per_n(graphs, d)

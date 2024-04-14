from collections import defaultdict
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

from src.analysis.an_utils import (
    PATH_NAMES,
    read_file,
    calculate_reduced_chi2,
    fit_expo1, fit_expo_corr, flat,
)
from src.utils import nrange

# matplotlib.use("TkAgg")

parameters = {
    "longest": [2.11, .25],
    "greedy_e": [2, 0.25],
    "random": [2.8, 0.05],
    "greedy_m": [2.8, 0.05]
}


def length_of_paths_with_n(graphs, d):
    for path in PATH_NAMES:
        n_lengths = defaultdict(list)
        for graph in graphs:
            n_lengths[graph["n"]].append(len(graph["paths"][path]))

        x_data = list(n_lengths.keys())
        y_data = [np.mean(v) for v in n_lengths.values()]
        y_err = [np.std(v) / np.sqrt(len(v)) for v in n_lengths.values()]

        # cut = 20
        # x_data, y_data, y_err = x_data[cut:], y_data[cut:], y_err[cut:]

        if path == "shortest":
            plt.plot(x_data, [3]*len(x_data), label="Shortest fit: 3.0")
            plt.plot(
                x_data,
                y_data,
                ls="none",
                marker=".",
                label=path,
                zorder=100
            )
        elif path in ["longestd"]:
            plt.errorbar(
                x_data,
                y_data,
                yerr=y_err,
                ls="none",
                capsize=2,
                marker=",",
                label=path,
            )
            fit_expo_corr(x_data, y_data, y_err, path, params=parameters[path], p=True, d=d)
        else:
            plt.errorbar(
                x_data,
                y_data,
                yerr=y_err,
                ls="none",
                capsize=2,
                marker=",",
                label=path,
                zorder=100
            )
            fit_expo1(x_data, y_data, y_err, path, params=parameters[path])

    plt.grid(which="major")
    plt.grid(which="minor")
    plt.loglog()
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 0.875))
    # plt.title(f"Path Lengths Against Number of Nodes")
    plt.xlabel("Number of  Nodes", fontsize=12)
    plt.ylabel("Path Length", fontsize=12)
    plt.savefig(
        f"images/Path Lengths Against Number of Nodes {d}D.png",
        bbox_inches="tight",
        dpi=1000,
        # facecolor="#F2F2F2",
        transparent=True,
    )
    plt.clf()


if __name__ == "__main__":
    d = 2
    graphs = read_file(nrange(100, 8000, 100), 1, d, 250, extra="paths")
    length_of_paths_with_n(graphs, d=d)

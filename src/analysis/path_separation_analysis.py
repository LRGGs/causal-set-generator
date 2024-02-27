from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis.utils import PATH_NAMES, read_file, fit_expo, fit_expo_poly
from src.utils import nrange


# matplotlib.use("TkAgg")


def separation_from_geodesic_by_path(graphs):
    fit_legends = []
    plot_lines = []

    for path in PATH_NAMES:
        n_seps = defaultdict(list)
        for graph in graphs:
            n_seps[graph["n"]].append(
                np.mean([abs(pos[1]) for pos in graph["paths"][path]])
            )
            # n_seps[graph["n"]] += [abs(pos[1]) for pos in graph["paths"][path]]

        x_data = list(n_seps.keys())
        y_data = [np.mean(v) for v in n_seps.values()]
        y_err = [np.std(v) / np.sqrt(len(v)) for v in n_seps.values()]
        plt.errorbar(
            x_data,
            y_data,
            yerr=y_err,
            ls="none",
            capsize=5,
            marker=".",
            label=path,
        )

        if path in ["longest", "greedy_e", "greedy_o"]:
            l, legend = fit_expo(x_data, y_data, y_err, path, params=[0.4, -0.2])
            plot_lines.append(l)
            fit_legends.append(legend)

    legend1 = plt.legend(plot_lines, fit_legends, loc='upper right', bbox_to_anchor=(1, 0.485),
                         fancybox=True, shadow=True, labelspacing=-0.4)

    plt.legend(loc='lower left', bbox_to_anchor=(0.3, 0.65), ncol=3, columnspacing=0.5,
               fancybox=True, shadow=True)
    plt.gca().add_artist(legend1)
    plt.title(f"Mean Separation from Geodesic per Path")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Mean Separation from Geodesic")
    plt.savefig("images/Mean Separation from Geodesic per Path.png", bbox_inches="tight")
    plt.clf()

    fit_legends = []
    plot_lines = []

    for path in PATH_NAMES:
        n_seps = defaultdict(list)
        for graph in graphs:
            n_seps[graph["n"]].append(
                max([abs(pos[1]) for pos in graph["paths"][path]])
            )

        x_data = list(n_seps.keys())
        y_data = [np.mean(v) for v in n_seps.values()]
        y_err = [np.std(v) / np.sqrt(len(v)) for v in n_seps.values()]
        plt.errorbar(
            x_data,
            y_data,
            yerr=y_err,
            ls="none",
            capsize=5,
            marker=".",
            label=path,
        )

        if path in ["longest", "greedy_e", "greedy_o"]:
            l, legend = fit_expo(x_data, y_data, y_err, path, params=[0.4, -0.2])
            plot_lines.append(l)
            fit_legends.append(legend)

    legend1 = plt.legend(plot_lines, fit_legends, loc='upper right', bbox_to_anchor=(1, 0.4),
                         fancybox=True, shadow=True)

    plt.legend(loc='lower left', bbox_to_anchor=(-0.08, -0.23, 1, 0.2), ncol=5, columnspacing=0.7,
               fancybox=True, shadow=True)
    plt.gca().add_artist(legend1)
    plt.title(f"Max Separation from Geodesic per Path")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Max Separation from Geodesic")
    plt.savefig("images/Max Separation from Geodesic per Path.png", bbox_inches="tight")


if __name__ == "__main__":
    graphs = read_file(nrange(200, 10000, 50), 0.1, 2, 100, extra="paths")
    separation_from_geodesic_by_path(graphs)

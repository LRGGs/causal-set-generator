from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis.utils import PATH_NAMES, read_file, fit_expo, fit_expo_poly
from src.utils import nrange


# matplotlib.use("TkAgg")


def separation_from_geodesic_by_path(graphs):
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

        fit_expo(x_data, y_data, y_err, path, params=[0.2, -0.2])
        # fit_expo_poly(x_data, y_data, y_err, path, params=[0.2, -0.2, 1, 0, 0])

    plt.legend()
    plt.title(f"Mean Separation from Geodesic Per Path")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Mean Separation from Geodesic")
    plt.show()

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

        fit_expo(x_data, y_data, y_err, path, params=[0.4, -0.2])
        # fit_expo_poly(x_data, y_data, y_err, path, params=[0.4, -0.2, 1, 0, 0])

    plt.legend()
    plt.title(f"Max Separation from Geodesic Per Path")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Max Separation from Geodesic")
    plt.show()


if __name__ == "__main__":
    graphs = read_file(nrange(500, 5000, 20), 0.1, 2, 100)
    separation_from_geodesic_by_path(graphs)

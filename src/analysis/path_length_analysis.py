from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis.an_utils import (
    PATH_NAMES,
    read_file,
    calculate_reduced_chi2,
    fit_expo,
)
from src.utils import nrange

# matplotlib.use("TkAgg")


def length_of_paths_with_n(graphs):
    for path in PATH_NAMES:
        n_lengths = defaultdict(list)
        for graph in graphs:
            n_lengths[graph["n"]].append(len(graph["paths"][path]))

        x_data = list(n_lengths.keys())
        y_data = [np.mean(v) for v in n_lengths.values()]
        y_err = [np.std(v) / np.sqrt(len(v)) for v in n_lengths.values()]
        plt.errorbar(
            x_data,
            y_data,
            yerr=y_err,
            ls="none",
            capsize=5,
            marker=".",
            label=path,
        )

        fit_expo(x_data, y_data, y_err, path, params=[1.7, 0.5])

    plt.legend()
    plt.title(f"Path Lengths Against Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Path Length")
    plt.show()


if __name__ == "__main__":
    graphs = read_file(nrange(200, 10000, 50), 0.1, 2, 100, extra="paths")
    length_of_paths_with_n(graphs)

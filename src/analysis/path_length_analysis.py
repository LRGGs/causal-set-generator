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
    fit_expo, fit_expo_corr,
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
        if path in ["longest"]:
            par1 = np.linspace(2.1, 2.13, 10)
            par2 = np.linspace(-0.001, 0.001, 10)
            par3 = np.linspace(-0.5, -0.4, 10)
            vars = list(product(par1, par2, par3))
            results = []
            for var in tqdm(vars):
                try:
                    _, _, _, xi, m_vals = fit_expo_corr(x_data, y_data, y_err, path, params=var)
                    results.append((xi, var, m_vals))
                except Exception as e:
                    print(e)
                    print(var)
            best_var = min(results, key=lambda i: i[0])[1:]
            print(best_var)
            _, _, _, xi = fit_expo_corr(x_data, y_data, y_err, path, params=best_var, p=True)

    plt.loglog()
    plt.legend()
    plt.title(f"Path Lengths Against Number of Nodes")
    plt.xlabel("Number of  Nodes")
    plt.ylabel("Path Length")
    plt.show()


if __name__ == "__main__":
    graphs = read_file(nrange(200, 10000, 50), 0.1, 2, 100, extra="paths", specific="big_temp_data.json")
    length_of_paths_with_n(graphs)

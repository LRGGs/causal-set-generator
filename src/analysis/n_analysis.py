import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from src.analysis.utils import PATH_NAMES, read_pickle
from src.utils import nrange
import numpy as np
from scipy.optimize import curve_fit

# matplotlib.use("TkAgg")


def length_of_paths_with_n(graphs):
    for path in ["longest", "greedy"]:
        n_lengths = defaultdict(list)
        for graph in graphs:
            n_lengths[len(graph["nodes"])].append(len(graph["paths"][path]))

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

        def f(x, a, b):
            return x**a * b

        params = [0.5, 1.7]
        popt, pcov = curve_fit(f=f, xdata=x_data, ydata=y_data, p0=params, sigma=y_err)
        error = np.sqrt(np.diag(pcov))
        print(f"{popt[1]} * x ^ {popt[0]}")
        print(f"error: {error}")

        poptreg = np.polyfit(np.log(x_data), np.log(y_data), deg=1)
        print(f"log(y) = {poptreg[0]} * log(x) + {poptreg[1]}")

        plt.plot(x_data, [f(x, *popt) for x in x_data], label=f"{path} fit")

    plt.legend()
    plt.title(f"Path Lengths Against Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Path Length")

    plt.show()


def length_of_paths_with_interval(graphs):
    ns = []
    path_lengths = {name: [] for name in PATH_NAMES}
    for graph in graphs:
        ns.append(len(graph["interval"]))
        for name in PATH_NAMES:
            path_lengths[name].append(len(graph["paths"][name]))

    for name in PATH_NAMES:
        plt.plot(ns, path_lengths[name], label=name)

    plt.legend()
    plt.title(f"Path Lengths Against Size of Interval")
    plt.xlabel("Length of Interval")
    plt.ylabel("Path Length")

    plt.show()


def interval_node_discrepancy(graphs):
    ns, intervals = [], []
    for graph in graphs:
        intervals.append(len(graph["interval"]))
        ns.append(len(graph["nodes"]))
    plt.plot(ns, [n - i for n, i in zip(ns, intervals)])

    plt.title(f"Difference Between Number of Nodes and Interval Size")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Number of Nodes - Interval Size")

    plt.show()


if __name__ == "__main__":
    graphs = read_pickle(nrange(100, 7000, 100), 0.1, 2, 5)
    length_of_paths_with_n(graphs)
    # length_of_paths_with_interval(graphs)
    # interval_node_discrepancy(graphs)

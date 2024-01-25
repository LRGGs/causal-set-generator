import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from src.analysis.utils import PATH_NAMES, read_pickle
from scipy.optimize import curve_fit
# matplotlib.use("TkAgg")


def mean_distance_by_path(graphs):
    total_seps = []
    for graph in graphs:
        paths = []
        for name in PATH_NAMES:
            path = graph["paths"].get(name)
            paths.append(set(list(sum(path, ()))[1:-1]))
        mean_seps = []
        for path in paths:
            mean_seps.append(
                np.sqrt(
                    np.mean([graph["nodes"][node]["position"][1] ** 2 for node in path])
                )
            )
        total_seps.append(mean_seps)
    means = np.mean(total_seps, axis=0)
    stdvs = np.std(total_seps, axis=0) / np.sqrt(len(total_seps))
    plt.errorbar(PATH_NAMES, means, yerr=stdvs, ls="none", capsize=5, marker="x")
    plt.title(f"Mean Separation from Geodesic for Each Path")
    plt.xlabel("Path")
    plt.ylabel("Mean Separation")
    plt.show()


def greatest_distance_by_path(graphs):
    all_seps = []
    for graph in graphs:
        paths = []
        for name in PATH_NAMES:
            path = graph["paths"].get(name)
            paths.append(set(list(sum(path, ()))[1:-1]))
        largest_seps = []
        for path in paths:
            largest_seps.append(
                max([abs(graph["nodes"][node]["position"][1]) for node in path])
            )
        all_seps.append(largest_seps)
    means = np.mean(all_seps, axis=0)
    stdvs = np.std(all_seps, axis=0) / np.sqrt(len(all_seps))
    plt.errorbar(PATH_NAMES, means, yerr=stdvs, ls="none", capsize=5, marker="x")
    plt.title(f"Largest Separation from Geodesic for Each Path")
    plt.xlabel("Path")
    plt.ylabel("largest Separation")
    plt.show()


if __name__ == "__main__":
    graphs = read_pickle(10000, 0.5, 2, 100)
    mean_distance_by_path(graphs)
    greatest_distance_by_path(graphs)

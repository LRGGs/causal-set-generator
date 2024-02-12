from collections import defaultdict
from itertools import chain

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis.utils import PATH_NAMES, read_file, fit_expo, fit_inv_poly, fit_linear
from src.utils import nrange

# matplotlib.use("TkAgg")


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def mean_angular_deviations_per_path_per_n(graphs):
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

                # n_angles[graph["n"]].append(np.mean(angles))
                n_angles[graph["n"]] += angles

            x_data = list(n_angles.keys())
            y_data = [np.mean(v) for v in n_angles.values()]
            y_err = [np.std(v) / np.sqrt(len(v)) for v in n_angles.values()]
            plt.errorbar(
                x_data,
                y_data,
                yerr=y_err,
                ls="none",
                capsize=5,
                marker=".",
                label=path_name,
            )

            # fit_expo(x_data, y_data, y_err, path_name, params=[0.5, 0.002])
            # fit_inv_poly(x_data, y_data, y_err, path_name, params=[1, 0, 0])
            fit_linear(x_data, y_data, y_err, path_name, params=[0.5])

    plt.legend()
    plt.xlabel("Number of Nodes")
    plt.ylabel("Angular Deviation Between Edges")
    plt.show()


if __name__ == "__main__":
    graphs = read_file(nrange(200, 10000, 50), 0.1, 2, 100, extra="paths")
    mean_angular_deviations_per_path_per_n(graphs)

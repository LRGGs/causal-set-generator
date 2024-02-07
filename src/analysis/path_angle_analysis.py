from collections import defaultdict
from itertools import chain

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis.utils import PATH_NAMES, read_file
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
        if path_name not in ["shortestx", "randomx"]:
            n_angles = defaultdict(list)
            n_errors = defaultdict(list)
            for graph in graphs:
                angles = []
                path = graph["paths"][path_name]
                for i in range(len(path) - 2):
                    v1 = np.array(path[i + 1]) - np.array(path[i])
                    v2 = np.array(path[i + 2]) - np.array(path[i + 1])
                    ang = angle_between(v1, v2)
                    # if v2[1] < v1[1]:
                    #     ang *= - 1
                    angles.append(ang)
                n_angles[graph["n"]].append(np.mean(angles))
            for key, values in n_angles.items():
                n_angles[key] = np.mean(values)
                n_errors[key] = np.std(values) / len(values) ** 0.5
            plt.errorbar(
                list(n_angles.keys()),
                list(n_angles.values()),
                label=path_name,
                yerr=list(n_errors.values()),
                ls="none",
                capsize=5,
                marker=".",
            )
    plt.legend()
    plt.xlabel("Number of Nodes")
    plt.ylabel("Angular Deviation Between Edges")
    plt.show()


if __name__ == "__main__":
    graphs = read_file(nrange(200, 10000, 50), 0.1, 2, 100, extra="paths")
    mean_angular_deviations_per_path_per_n(graphs)

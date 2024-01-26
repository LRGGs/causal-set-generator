from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis.utils import PATH_NAMES, read_pickle
from src.utils import nrange

# matplotlib.use("TkAgg")


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def mean_deviation_by_path(graphs):
    total_angs = []
    for graph in graphs:
        paths = []
        for name in PATH_NAMES:
            path = graph["paths"].get(name)
            paths.append(set(list(sum(path, ()))))
        mean_angs = []
        for path in paths:
            path = sorted(list(path))
            angs = []
            for i in range(len(path) - 2):
                v1 = (
                    graph["nodes"][path[i + 1]]["position"]
                    - graph["nodes"][path[i]]["position"]
                )
                v2 = (
                    graph["nodes"][path[i + 2]]["position"]
                    - graph["nodes"][path[i + 1]]["position"]
                )
                if all(v1) == 0 or all(v2) == 0:
                    print(path[i], path[i + 1])
                angs.append(angle_between(v1, v2))
            mean_angs.append(np.sqrt(np.mean([ang**2 for ang in angs])))
        total_angs.append(mean_angs)
    means = np.mean(total_angs, axis=0)
    stdvs = np.std(total_angs, axis=0) / np.sqrt(len(total_angs))
    plt.errorbar(PATH_NAMES, means, yerr=stdvs, ls="none", capsize=5, marker="x")
    plt.title(f"Mean Angular Deviation for Each Path")
    plt.xlabel("Path")
    plt.ylabel("Mean Angular Deviation")
    plt.show()


def greatest_deviation_by_path(graphs):
    total_angs = []
    for graph in graphs:
        paths = []
        for name in PATH_NAMES:
            path = graph["paths"].get(name)
            paths.append(set(list(sum(path, ()))))
        mean_angs = []
        for path in paths:
            path = sorted(list(path))
            angs = []
            for i in range(len(path) - 2):
                v1 = (
                    graph["nodes"][path[i + 1]]["position"]
                    - graph["nodes"][path[i]]["position"]
                )
                v2 = (
                    graph["nodes"][path[i + 2]]["position"]
                    - graph["nodes"][path[i + 1]]["position"]
                )
                angs.append(angle_between(v1, v2))
            mean_angs.append(max([abs(ang) for ang in angs]))
        total_angs.append(mean_angs)
    means = np.mean(total_angs, axis=0)
    stdvs = np.std(total_angs, axis=0) / np.sqrt(len(total_angs))
    plt.errorbar(PATH_NAMES, means, yerr=stdvs, ls="none", capsize=5, marker="x")
    plt.title(f"Max Angular Deviation for Each Path")
    plt.xlabel("Path")
    plt.ylabel("Max Angular Deviation")
    plt.show()


def mean_angular_deviations_per_path_per_n(graphs):
    for path_name in PATH_NAMES:
        if path_name != "shortest":
            n_angles = defaultdict(list)
            n_errors = defaultdict(list)
            for graph in graphs:
                angles = []
                path = graph["paths"][path_name]
                path = set(list(sum(path, ())))
                path = sorted(list(path))
                for i in range(len(path) - 2):
                    v1 = (
                            graph["nodes"][path[i + 1]]["position"]
                            - graph["nodes"][path[i]]["position"]
                    )
                    v2 = (
                            graph["nodes"][path[i + 2]]["position"]
                            - graph["nodes"][path[i + 1]]["position"]
                    )
                    angles.append(angle_between(v1, v2))
                # TODO: do i need to abs() the angles here?
                n_angles[len(graph["nodes"])].append(np.mean([abs(a) for a in angles]))
            for key, values in n_angles.items():
                n_angles[key] = np.mean(values)
                n_errors[key] = np.std(values) / len(values) ** 0.5
            plt.errorbar(list(n_angles.keys()), list(n_angles.values()), label=path_name, yerr=list(n_errors.values()), ls="none", capsize=5, marker=".")
    plt.legend()
    plt.xlabel("Number of Nodes")
    plt.ylabel("Angular Deviation Between Edges")
    plt.show()


if __name__ == "__main__":
    # graphs = read_pickle(10000, 0.5, 2, 100)
    # mean_deviation_by_path(graphs)
    # greatest_deviation_by_path(graphs)

    graphs = read_pickle(nrange(100, 7000, 100), 0.1, 2, 5)
    mean_angular_deviations_per_path_per_n(graphs)
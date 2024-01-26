import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from src.analysis.utils import PATH_NAMES, read_pickle

# matplotlib.use("TkAgg")


def mean_distance_by_order(order_collections, orders=10):
    means = []
    for graph in order_collections:
        order_means = []
        for i in range(orders):
            order_means.append(np.sqrt(np.mean([node[1] ** 2 for node in graph[i]])))
        means.append(order_means)
    total_means = np.mean(means, axis=0)
    total_stdvs = np.std(means, axis=0) / np.sqrt(len(means))
    x_data, y_data, y_err = [i for i in range(orders)], total_means, total_stdvs
    plt.errorbar(
        x_data, y_data, yerr=y_err, ls="none", capsize=5, marker=".", label="mean sep"
    )

    popt = np.polyfit(x_data, y_data, deg=7)
    print(popt)

    plt.plot(x_data, np.poly1d(popt)(x_data), label="mean sep fit")

    plt.legend()
    plt.title(f"Mean Separation from Geodesic for the First {orders} Orders")
    plt.xlabel("Order")
    plt.ylabel("Mean Separation")

    plt.show()


def max_distance_by_order(order_collections, orders=10):
    means = []
    for graph in order_collections:
        order_means = []
        for i in range(orders):
            order_means.append(max([abs(node[1]) for node in graph[i]]))
        means.append(order_means)
    total_means = np.mean(means, axis=0)
    total_stdvs = np.std(means, axis=0) / np.sqrt(len(means))
    x_data, y_data, y_err = [i for i in range(orders)], total_means, total_stdvs
    plt.errorbar(
        x_data, y_data, yerr=y_err, ls="none", capsize=5, marker=".", label="max sep"
    )

    popt = np.polyfit(x_data, y_data, deg=7)
    print(popt)

    plt.plot(x_data, np.poly1d(popt)(x_data), label="max sep fit")

    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Maximum Separation from Geodesic for the First {orders} Orders")
    plt.xlabel("Order")
    plt.ylabel("Maximum Separation")
    plt.show()


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


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def mean_deviation_by_path(graphs):
    total_angs = []
    j = -1
    for graph in graphs:
        j += 1
        paths = []
        for name in PATH_NAMES:
            path = graph["paths"].get(name)
            paths.append(set(list(sum(path, ()))))
        mean_angs = []
        for path in paths:
            path = sorted(list(path))
            angs = []
            if len(path) - 1 == 0:
                print(path, print(j))
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


if __name__ == "__main__":
    graphs = read_pickle(10000, 0.5, 2, 100)
    order_collections = [graph["order_collections"] for graph in graphs]
    # mean_distance_by_order(order_collections, 50)
    # max_distance_by_order(order_collections, 50)
    # mean_distance_by_path(graphs)
    # greatest_distance_by_path(graphs)
    mean_deviation_by_path(graphs)
    greatest_deviation_by_path(graphs)

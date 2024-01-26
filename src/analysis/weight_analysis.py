from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use("TkAgg")
import seaborn as sns
from scipy.optimize import curve_fit

from src.analysis.utils import PATH_NAMES, read_pickle
from src.utils import nrange


def mean_distance_by_weight(order_collections, orders=10):
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


def max_distance_by_weight(order_collections, orders=10):
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


def weight_n_distance_heatmap(graphs):
    weights = 25
    n_weight_seps = defaultdict(list)
    n_weight_errs = defaultdict(list)
    for graph in graphs:
        mean_seps = []
        for weight in graph["order_collections"]:
            mean_sep = np.mean([abs(pos[1]) for pos in weight])
            mean_seps.append(mean_sep)
        n_weight_seps[len(graph["nodes"])].append(mean_seps[:weights])
    for key, value in n_weight_seps.items():
        n_weight_seps[key] = np.mean(value, axis=0)
        n_weight_errs[key] = np.std(value, axis=0) / np.sqrt(len(value))
    vals = np.array(list(n_weight_seps.values()))

    min_y, max_y = float(min(n_weight_seps.keys())), float(max(n_weight_seps.keys()))
    min_x, max_x = 0.0, float(weights)
    plt.figure(figsize=(5, 9))
    plt.imshow(vals, extent=(min_x, max_x, min_y, max_y), aspect=0.01)
    plt.xlabel("Weight")
    plt.ylabel("Number of Nodes")
    colorbar = plt.colorbar()
    colorbar.set_label('Mean Separation From Geodesic')

    plt.show()

    plt.errorbar(
        list(n_weight_seps.keys()),
        [x[0] for x in n_weight_seps.values()],
        yerr=[x[0] for x in n_weight_errs.values()],
        ls="none",
        capsize=5,
        marker=".",
    )
    plt.xlabel("Number of Nodes")
    plt.ylabel("Mean Separation from Geodesic for Zeroth Weight")
    plt.show()

    xs, ys = list(n_weight_seps.keys()), [max(v) for v in n_weight_seps.values()]
    poptreg = np.polyfit(xs, ys, deg=1)
    print(f"sep = {poptreg[0]} * n + {poptreg[1]}")
    plt.plot(xs, ys)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Mean Separation from Geodesic for 25th Weight")
    plt.show()

    select_n = list(n_weight_seps.keys())[14::16]
    for key, value in n_weight_seps.items():
        if key in select_n:
            y_vals = np.array(list(value)) / (poptreg[0] * key + poptreg[1])
            plt.plot(np.arange(weights), y_vals, ls="none", marker=".", label=key)

    plt.legend()
    plt.xlabel("order")
    plt.ylabel(f"mean separation from geodesic normalised by:\n {poptreg[0]} * N + {poptreg[1]}")
    plt.show()


if __name__ == "__main__":
    # graphs = read_pickle(10000, 0.5, 2, 100)
    # order_collections = [graph["order_collections"] for graph in graphs]
    # mean_distance_by_weight(order_collections, 50)
    # max_distance_by_weight(order_collections, 50)

    graphs = read_pickle(nrange(4000, 10000, 200), 0.1, 2, 5)
    weight_n_distance_heatmap(graphs)

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
    for graph in graphs:
        if len(graph["nodes"]) > 4000:
            mean_seps = []
            for weight in graph["order_collections"]:
                mean_sep = np.mean([abs(pos[1]) for pos in weight])
                mean_seps.append(mean_sep)
            n_weight_seps[len(graph["nodes"])].append(mean_seps[:weights])
    for key, value in n_weight_seps.items():
        n_weight_seps[key] = np.mean(value, axis=0)
    vals = np.array(list(n_weight_seps.values()))

    plt.figure(figsize=(8, 12))
    plt.imshow(vals)
    plt.yticks(np.arange(len(n_weight_seps.keys())), labels=list(n_weight_seps.keys()))
    plt.xticks(np.arange(weights), rotation="vertical")
    plt.xlabel("Weight")
    plt.ylabel("Number of Nodes")
    colorbar = plt.colorbar()
    colorbar.set_label('Mean Separation From Geodesic')

    plt.show()

    xs, ys = list(n_weight_seps.keys()), [max(v) for v in n_weight_seps.values()]
    poptreg = np.polyfit(xs, ys, deg=1)
    print(f"sep = {poptreg[0]} * n + {poptreg[1]}")
    plt.plot(xs, ys)
    plt.show()

    select_n = list(n_weight_seps.keys())[7::8]
    for key, value in n_weight_seps.items():
        if key in select_n:
            y_vals = np.array(list(value)) / (poptreg[0] * key + poptreg[1])
            plt.plot(np.arange(weights), y_vals, ls="none", marker=".", label=key)

    plt.legend()
    plt.xlabel("order")
    plt.ylabel("mean separation from geodesic")
    plt.show()


if __name__ == "__main__":
    # graphs = read_pickle(10000, 0.5, 2, 100)
    # order_collections = [graph["order_collections"] for graph in graphs]
    # mean_distance_by_weight(order_collections, 50)
    # max_distance_by_weight(order_collections, 50)

    graphs = read_pickle(nrange(4000, 10000, 200), 0.1, 2, 5)
    weight_n_distance_heatmap(graphs)

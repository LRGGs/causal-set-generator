import itertools
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use("TkAgg")
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

from src.analysis.utils import PATH_NAMES, read_file
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
    weights = 50
    for graph in graphs:
        for i, weight in enumerate(graph["weight_collections"]):
            if i < weights and len(weight) == 0:
                weights = i
    n_weight_mean_seps = defaultdict(list)
    n_weight_mean_errs = defaultdict(list)
    for graph in graphs:
        mean_seps = []
        for weight in graph["weight_collections"]:
            mean_sep = np.mean([abs(pos[1]) for pos in weight])
            mean_seps.append(mean_sep)
        n_weight_mean_seps[graph["n"]].append(mean_seps[:weights])
    for key, value in n_weight_mean_seps.items():
        n_weight_mean_seps[key] = np.mean(value, axis=0)
        n_weight_mean_errs[key] = np.std(value, axis=0) / np.sqrt(len(value))
    vals = np.flip(np.array(list(n_weight_mean_seps.values())), axis=0)

    min_y, max_y = float(min(n_weight_mean_seps.keys())), float(max(n_weight_mean_seps.keys()))
    min_x, max_x = 0.0, float(weights)
    plt.figure(figsize=(5, 9))
    plt.imshow(vals, extent=(min_x, max_x, min_y, max_y), aspect=0.01)
    plt.xlabel("Weight")
    plt.ylabel("Number of Nodes")
    colorbar = plt.colorbar()
    colorbar.set_label('Mean Separation From Geodesic')

    plt.show()

    def f(x, a, b):
        return a * x ** b

    As = []
    Bs = []
    Ws = list(range(weights))

    for i in range(weights):
        xss = list(n_weight_mean_seps.keys())
        yss = [v[i] for v in n_weight_mean_seps.values()]
        y_errss = [v[i] for v in n_weight_mean_errs.values()]

        xs, ys, y_errs = [], [], []
        for x, y, e in zip(xss, yss, y_errss):
            if not np.isnan(y):
                xs.append(x)
                ys.append(y)
                y_errs.append(e)

        params = [-1, 1]
        popt, pcov = curve_fit(f=f, xdata=xs, ydata=ys, p0=params, sigma=y_errs)
        error = np.sqrt(np.diag(pcov))
        As.append(popt[0])
        Bs.append(popt[1])

        if i == 0 or i == weights-1:
            print(f"weight {i}: ({popt[0]}+-{error[0]}) * N ^ ({popt[1]}+-{error[1]})")
            plt.errorbar(xs, ys, yerr=y_errs, ls="none", capsize=5, marker=".", label="data")
            plt.plot(xs, [f(x, *popt) for x in xs], label="fit")

            plt.xlabel("Number of Nodes")
            plt.ylabel(f"Mean Separation from Geodesic for Weight {i}")
            plt.show()

    plt.plot(Ws, As, label="factor: a")
    plt.plot(Ws, Bs, label="exponent: b")
    plt.xlabel("weight")
    plt.ylabel("value")
    plt.title("a * N ** b")
    plt.legend()
    plt.show()

    select_n = list(n_weight_mean_seps.keys())[0::int(len(n_weight_mean_seps.keys())/10)]
    for key, value in n_weight_mean_seps.items():
        y_vals = np.array(list(value))
        if key in select_n:
            plt.plot(np.arange(weights), y_vals, ls="none", marker=".", label=key)

    plt.legend()
    plt.xlabel("weight")
    plt.ylabel(f"mean separation from geodesic")
    plt.show()

    collapsed_y = []
    for key, value in n_weight_mean_seps.items():
        y_vals = []
        for val in value:
            y_vals.append(val/f(key, As[-1], Bs[-1]))
        collapsed_y.append(y_vals)
        if key in select_n:
            plt.plot(np.arange(weights), y_vals, ls="none", marker=".", label=key)

    plt.legend()
    plt.xlabel("weight")
    plt.ylabel(f"mean separation from geodesic normalised by ??")
    plt.show()

    x_data, y_data, y_err = np.arange(weights), np.mean(collapsed_y, axis=0), np.std(collapsed_y, axis=0)/len(collapsed_y)**0.5
    plt.errorbar(x_data, y_data, y_err, ls="none", marker="x", capsize=5, label="data")
    popt = np.polyfit(x_data, y_data, deg=7)
    print(f"seventh order polynomial with with constants: {popt}")

    plt.plot(np.arange(weights), np.poly1d(popt)(x_data), label="fit")

    plt.legend()
    plt.xlabel("weight")
    plt.ylabel(f"mean separation from geodesic normalised by: ??")
    plt.show()

    results = []
    passed = 0
    for dataset1, dataset2 in itertools.combinations(collapsed_y, 2):
        _, p_value = ks_2samp(dataset1, dataset2)
        results.append(p_value)
        if p_value > 0.9:
            passed += 1
    print(f"mean KS_2samp P val: {np.mean(results)}")
    print(f"number of pairs accepted at 10%: {passed}/{len(results)}")

    data = np.array([list(vals) for vals in n_weight_mean_seps.values()])
    zeroth = data[:, 0]
    first = data[:, 1]
    counts_0, bins = np.histogram(zeroth)
    counts_1, bins = np.histogram(first, bins)
    plt.stairs(counts_0, bins, label="zeroth")
    plt.stairs(counts_1, bins, label="first")
    plt.legend()
    plt.show()
    _, p_value = ks_2samp(counts_0, counts_1)
    print(f"KS_2samp P val: {p_value}")


if __name__ == "__main__":
    # graphs = read_pickle(10000, 0.5, 2, 100)
    # order_collections = [graph["order_collections"] for graph in graphs]
    # mean_distance_by_weight(order_collections, 50)
    # max_distance_by_weight(order_collections, 50)

    graphs = read_file(nrange(500, 10000, 60), 0.1, 2, 50)
    weight_n_distance_heatmap(graphs)

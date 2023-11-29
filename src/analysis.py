import pickle
from src.rgg import Node, Order, Paths
import numpy as np
import matplotlib.pyplot as plt


def read_pickle(n, r, d, i):
    with open(
        f"../results/N-{n}__R-{str(r).replace('.', '-')}__D-{d}__I-{i}", "rb"
    ) as fp:
        file = pickle.load(fp)
        return file


def mean_distance_by_order(order_collections, orders=10):
    means = []
    for graph in order_collections:
        order_means = []
        for i in range(orders):
            order_means.append(np.sqrt(np.mean([node[1]**2 for node in graph[i]])))
        means.append(order_means)
    total_means = np.mean(means, axis=0)
    total_stdvs = np.std(means, axis=0)
    plt.errorbar([i for i in range(orders)], total_means, yerr=total_stdvs, ls='none', capsize=5, marker="x")
    plt.title(f"Mean Separation from Geodesic for the First {orders} Orders")
    plt.xlabel("Order")
    plt.ylabel("Mean Separation")

    plt.show()


def mean_distance_by_path(graphs, orders=10):
    pass


if __name__ == '__main__':
    graphs = read_pickle(10000, 1, 2, 100)
    mean_distance_by_order([graph["order_collections"] for graph in graphs])
import pickle
from src.rgg import Node, Order, Paths
import numpy as np
import matplotlib.pyplot as plt

PATH_NAMES = ["longest", "greedy", "random", "shortest"]


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


def mean_distance_by_path(graphs):
    total_seps = []
    for graph in graphs:
        paths = []
        for name in PATH_NAMES:
            path = getattr(graph["paths"], name)
            paths.append(set(list(sum(path, ()))[1:-1]))
        mean_seps = []
        for path in paths:
            mean_seps.append(np.sqrt(np.mean([graph["nodes"][node].position[1]**2 for node in path])))
        total_seps.append(mean_seps)
    means = np.mean(total_seps, axis=0)
    stdvs = np.std(total_seps, axis=0)/np.sqrt(len(total_seps))
    plt.errorbar([i for i in PATH_NAMES], means, yerr=stdvs, ls='none', capsize=5, marker="x")
    plt.title(f"Mean Separation from Geodesic for Each Path")
    plt.xlabel("Path")
    plt.ylabel("Mean Separation")
    plt.show()


def greatest_distance_by_path(graphs):
    all_seps = []
    for graph in graphs:
        paths = []
        for name in PATH_NAMES:
            path = getattr(graph["paths"], name)
            paths.append(set(list(sum(path, ()))[1:-1]))
        largest_seps = []
        for path in paths:
            largest_seps.append(max([abs(graph["nodes"][node].position[1]) for node in path]))
        all_seps.append(largest_seps)
    means = np.mean(all_seps, axis=0)
    stdvs = np.std(all_seps, axis=0)/np.sqrt(len(all_seps))
    plt.errorbar([i for i in PATH_NAMES], means, yerr=stdvs, ls='none', capsize=5, marker="x")
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
    total_devs = []
    for graph in graphs:
        paths = []
        for name in PATH_NAMES:
            if name != "shortest":
                path = getattr(graph["paths"], name)
                paths.append(list(set(list(sum(path, ()))[1:-1])))
        for path in paths:
            devs = []
            for n, j in zip(path[:-1], path[1:]):
                v1 = graph["nodes"][j].position
                v2 = graph["nodes"][n].position
                dev = angle_between(v1, v2)
                devs.append(dev)
            mean = np.mean(devs)
            total_devs.append([mean])
    means = np.mean(total_devs, axis=0)
    stdvs = np.std(total_devs, axis=0)/np.sqrt(len(total_devs))
    plt.errorbar([i for i in PATH_NAMES if i != "shortest"], means, yerr=stdvs, ls='none', capsize=5, marker="x")
    plt.title(f"Mean Angular Deviation for Each Path")
    plt.xlabel("Path")
    plt.ylabel("Mean Angular Deviation")
    plt.show()


def greatest_deviation_by_path(graphs):
    total_devs = []
    for graph in graphs:
        paths = []
        for name in PATH_NAMES:
            if name != "shortest":
                path = getattr(graph["paths"], name)
                paths.append(set(list(sum(path, ()))[1:-1]))
        for path in paths:
            devs = []
            for n, j in zip(list(path)[:-1], list(path)[1:]):
                devs.append(angle_between(graph["nodes"][j].position, graph["nodes"][n].position))
            total_devs.append(max(devs))
    means = np.mean(total_devs, axis=0)
    stdvs = np.std(total_devs, axis=0)/np.sqrt(len(total_devs))
    plt.errorbar([i for i in PATH_NAMES if i != "shortest"], means, yerr=stdvs, ls='none', capsize=5, marker="x")
    plt.title(f"Max Angular Deviation for Each Path")
    plt.xlabel("Path")
    plt.ylabel("Max Angular Deviation")
    plt.show()


if __name__ == '__main__':
    graphs = read_pickle(10000, 1, 2, 100)
    # mean_distance_by_order([graph["order_collections"] for graph in graphs])
    mean_distance_by_path(graphs)
    greatest_distance_by_path(graphs)
    mean_deviation_by_path(graphs)
    greatest_deviation_by_path(graphs)
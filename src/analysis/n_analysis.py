import matplotlib.pyplot as plt

from src.analysis.utils import PATH_NAMES, read_pickle
from src.utils import nrange


def length_of_paths_with_n(graphs):
    ns = []
    path_lengths = {name: [] for name in PATH_NAMES}
    for graph in graphs:
        ns.append(len(graph["nodes"]))
        for name in PATH_NAMES:
            path_lengths[name].append(len(graph["paths"][name]))

    for name in PATH_NAMES:
        plt.plot(ns, path_lengths[name], label=name)

    plt.legend()
    plt.title(f"Path Lengths Against Number of Nodes")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Path Length")

    plt.show()


def length_of_paths_with_interval(graphs):
    ns = []
    path_lengths = {name: [] for name in PATH_NAMES}
    for graph in graphs:
        ns.append(len(graph["interval"]))
        for name in PATH_NAMES:
            path_lengths[name].append(len(graph["paths"][name]))

    for name in PATH_NAMES:
        plt.plot(ns, path_lengths[name], label=name)

    plt.legend()
    plt.title(f"Path Lengths Against Size of Interval")
    plt.xlabel("Length of Interval")
    plt.ylabel("Path Length")

    plt.show()


def interval_node_discrepancy(graphs):
    ns, intervals = [], []
    for graph in graphs:
        intervals.append(len(graph["interval"]))
        ns.append(len(graph["nodes"]))
    plt.plot(ns, [n - i for n, i in zip(ns, intervals)])

    plt.title(f"Difference Between Number of Nodes and Interval Size")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Number of Nodes - Interval Size")

    plt.show()


if __name__ == "__main__":
    graphs = read_pickle(nrange(1000, 10000, 100), 0.5, 2, 1)
    length_of_paths_with_n(graphs)
    length_of_paths_with_interval(graphs)
    interval_node_discrepancy(graphs)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations


class Node():
    def __init__(self, pos, targets):
        pass


class Graph():
    def __init__(self, n, radius, d):
        self.n = n
        self.d = d
        self.radius = radius
        self.nodes = [(_, np.random.uniform(0, 1, d)) for _ in range(self.n)]
        self.edges = []
        self.adjacency = [[] for _ in range(self.n)]

    @staticmethod
    def find_max_list(lst):
        max_list = max(lst, key=lambda i: len(i))
        return max_list

    def make_edges_minkowski(self):
        for (node1, pos1), (node2, pos2) in combinations(self.nodes, 2):
            dt2 = (pos2[0] - pos1[0])**2
            dx2 = sum([(pos2[i] - pos1[i])**2 for i in range(1, self.d)])
            interval = -dt2 + dx2
            radius = dt2 + dx2
            if interval <= 0 and radius <= self.radius**2:
                edge = (node1, node2) if pos1[0] < pos2[0] else (node2, node1)
                self.edges.append(edge)
                self.adjacency[edge[0]].append(edge[1])

    def longest_path(self):
        chains = [set() for _ in range(self.n)]
        vis = [False] * self.n
        for node in range(self.n):
            if not vis[node]:
                self.longest_path_per_node(node, chains, vis)

        return self.find_max_list(chains)

    def longest_path_per_node(self, node, chains, vis):
        vis[node] = True

        for child in self.adjacency[node]:
            if not vis[child]:
                self.longest_path_per_node(child, chains, vis)

            chains[node] = self.find_max_list([chains[node], {(node, child)}.union(chains[child])])


def rgg():
    return nx.random_geometric_graph(100, 0.5)


if __name__ == '__main__':
    # graph = rgg()
    n = 10
    graph = Graph(n, 0.7, 2)
    graph.make_edges_minkowski()
    print(graph.longest_path())
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(graph.edges)
    nx.draw(g, [(n[1][1], n[1][0]) for n in graph.nodes], with_labels=True)
    plt.show()

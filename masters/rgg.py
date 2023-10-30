import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from itertools import combinations

class Node():
    def __init__(self, pos, targets):
        pass


class Graph():
    def __init__(self, n, radius, d):
        self.n = n
        self.d = d
        self.radius = radius
        self.nodes = [_ for _ in range(n)]
        self.nodes = [(_, np.random.uniform(0, 1, d)) for _ in range(self.n)]
        self.edges = []

    def make_edges_minkowski(self):
        for (node1, pos1), (node2, pos2) in combinations(self.nodes, 2):
            dt2 = (pos2[0] - pos1[0])**2
            dx2 = sum([(pos2[i] - pos1[i])**2 for i in range(1, self.d)])
            interval = -dt2 + dx2
            radius = dt2 + dx2
            if interval <= 0 and radius <= self.radius**2:
                edge = (node1, node2) if pos1[0] < pos2[0] else (node2, node1)
                self.edges.append(edge)


def rgg():
    return nx.random_geometric_graph(100, 0.5)


if __name__ == '__main__':
    # graph = rgg()
    n = 200
    graph = Graph(n, 2, 2)
    graph.make_edges_minkowski()
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(graph.edges)
    nx.draw(g, [(n[1][1], n[1][0]) for n in graph.nodes])
    plt.show()
    print("hi")
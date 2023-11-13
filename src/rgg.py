import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import matplotlib
from dataclasses import dataclass

matplotlib.use("TkAgg")


@dataclass
class Node:
    node: int
    position: np.ndarray


@dataclass
class Order:
    node: int
    height: int
    depth: int


@dataclass
class Neighbours:
    node: int
    children: list
    parents: list


class Graph:
    def __init__(self, n, radius, d):
        """
        Instantiate a time-ordered list of nodes as well as other information
        required to generate an RGG graph in minkowski space time

        Args:
            n: Number of nodes
            radius: Proper time radius that defines threshold of node connection
            d: Number of dimensions (including time)
        """
        self.n = n
        self.d = d
        self.radius = radius
        self.edges = []
        self.longest_path = []
        self.neighbours = [Neighbours(_, [], []) for _ in range(self.n)]
        positions = [np.random.uniform(0, 1, d) for _ in range(self.n)]
        positions = sorted(positions, key=lambda pos: pos[0])
        self.nodes = [Node(node, pos) for node, pos in enumerate(positions)]
        self.height_and_depth = [Order(node, 0, 0) for node in range(self.n)]

    @staticmethod
    def find_max_list(lst):
        """
        Finds longest element in list of lists
        Args:
            lst: some iterable variable, list, set, etc
        """
        max_list = max(lst, key=lambda i: len(i))
        return max_list

    def node_position(self, index):
        """
        Find the position of a node given the index
        """
        return self.nodes[index].position

    @property
    def minkowski_metric(self):
        identity = np.identity(n=self.d)
        identity[0][0] *= -1
        return identity

    def make_edges_minkowski(self):
        """
        Generate edges if two nodes are within self.radius of each other
        and are time-like separated
        """
        for node1, node2 in combinations(self.nodes, 2):
            interval = self.interval((node1, node2))

            if -self.radius * self.radius < interval < 0:
                edge = (
                    (node1.node, node2.node)
                    if node1.position[0] < node2.position[0]
                    else (node2.node, node1.node)
                )
                self.edges.append(edge)
                self.neighbours[edge[0]].children.append(edge[1])
                self.neighbours[edge[1]].parents.append(edge[0])

    def interval(self, node_pair):
        """
        ds^2 between two nodes
        Args:
            node_pair: either the index or the node itself
        """
        if all([isinstance(node, int) for node in node_pair]):
            pos0 = self.node_position(node_pair[0])
            pos1 = self.node_position(node_pair[1])
        elif all([isinstance(node, Node) for node in node_pair]):
            pos0 = node_pair[0].position
            pos1 = node_pair[1].position
        else:
            raise AttributeError("wrong format of node passed to distance function")
        dx = pos1 - pos0

        return self.minkowski_metric @ dx @ dx

    def proper_time(self, node_pair):
        return np.sqrt(-self.interval(node_pair))

    def geometric_length(self, path=None):
        """
        Find the geometric length of a path
        Args:
            path: a list of tuples of node pairs forming the
            edges in this path
        """
        if path is None:
            path = self.longest_path

        tot_distance = 0
        for node_pair in path:
            tot_distance += self.proper_time(node_pair)

        return tot_distance

    def find_longest_path(self):
        """
        Find a longest path in the network. Note that this will return
        only one possible longest path. The nature of which is chosen
        is decided by python's max function.
        """
        chains = [set() for _ in range(self.n)]
        vis = [False] * self.n
        for node in range(self.n):
            if not vis[node]:
                self.longest_path_per_node(node, chains, vis)

        self.longest_path = list(self.find_max_list(chains))

    def longest_path_per_node(self, node, chains, vis):
        """
        iterate through all nested children, considering them
        only if they have not been previously visited. Choose longest
        between current longest path to node and path to current
        child plus edge from child to node.

        Args:
            node: the parent node we are considering
            chains: the longest known chains of all nodes
            vis: which nodes have already been visited
        """
        vis[node] = True

        for child in self.neighbours[node].children:
            if not vis[child]:
                self.longest_path_per_node(child, chains, vis)

            chains[node] = self.find_max_list(
                [chains[node], {(node, child)}.union(chains[child])]
            )
            self.height_and_depth[node].height = len(chains[node])

    def angular_deviation(self, path=None):
        """
        Find the total angular deviation along a path
        Args:
            path: a list of edges included in the path
        """
        if path is None:
            path = self.longest_path
        ordered_path = sorted(path, key=lambda e: e[0])
        angle_index_combos = [
            (i, j) for i, j in combinations([_ for _ in range(self.d)], 2)
        ]

        angles = []
        for node_pair in ordered_path:
            position_changes = [
                self.node_position(node_pair[1])[d]
                - self.node_position(node_pair[0])[d]
                for d in range(self.d)
            ]
            angles.append(
                [
                    np.arctan(position_changes[i] / position_changes[j])
                    for i, j in angle_index_combos
                ]
            )

        print(angles)
        deviation = []
        for i, j in zip(angles[0::], angles[1::]):
            deviation.append([j[d] - i[d] for d in range(len(i))])
        return deviation


if __name__ == "__main__":
    n = 10
    graph = Graph(n, 0.3, 2)
    graph.make_edges_minkowski()
    graph.find_longest_path()
    print(graph.geometric_length())
    print(graph.longest_path)
    print(graph.angular_deviation())
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(graph.edges)
    nx.draw(g, [(n.position[1], n.position[0]) for n in graph.nodes], with_labels=True)
    plt.show()

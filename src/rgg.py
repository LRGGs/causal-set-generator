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
        self.nodes = []
        self.order = [Order(node, 0, 0) for node in range(self.n)]

    def generate_nodes(self):
        positions = [np.random.uniform(0, 1, self.d) for _ in range(self.n - 2)]
        rot = 1/np.sqrt(2)
        rotation_mat = np.array([[rot, rot],
                                 [-rot, rot]])
        positions = [rotation_mat @ p for p in positions]
        positions.append([0, 0])
        positions.append([np.sqrt(2), 0])
        positions = sorted(positions, key=lambda pos: pos[0])
        self.nodes = [Node(node, pos) for node, pos in enumerate(positions)]

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
        dx = np.array(pos1) - np.array(pos0)

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

    def find_order(self):
        """"""
        possible_directions = ["children", "parents"]
        direction_to_start_map = {"children": 0, "parents": self.n -1}
        for direction in possible_directions:
            vis = [False] * self.n
            node = direction_to_start_map[direction]
            self.direction_first_search(node, vis, direction)

    def direction_first_search(self, node, vis, direction):
        """
        iterate through all nested children, considering them
        only if they have not been previously visited. Choose longest
        between current longest path to node and path to current
        child plus edge from child to node.

        Args:
            node: the parent node we are considering
            vis: which nodes have already been visited
            direction:
        """
        direction_to_order_map = {"children": "depth", "parents": "height"}
        vis[node] = True

        for relative in getattr(self.neighbours[node], direction):
            if not vis[relative]:
                self.direction_first_search(relative, vis, direction)

            current_order = getattr(self.order[node], direction_to_order_map[direction])
            child_order = getattr(self.order[relative], direction_to_order_map[direction])

            setattr(self.order[node], direction_to_order_map[direction], max([current_order, child_order + 1]))

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
    n = 100
    graph = Graph(n, 0.3, 2)
    graph.generate_nodes()
    graph.make_edges_minkowski()
    graph.find_order()
    print(graph.order)
    # print(graph.geometric_length())
    # print(graph.longest_path)
    # print(graph.angular_deviation())
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(graph.edges)
    nx.draw(g, [(n.position[1], n.position[0]) for n in graph.nodes], with_labels=True)
    plt.show()

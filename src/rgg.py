import multiprocessing
import random
from dataclasses import dataclass
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

matplotlib.use("TkAgg")


@dataclass
class Node:
    ind: int
    position: np.ndarray


@dataclass
class Order:
    node: int
    height: int
    depth: int


@dataclass
class Relatives:
    node: int
    children: list
    parents: list


@dataclass
class Paths:
    longest: list
    greedy: list
    random: list
    shortest: list


def multi_edge(node_pairs, r_squared, metric):
    edges = []

    for node1, node2 in node_pairs:
        pos0 = node1.position
        pos1 = node2.position
        dx = pos1 - pos0

        separation = dx @ metric @ dx

        if -r_squared < separation < 0:
            edges.append((node1.ind, node2.ind))
    return edges


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
        self.relatives = [Relatives(_, [], []) for _ in range(self.n)]
        self.nodes = []
        self.order = [Order(node, 0, 0) for node in range(self.n)]
        identity = np.identity(n=self.d)
        identity[0][0] *= -1
        self.minkowski_metric = identity
        self.paths = Paths([], [], [], [])
        self.connected_interval = []

    def configure_graph(self):
        self.generate_nodes()
        self.make_edges_minkowski_multi()
        self.find_order()
        self.find_valid_interval()

    def find_paths(self):
        self.longest_path()
        self.shortest_path()
        self.random_path()
        self.greedy_path()

    def generate_nodes(self):
        positions = [np.random.uniform(0, 0.5, self.d) for _ in range(self.n - 2)]
        rotation_mat = np.array([[1, 1], [-1, 1]])
        positions.append(np.array([0, 0]))
        positions.append(np.array([0.5, 0.5]))
        positions = [rotation_mat @ p for p in positions]
        positions = sorted(positions, key=lambda pos: pos[0])
        self.nodes = [Node(node, pos) for node, pos in enumerate(positions)]

    def node_position(self, index):
        """
        Find the position of a node given the index
        """
        return self.nodes[index].position

    def make_edges_minkowski_multi(self):
        """
        Generate edges if two nodes are within self.radius of each other
        and are time-like separated. S
        """
        node_pairs = [(node1, node2) for node1, node2 in combinations(self.nodes, 2)]

        cpus = multiprocessing.cpu_count() - 1
        p = multiprocessing.Pool(processes=cpus)

        pair_lists = [node_pairs[i : i + cpus] for i in range(0, len(node_pairs), cpus)]
        radius_squared = self.radius**2

        inputs = [
            [pairs, radius_squared, self.minkowski_metric] for pairs in pair_lists
        ]
        results = p.starmap(multi_edge, inputs)
        for edges in results:
            for edge in edges:
                self.edges.append(edge)
                self.relatives[edge[0]].children.append(edge[1])
                self.relatives[edge[1]].parents.append(edge[0])

    def find_valid_interval(self):
        """
        Find all nodes that are not a source or sink apart from the
        main source and sink
        """
        for node in self.nodes:
            order = self.order[node.ind]
            if (
                order.height != 0 and order.depth != 0
            ) or (node.ind == 0 or node.ind == self.n - 1):
                self.connected_interval.append(node.ind)

    def interval(self, node_pair):
        """
        ds^2 between two nodes
        Args:
            node_pair: the index of each node in a tuple
        """
        pos0 = self.node_position(node_pair[0])
        pos1 = self.node_position(node_pair[1])

        dx = pos1 - pos0

        return dx @ self.minkowski_metric @ dx

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
            path = self.paths.longest

        tot_distance = 0
        for node_pair in path:
            tot_distance += self.proper_time(node_pair)

        return tot_distance

    def find_order(self):
        """"""
        possible_directions = ["children", "parents"]
        direction_to_start_map = {"children": 0, "parents": self.n - 1}
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
            direction: either look through parents or children
        """
        direction_to_order_map = {"children": "depth", "parents": "height"}
        vis[node] = True

        for relative in getattr(self.relatives[node], direction):
            if not vis[relative]:
                self.direction_first_search(relative, vis, direction)

            current_order = getattr(self.order[node], direction_to_order_map[direction])
            relative_order = getattr(
                self.order[relative], direction_to_order_map[direction]
            )

            setattr(
                self.order[node],
                direction_to_order_map[direction],
                max([current_order, relative_order + 1]),
            )

    def longest_path(self):
        """
        Run this only after finding the order. Randomly choose
        a valid sequential path from nodes with the highest order.
        Returns:
            longest path list of edges
        """
        path = []
        node = self.nodes[0]
        while node != self.nodes[-1]:
            current_depth = self.order[node.ind].depth
            valid_children = [
                child
                for child in self.relatives[node.ind].children
                if child in self.connected_interval and self.order[child].depth == current_depth - 1
            ]
            next_node = random.choice(valid_children)
            path.append((node.ind, next_node))
            node = self.nodes[next_node]
        self.paths.longest = path

    def shortest_path(self):
        """
        Run this only after finding the order. Randomly choose
        a valid sequential path from nodes with the lowest order.
        Returns:
            shortest path list of edges
        """
        path = []
        node = self.nodes[0]
        while node != self.nodes[-1]:
            children = [child for child in self.relatives[node.ind].children if child in self.connected_interval]
            min_depth = min([self.order[child].depth for child in children])
            valid_children = [
                child for child in children if self.order[child].depth == min_depth
            ]
            next_node = random.choice(valid_children)
            path.append((node.ind, next_node))
            node = self.nodes[next_node]
        self.paths.shortest = path

    def random_path(self):
        """
        Run this only after finding the order. Randomly choose
        a valid sequential path.
        Returns:
            random path list of edges
        """
        path = []
        node = self.nodes[0]
        while node != self.nodes[-1]:
            valid_children = [
                child
                for child in self.relatives[node.ind].children
                if child in self.connected_interval and (self.order[child].depth != 0 or self.nodes[child] == self.nodes[-1])
            ]
            next_node = random.choice(valid_children)
            path.append((node.ind, next_node))
            node = self.nodes[next_node]
        self.paths.random = path

    def greedy_path(self):
        """
        Run this only after finding the order. Choose the next
        node at each step as the node with the largest interval.
        Returns:
            greedy path list of edges
        """
        path = []
        node = self.nodes[0]
        while node != self.nodes[-1]:
            child_intervals = [
                (child, self.interval((node.ind, child)))
                for child in self.relatives[node.ind].children if child in self.connected_interval
            ]
            next_node = max(child_intervals, key=lambda l: l[1])[0]
            path.append((node.ind, next_node))
            node = self.nodes[next_node]
        self.paths.greedy = path


def run():
    n = 3000
    graph = Graph(n, 0.3, 2)
    graph.configure_graph()
    graph.find_paths()

    print(graph.paths.longest)
    print(graph.paths.shortest)
    print(graph.paths.random)
    print(graph.paths.greedy)
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(graph.edges)
    nx.draw(g, [(n.position[1], n.position[0]) for n in graph.nodes], with_labels=True)
    plt.show()


if __name__ == "__main__":
    import cProfile
    import io
    import pstats

    pr = cProfile.Profile()
    pr.enable()

    run()

    filename = "profile.prof"  # You can change this if needed
    pr.dump_stats(filename)

    # cProfile.run("run()", "profiler")
    # pstats.Stats("profiler").strip_dirs().sort_stats("tottime").print_stats()

import multiprocessing
import random
from dataclasses import dataclass
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from numba.typed import List

matplotlib.use("TkAgg")


@dataclass
class Node:
    indx: int
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


def multi_edge(nodes, r_squared, metric, start):
    numba_nodes = List()
    [numba_nodes.append(np.array(node.position, dtype=np.float32)) for node in nodes]
    return list(numba_edge(numba_nodes, r_squared, metric, start))


@njit()
def numba_edge(nodes, r2, metric, start):
    edges = List()
    for i in range(start, len(nodes)):
        for j in range(i+1, len(nodes)):
            pos1 = nodes[i]
            pos2 = nodes[j]
            dx = pos2 - pos1
            interval = dx @ metric @ dx
            if -r2 < interval < 0:
                edges.append([i, j])
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
        self.numba_nodes = List()
        self.order = [Order(node, 0, 0) for node in range(self.n)]
        identity = np.identity(n=self.d, dtype=np.float32)
        identity[0][0] *= -1
        self.minkowski_metric = identity
        self.paths = Paths([], [], [], [])
        self.connected_interval = []

    def configure_graph(self):
        self.generate_nodes()
        self.make_edges_minkowski_numba()
        # self.find_order()
        # self.find_valid_interval()

    def find_paths(self):
        self.longest_path()
        self.shortest_path()
        self.random_path()
        self.greedy_path()

    @property
    def node_x_positions(self):
        return [node.position[1] for node in self.nodes]

    @property
    def node_t_positions(self):
        return [node.position[0] for node in self.nodes]

    def generate_nodes(self):
        positions = [np.random.uniform(0, 0.5, self.d) for _ in range(self.n - 2)]
        rotation_mat = np.array([[1, 1], [-1, 1]])
        positions.append(np.array([0, 0]))
        positions.append(np.array([0.5, 0.5]))
        positions = [rotation_mat @ p for p in positions]
        positions = sorted(positions, key=lambda pos: pos[0])
        self.nodes = [Node(node, pos) for node, pos in enumerate(positions)]
        [self.numba_nodes.append(np.array(pos, dtype=np.float32)) for pos in positions]

    def node_position(self, index):
        """
        Find the position of a node given the index
        """
        return self.nodes[index].position

    def make_edges_minkowski_numba(self):
        """
        Generate edges if two nodes are within self.radius of each other hnbc  cfvgbncnfvgb
        and are time-like separated
        """
        edges = numba_edge(self.numba_nodes, self.radius**2, self.minkowski_metric, start=0)
        for edge in edges:
            self.relatives[edge[0]].children.append(edge[1])
            self.relatives[edge[1]].parents.append(edge[0])

    def make_edges_minkowski_multi(self):
        """
        Generate edges if two nodes are within self.radius of each other
        and are time-like separated.
        """
        cpus = multiprocessing.cpu_count() - 1
        starts = list(range(self.n))[0::self.n//(cpus-1)]
        p = multiprocessing.Pool(processes=cpus)

        radius_squared = self.radius**2

        inputs = [
            [self.nodes, radius_squared, self.minkowski_metric, start] for start in starts
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
            order = self.order[node.indx]
            if (order.height != 0 and order.depth != 0) or (
                node.indx == 0 or node.indx == self.n - 1
            ):
                self.connected_interval.append(node.indx)

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
            current_depth = self.order[node.indx].depth
            valid_children = [
                child
                for child in self.relatives[node.indx].children
                if child in self.connected_interval
                and self.order[child].depth == current_depth - 1
            ]
            next_node = random.choice(valid_children)
            path.append((node.indx, next_node))
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
            children = [
                child
                for child in self.relatives[node.indx].children
                if child in self.connected_interval
            ]
            min_depth = min([self.order[child].depth for child in children])
            valid_children = [
                child for child in children if self.order[child].depth == min_depth
            ]
            next_node = random.choice(valid_children)
            path.append((node.indx, next_node))
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
                for child in self.relatives[node.indx].children
                if child in self.connected_interval
                and (
                    self.order[child].depth != 0 or self.nodes[child] == self.nodes[-1]
                )
            ]
            next_node = random.choice(valid_children)
            path.append((node.indx, next_node))
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
                (child, self.interval((node.indx, child)))
                for child in self.relatives[node.indx].children
                if child in self.connected_interval
            ]
            next_node = max(child_intervals, key=lambda l: l[1])[0]
            path.append((node.indx, next_node))
            node = self.nodes[next_node]
        self.paths.greedy = path

    def path_positions(self, path="longest"):
        """
        return all the node positions along a chosen path
        Args:
            path: choice between "longest", "shortest", "random", and "greedy"
        """
        path = getattr(self.paths, path)
        node_positions = [self.nodes[edge[0]].position for edge in path]
        node_positions.append(self.nodes[path[-1][1]].position)
        return np.array(node_positions)

    def plot_nodes(self):
        """
        Plot all nodes on a pyplot plot without showing it yet
        """
        plt.plot(self.node_x_positions, self.node_t_positions, "g,")


def run():
    n = 10000
    graph = Graph(n, 0.3, 2)
    graph.configure_graph()
    # graph.find_paths()

    # print(graph.paths.longest)
    # print(graph.paths.shortest)
    # print(graph.paths.random)
    # print(graph.paths.greedy)

    # graph.plot_nodes()
    # for i in ["longest", "shortest", "random", "greedy"]:
    #     path = graph.path_positions(i)
    #     plt.plot(path[:, 1], path[:, 0], "o", label=i)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    import cProfile
    import io
    import pstats

    # pr = cProfile.Profile()
    # pr.enable()

    # run()

    # filename = "profile.prof"  # You can change this if needed
    # pr.dump_stats(filename)

    cProfile.run("run()", "profiler")
    pstats.Stats("profiler").strip_dirs().sort_stats("tottime").print_stats()

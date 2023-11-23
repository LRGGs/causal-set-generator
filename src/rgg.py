import multiprocessing
import random
import time
from dataclasses import dataclass
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numba.np.arraymath
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
    children: list
    parents: list


@dataclass
class Paths:
    longest: list
    greedy: list
    random: list
    shortest: list


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
        self.relatives = Relatives([], [])
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
        self.find_order()
        self.find_valid_interval()

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
        Generate edges if two nodes are within self.radius of each other
        and are time-like separated
        """
        a = time.time()
        self.edges, children, parents = self.numba_edges(self.numba_nodes, self.radius, self.minkowski_metric)
        self.relatives.children = list(children)
        self.relatives.parents = list(parents)
        print(time.time() - a)

    @staticmethod
    @njit()
    def numba_edges(nodes, r, metric):
        r2 = r * r
        edges = List()
        n = len(nodes)
        children = List()
        [children.append(List.empty_list(numba.int64)) for _ in range(n)]
        parents = List()
        [parents.append(List.empty_list(numba.int64)) for _ in range(n)]

        for i in range(n):
            node1 = nodes[i]
            tmax = 0.5 * (1 + r + node1[0] - node1[1])
            l1 = (r + node1[0] - node1[1])
            l2 = (r + node1[0] + node1[1])
            for j in range(i + 1, n):
                node2 = nodes[j]
                if node2[0] > tmax:
                    break
                if node2[0] - node2[1] > l1 and node2[0] + node2[1] > l2:
                    continue
                pos1 = node1
                pos2 = node2
                dx = pos2 - pos1
                interval = dx @ metric @ dx
                if -r2 < interval < 0:
                    edges.append([i, j])
                    children[i].append(j)
                    parents[j].append(i)

        new_children = List.empty_list(numba.int64[:])
        new_parents = List.empty_list(numba.int64[:])

        for i in range(n):
            kids = children[i]
            kidz = len(kids)
            a = np.zeros(kidz, dtype=np.int64)
            for j in range(kidz):
                a[j] = kids[j]
            new_children.append(a)

            pars = parents[i]
            parz = len(pars)
            a = np.zeros(parz, dtype=np.int64)
            for j in range(parz):
                a[j] = pars[j]
            new_parents.append(a)

        return edges, new_children, new_parents

    def find_valid_interval(self):
        """
        Find all nodes that are not a source or sink apart from the
        main source and sink
        """

        self.connected_interval.append(self.nodes[0].indx)
        for node in self.nodes:
            order = self.order[node.indx]
            if order.height != 0 and order.depth != 0:
                self.connected_interval.append(node.indx)
        self.connected_interval.append(self.nodes[-1].indx)


    def interval(self, node_pair):
        """
        ds^2 between two nodes
        Args:
            node_pair: the index of each node in a tuple
        """
        pos0 = self.node_position(node_pair[0])
        pos1 = self.node_position(node_pair[1])

        dx = pos1 - pos0

        return self.minkowski_metric @ dx @ dx

    def proper_time(self, node_pair):
        return np.sqrt(-self.interval(node_pair))

    def geometric_length(self, path="longest"):
        """
        Find the geometric length of a path
        Args:
            path: a list of tuples of node pairs forming the
            edges in this path
        """
        path = getattr(self.paths, path)

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

        for relative in getattr(self.relatives, direction)[node]:
            relative = int(relative)
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
                for child in self.relatives.children[node.indx]
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
                for child in self.relatives.children[node.indx]
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
                for child in self.relatives.children[node.indx]
                if child in self.connected_interval
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
                for child in self.relatives.children[node.indx]
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

    def weight_collections(self):
        tot_orders = [self.order[i].depth + self.order[i].height
                     for i in self.connected_interval]
        max_ord = max(tot_orders)
        tot_orders = np.array(tot_orders)
        weights = max_ord - tot_orders
        max_weight, min_weight = max(weights), min(weights)

        all_posses = np.array([self.nodes[node].position
                               for node in self.connected_interval])
        collections = []
        for weight in range(min_weight, max_weight):
            collections.append(all_posses[weights == weight])

        return collections

    def pickle(self):
        info = {
            "nodes": self.nodes,
            "order": self.order,
            "order_collections": self.order_collections(),
            "paths": self.paths,
        }
        return pickle.dumps(info)


def run(n, r, d):
    graph = Graph(n, r, d)
    graph.configure_graph()
    graph.find_paths()

    # print(graph.paths.longest)
    # print(graph.paths.shortest)
    # print(graph.paths.random)
    # print(graph.paths.greedy)

    graph.plot_nodes()
    for i in ["longest", "shortest", "random", "greedy"]:
        path = graph.path_positions(i)
        plt.plot(path[:, 1], path[:, 0], "o", label=i)
    plt.legend()
    plt.show()

    return graph.pickle()


def multi_run(n, r, d, iters):
    cpus = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(processes=cpus)

    inputs = [
        [n, r, d] for _ in range(iters)
    ]
    result = p.starmap(run, inputs)
    # for res in result:
    #     print(pickle.loads(res)["paths"].longest)


if __name__ == "__main__":
    import cProfile
    import io
    import pstats

    # pr = cProfile.Profile()
    # pr.enable()

    # run()

    # filename = "profile.prof"  # You can change this if needed
    # pr.dump_stats(filename)

    cProfile.run("run(10000, 0.3, 2)", "profiler")
    pstats.Stats("profiler").strip_dirs().sort_stats("tottime").print_stats()

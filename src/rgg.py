import multiprocessing
import pickle
import random
import time
from itertools import product

# import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numba.np.arraymath
import numpy as np
from numba import njit
from numba.typed import List

from src.mlogging.handler import update_status
from src.utils import (Node, Order, Paths, Relatives, bcolors, file_namer,
                       nrange)

# matplotlib.use("TkAgg")


class Graph:
    def __init__(self, n, radius, d):
        """
        Instantiate a time-ordered list of nodes as well as other information
        required to generate an RGG graph in minkowski space-time

        Args:
            n: Number of nodes
            radius: Proper time radius that defines threshold of node connection
            d: Number of dimensions (including time)
        """
        self.n = n
        self.d = d
        self.radius = radius
        self.edges = []
        self.relatives = []
        self.nodes = []
        self.numba_nodes = List()
        self.orders = [Order(node, 0, 0) for node in range(self.n)]
        identity = np.identity(n=self.d, dtype=np.float32)
        identity[0][0] *= -1
        self.minkowski_metric = identity
        self.paths = Paths([], [], [], [])
        self.connected_interval = []

    def configure_graph(self, timing=False):
        a = time.time()
        self.generate_nodes()
        b = time.time()
        self.make_edges_minkowski_numba()
        c = time.time()
        self.find_order()
        d = time.time()
        self.find_valid_interval()
        e = time.time()
        if timing:
            print(f"gen nodes: {b - a}")
            print(f"make edges: {c - b}")
            print(f"find order: {d - c}")
            print(f"find interval: {e - d}")

    def find_paths(self, timing=False):
        a = time.time()
        self.longest_path()
        b = time.time()
        self.shortest_path()
        c = time.time()
        self.random_path()
        d = time.time()
        self.greedy_path()
        e = time.time()
        if timing:
            print(f"longest: {b - a}")
            print(f"shortest: {c - b}")
            print(f"random: {d - c}")
            print(f"greedy: {e - d}")

    @property
    def node_x_positions(self):
        return [node.position[1] for node in self.nodes]

    @property
    def node_t_positions(self):
        return [node.position[0] for node in self.nodes]

    def generate_nodes(self):
        # np.random.seed(1)
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
        self.edges, children, parents = self.numba_edges(
            self.numba_nodes, self.radius, self.minkowski_metric
        )
        b = time.time()
        children = list(children)
        parents = list(parents)
        for i in range(len(children)):
            self.relatives.append(Relatives(i, list(children[i]), list(parents[i])))
        c = time.time()
        # print(f"numba: {b-a}")
        # print(f"loop: {c-b}")

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

        for i in range(len(nodes)):
            node1 = nodes[i]
            tmax = max(
                [
                    0.5 * (1 + r + node1[0] - node1[1]),
                    0.5 * (1 + r + node1[0] + node1[1]),
                ]
            )
            l1 = r + node1[0] - node1[1]
            l2 = r + node1[0] + node1[1]
            for j in range(i + 1, n):
                node2 = nodes[j]
                if node2[0] > tmax:
                    break
                if node2[0] - node2[1] > l1 and node2[0] + node2[1] > l2:
                    continue
                dx = node2 - node1
                interval = dx @ metric @ dx
                if -r2 < interval < 0:
                    edges.append([i, j])
                    children[i].append(j)
                    parents[j].append(i)

        new_children = List.empty_list(numba.int32[:])
        new_parents = List.empty_list(numba.int32[:])

        for i in range(n):
            a = np.zeros(len(children[i]), dtype=np.int32)
            for j in range(len(children[i])):
                a[j] = children[i][j]
            new_children.append(a)

            a = np.zeros(len(parents[i]), dtype=np.int32)
            for j in range(len(parents[i])):
                a[j] = parents[i][j]
            new_parents.append(a)

        return edges, new_children, new_parents

    def find_valid_interval(self):
        """
        Find all nodes that are not a source or sink apart from the
        main source and sink
        """

        self.connected_interval.append(self.nodes[0].indx)
        for node in self.nodes:
            order = self.orders[node.indx]
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

    def separation(self, node_pair):
        """
        Euclidean distance between two nodes
        Args:
            node_pair: the index of each node in a tuple
        """
        pos0 = self.node_position(node_pair[0])
        pos1 = self.node_position(node_pair[1])

        dx = pos1 - pos0

        return np.sqrt(dx @ dx)

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

        for relative in getattr(self.relatives[node], direction):
            relative = int(relative)
            if not vis[relative]:
                self.direction_first_search(relative, vis, direction)

            current_order = getattr(
                self.orders[node], direction_to_order_map[direction]
            )
            relative_order = getattr(
                self.orders[relative], direction_to_order_map[direction]
            )

            setattr(
                self.orders[node],
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
            current_depth = self.orders[node.indx].depth
            valid_children = [
                int(child)
                for child in self.relatives[node.indx].children
                if int(child) in self.connected_interval
                and self.orders[int(child)].depth == current_depth - 1
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
                int(child)
                for child in self.relatives[node.indx].children
                if int(child) in self.connected_interval
            ]
            min_depth = min([self.orders[child].depth for child in children])
            valid_children = [
                child for child in children if self.orders[child].depth == min_depth
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
                int(child)
                for child in self.relatives[node.indx].children
                if int(child) in self.connected_interval
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
                (int(child), self.separation((node.indx, int(child))))
                for child in self.relatives[node.indx].children
                if int(child) in self.connected_interval
            ]
            next_node = min(child_intervals, key=lambda l: l[1])[0]
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
        tot_orders = [
            self.orders[i].depth + self.orders[i].height
            for i in self.connected_interval
        ]
        max_ord = max(tot_orders)
        tot_orders = np.array(tot_orders)
        weights = max_ord - tot_orders
        max_weight, min_weight = max(weights), min(weights)

        all_posses = np.array(
            [self.nodes[node].position for node in self.connected_interval]
        )
        collections = []
        for weight in range(min_weight, max_weight):
            collections.append(all_posses[weights == weight])

        return collections

    def to_dict(self):
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "order": [order.to_dict() for order in self.orders],
            "order_collections": self.weight_collections(),
            "paths": self.paths.to_dict(),
            "interval": self.connected_interval,
        }


def run(n, r, d, i=1, p=False, g=False, m=False):
    graph = Graph(n, r, d)
    print(f"{bcolors.WARNING} Graph {i}: INSTANTIATED {bcolors.ENDC}")
    update_status(i + 1, "yellow")
    graph.configure_graph()
    print(f"{bcolors.OKBLUE} Graph {i}: CONFIGURED {bcolors.ENDC}")
    update_status(i + 1, "blue")
    graph.find_paths()
    print(f"{bcolors.OKGREEN} Graph {i}: PATHED {bcolors.ENDC}")
    update_status(i + 1, "green")

    if p:
        print(graph.paths.longest)
        print(graph.paths.shortest)
        print(graph.paths.random)
        print(graph.paths.greedy)

    if g:
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from(graph.edges)
        nx.draw(
            g, [(n.position[1], n.position[0]) for n in graph.nodes], with_labels=True
        )
        plt.savefig("../images/network")
        plt.clf()

    if m:
        graph.plot_nodes()
        for i in ["longest", "shortest", "random", "greedy"]:
            path = graph.path_positions(i)
            plt.plot(path[:, 1], path[:, 0], "o", label=i)
        plt.legend()
        plt.show()

    return graph.to_dict()


def multi_run(n, r, d, iters):
    cpus = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(processes=cpus)
    variables = [n, r, d]
    if any(isinstance(i, list) for i in variables):
        iters = 1
        variables = [[i] if not isinstance(i, list) else i for i in variables]
        variables = [list(i) for i in product(*variables)]
        inputs = [[*j, i] for i, j in enumerate(variables)]
    else:
        inputs = [[n, r, d, i] for i in range(iters)]

    result = p.starmap(run, inputs)

    with open(
        file_namer(n, r, d, iters),
        "wb",
    ) as fp:
        pickle.dump(result, fp)


def main():
    # import cProfile
    # import pstats
    # cProfile.run("run(20, 1, 2, i=1, p=False, m=False, g=False)", "profiler")
    # pstats.Stats("profiler").strip_dirs().sort_stats("tottime").print_stats()

    multi_run(nrange(1000, 10000, 100), 0.1, 2, 1)

    # multi_run(20, 0.5, 2, 10)


if __name__ == "__main__":
    main()

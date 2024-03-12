import itertools
import multiprocessing
import os
import pickle
import random
import time
from itertools import product, repeat
from collections import defaultdict

# import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numba.np.arraymath
import numpy as np
from numba import njit
from numba.typed import List
import pandas as pd

from analysis.an_utils import PATH_NAMES
from mlogging.handler import update_status
from utils import *


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
        self.paths = Paths([], [], [], [], [], [])
        self.connected_interval = []

    def configure_graph(self, seed=None, timing=False):
        a = time.time()
        self.generate_nodes(seed)
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
        self.greedy_path_euc()
        e = time.time()
        self.greedy_path_min()
        f = time.time()
        # self.greedy_path_opt()
        g = time.time()
        if timing:
            print(f"longest: {b - a}")
            print(f"shortest: {c - b}")
            print(f"random: {d - c}")
            print(f"greedy: {e - d}")
            print(f"minkovski: {f - e}")
            print(f"optimum: {g - f}")

    @property
    def node_x_positions(self):
        return [node.position[1] for node in self.nodes]

    @property
    def node_t_positions(self):
        return [node.position[0] for node in self.nodes]

    def generate_nodes(self, seed=None):
        if seed:
            np.random.seed(seed)
        positions = []

        while len(positions) < self.n-2:
            new_pos = np.array([np.random.uniform(0, 1), *np.random.uniform(-0.5, 0.5, self.d - 1)])
            if self.interval((np.zeros(self.d), new_pos)) <= 0 and self.interval((np.array([1, *[0]*(self.d-1)]), new_pos)) <= 0:
                positions.append(np.array(new_pos))

        positions.append(np.zeros(self.d))
        positions.append((np.array([1, *[0]*(self.d-1)])))

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
        self.edges, children, parents = self.numba_edges(
            self.numba_nodes, self.radius, self.minkowski_metric
        )
        children = list(children)
        parents = list(parents)
        for i in range(len(children)):
            self.relatives.append(Relatives(i, list(children[i]), list(parents[i])))

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
            # tmax = max(
            #     [
            #         0.5 * (1 + r + node1[0] - node1[1]),
            #         0.5 * (1 + r + node1[0] + node1[1]),
            #     ]
            # )
            # l1 = r + node1[0] - node1[1]
            # l2 = r + node1[0] + node1[1]
            for j in range(i + 1, n):
                node2 = nodes[j]
                # if node2[0] > tmax:
                #     break
                # if node2[0] - node2[1] > l1 and node2[0] + node2[1] > l2:
                #     continue
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

    def interval(self, node_pair):
        """
        ds^2 between two nodes
        Args:
            node_pair: the index of each node in a tuple
        """
        pos0 = node_pair[0]
        pos1 = node_pair[1]

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
            node_pair = (self.node_position(node_pair[0]), self.node_position(node_pair[1]))
            tot_distance += self.proper_time(node_pair)

        return tot_distance

    def find_order(self):
        """"""
        possible_directions = ["children", "parents"]
        direction_to_start_map = {"children": 0, "parents": self.n - 1}
        for direction in possible_directions:
            vis = [False] * self.n
            true_sink = [False] * self.n
            node = direction_to_start_map[direction]
            self.direction_first_search(node, vis, direction, true_sink)

    def direction_first_search(self, node, vis, direction, true_node):
        """
        iterate through all nested children, considering them
        only if they have not been previously visited. Choose longest
        between current longest path to node and path to current
        child plus edge from child to node.

        Args:
            node: the parent node we are considering
            vis: which nodes have already been visited
            direction: either look through parents or children
            true_node: list of bools for if a node is connected to the true sink/source
        """
        direction_to_order_map = {"children": "depth", "parents": "height"}
        vis[node] = True

        if not getattr(self.relatives[node], direction):
            true_node[node] = node == self.n - 1 or node == 0

        for relative in getattr(self.relatives[node], direction):
            relative = int(relative)

            if not vis[relative]:
                self.direction_first_search(relative, vis, direction, true_node)

            if true_node[relative]:
                true_node[node] = True

            if true_node[node]:
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

    def find_valid_interval(self):
        """
        Find all nodes that are not a source or sink apart from the
        main source and sink
        """

        self.connected_interval.append(self.nodes[0].indx)
        for node in self.nodes:
            order = self.orders[node.indx]
            if order.order != 0:
                self.connected_interval.append(node.indx)
        self.connected_interval.append(self.nodes[-1].indx)

    def longest_path(self):
        """
        Run this only after finding the order. Randomly choose
        a valid sequential path from nodes with the highest order.
        Returns:
            longest path list of edges
        """
        path = []
        node = self.nodes[0]
        max_order = max([order.order for order in self.orders])
        while node != self.nodes[-1]:
            current_depth = self.orders[node.indx].depth
            valid_children = [
                int(child)
                for child in self.relatives[node.indx].children
                if int(child) in self.connected_interval
                and self.orders[int(child)].depth == current_depth - 1
                and self.orders[int(child)].order == max_order
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

    def greedy_path_euc(self):
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
        self.paths.greedy_e = path

    def greedy_path_min(self):
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
                (int(child), self.proper_time((node.position, self.nodes[int(child)].position)))
                for child in self.relatives[node.indx].children
                if int(child) in self.connected_interval
            ]
            next_node = max(child_intervals, key=lambda l: l[1])[0]
            path.append((node.indx, next_node))
            node = self.nodes[next_node]
        self.paths.greedy_m = path

    def greedy_path_opt(self):
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
                (int(child), abs(self.nodes[int(child)].position[1]))
                for child in self.relatives[node.indx].children
                if int(child) in self.connected_interval
            ]
            next_node = min(child_intervals, key=lambda l: l[1])[0]
            path.append((node.indx, next_node))
            node = self.nodes[next_node]
        self.paths.greedy_o = path

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
        tot_orders = [self.orders[i].order for i in self.connected_interval]
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

        cleaned_collections = numpy_to_list(collections)
        return cleaned_collections

    def paths_info(self):
        paths = defaultdict(list)
        for name, path in self.paths.to_dict().items():
            nodes_in_path = sorted(list(set(itertools.chain(*path))))
            paths[name] += [
                [float(pos) for pos in self.nodes[node].position]
                for node in nodes_in_path
            ]
        return paths

    def to_weights_info(self):
        return {
            "n": len(self.nodes),
            "weight_collections": self.weight_collections(),
        }

    def to_paths_info(self, n_named=False):
        return {"n": len(self.nodes) if not n_named else n_named, "paths": self.paths_info()}

    def to_longest_info(self, n_named=False):
        return {"n": len(self.nodes) if not n_named else n_named, "longest": self.paths_info()["longest"]}


def run(n, r, d, seed=None, i=1, p=False, g=False, m=False, j=True, t=False):
    n_named = int(n)
    # n = int(np.random.poisson(n))
    graph = Graph(n, r, d)
    print(f"{bcolors.WARNING} Graph {i}: INSTANTIATED {bcolors.ENDC}")
    # update_status(i + 1, "yellow")
    graph.configure_graph(seed=seed, timing=t)
    print(f"{bcolors.OKBLUE} Graph {i}: CONFIGURED {bcolors.ENDC}")
    # update_status(i + 1, "blue")
    graph.find_paths(timing=t)
    print(f"{bcolors.OKGREEN} Graph {i}: PATHED {bcolors.ENDC}")
    # update_status(i + 1, "green")

    if p:
        print(f"longest ({len(graph.paths.longest)}: {graph.paths.longest}")
        print(f"greedy_e ({len(graph.paths.greedy_e)}: {graph.paths.greedy_e}")
        print(f"greedy_m ({len(graph.paths.greedy_m)}: {graph.paths.greedy_m}")
        print(f"random ({len(graph.paths.random)}: {graph.paths.random}")
        print(f"shortest ({len(graph.paths.shortest)}: {graph.paths.shortest}")

    if g:
        g = nx.DiGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from(graph.edges)
        color_map = []
        longest = sorted(list(set(itertools.chain(*graph.paths.longest))))
        shortest = sorted(list(set(itertools.chain(*graph.paths.shortest))))
        for node in graph.nodes:
            if node.indx in [0, len(graph.nodes) - 1]:
                color_map.append("red")
            elif node.indx in longest:
                color_map.append("green")
            elif node.indx in shortest:
                color_map.append("yellow")
            else:
                color_map.append("cyan")
        nx.draw(
            g,
            [(n.position[1], n.position[0]) for n in graph.nodes],
            node_color=color_map,
            with_labels=True,
        )
        plt.show()

    if m:
        plt.scatter(graph.node_x_positions, graph.node_t_positions, 1, color="green")
        graph.plot_nodes()
        for i in PATH_NAMES:
            path = graph.path_positions(i)
            if i not in ["longest", "greedy_e"]:
                plt.plot(path[:, 1], path[:, 0], "-o", label=label_map[i], color=colour_map[i])
            else:
                plt.plot(path[:, 1], path[:, 0], "-o", label=label_map[i])
        plt.plot([0, 0], [0, 1], color="black")
        plt.xticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="box")
        plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0.05))
        plt.title(f"A Fully Connected and Pathed Graph of {graph.n} Nodes")
        plt.xlabel("X - Spatial Dimension")
        plt.ylabel("T - Temporal Dimension")
        plt.savefig("../images/grap paths.png", transparent=True, dpi=1000, bbox_inches="tight")

    if j:
        thread = multiprocessing.current_process().name
        path = os.getcwd().split("src")[0]
        filename = f"{path}/json_results/temp/{str(thread)}"

        dict_to_save = graph.to_paths_info(n_named=n_named)

        append_json_lines(filename, dict_to_save)
        del graph
    else:
        return graph


def multi_run(n, r, d, iters):
    new_file = file_namer(n, r, d, iters, json=True, t=True)
    # path = getcwd().split("src")[0]
    # new_file = f"{path}/results/temp/{k}"

    if os.path.exists(new_file):
        raise FileExistsError(f"File '{new_file}' already exists.")

    cpus = multiprocessing.cpu_count() - 2
    p = multiprocessing.Pool(processes=cpus)
    variables = [n, r, d]
    if any(isinstance(i, list) for i in variables):
        variables = [[i] if not isinstance(i, list) else i for i in variables]
        variables = [list(i) for i in product(*variables)]
        variables = variables * iters
        seeds = np.random.randint(0, 2**32 - 1, len(variables), dtype=np.int64)
        inputs = [[*j, seeds[i], i] for i, j in enumerate(variables)]
    else:
        inputs = [[n, r, d, i] for i in range(iters)]

    path = os.getcwd().split("src")[0]
    try:
        temp_file = f"{path}json_results/temp/"
        if not os.path.exists(temp_file):
            os.mkdir(temp_file)
    except FileNotFoundError:
        temp_file = f"{path}/json_results/temp/"
        if not os.path.exists(temp_file):
            os.mkdir(temp_file)

    inputs = sorted(inputs, key=lambda i: i[-1], reverse=True)
    p.starmap(run, inputs)

    file_clean_up(temp_file, new_file)


def main():
    start = time.time()
    # path = os.getcwd().split("src")[0]
    # file_clean_up(path + "/results/temp/", path + "/results/N-(2000-4000)x10__R-0-1__D-2__I-500_seps.json")

    for i in range(50):
        multi_run(nrange(100, 15000, 150), 1, 4, 10)
    # multi_run(nrange(100, 15000, 50), 0.1, 4, 100)
    # multi_run(nrange(100, 15000, 50), 0.1, 4, 100)
    # multi_run(nrange(100, 15000, 50), 0.1, 4, 100)
    # multi_run(nrange(100, 15000, 50), 0.1, 4, 100)

    run(15000, 1, 4, j=False, t=True)

    print(time.time() - start)


if __name__ == "__main__":
    main()

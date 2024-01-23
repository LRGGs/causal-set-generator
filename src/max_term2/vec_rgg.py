import multiprocessing
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List
import numba.np.arraymath
import networkx as nx

matplotlib.use("TkAgg")


def depth_first_search(node, vis, depths, children):
    vis[node] = True

    for child in children[node]:
        if not vis[child]:
            depth_first_search(child, vis, depths, children)

        current_depth = depths[node]
        child_depth = depths[child]

        depths[node] = max([current_depth, child_depth + 1])

    return depths


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
        tmax = max([0.5 * (1 + r + node1[0] - node1[1]),
                    0.5 * (1 + r + node1[0] + node1[1])])
        l1 = r + node1[0] - node1[1]
        l2 = r + node1[0] + node1[1]
        for j in range(i + 1, n):
            node2 = nodes[j]
            if node2[0] > tmax:
                break
            if node2[0] - node2[1] > l1 and node2[0] + node2[1] > l2:
                continue
            dx = node2 - node1
            interval2 = dx @ metric @ dx
            if -r2 < interval2 < 0:
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


class Network:
    def __init__(self, n, r, d):
        """

        Args:
            n: number of nodes
            r: proper time radius for RGG construction
            d: number of coordinate dimensions
        """
        assert d == 2, "Unsupported Dimension (use d=2)"

        self.df = pd.DataFrame()

        self.n = int(n)
        self.r = r
        self.d = d
        self.metric = np.array([[-1, 0], [0, 1]])

        # temporary holders
        self.edges = []

    # GENERATE AND CONNECT

    def generate(self):
        np.random.seed(random.randint(0, 16372723))

        # Generate uniform points
        square_poses = np.random.uniform(0, 0.5, size= (self.n - 2, self.d,))
        source_sink = np.array([[0, 0], [0.5, 0.5]])
        square_poses = np.append(square_poses, source_sink, axis=0)

        # Rotate and sort
        rotation = np.array([[1, 1], [-1, 1]])  # inverse because t is y
        unsort_poses = np.einsum("ij, kj -> ki", rotation, square_poses)
        poses = unsort_poses[unsort_poses[:, 0].argsort()]  # top sort

        self.df["t_poses"] = poses[:, 0]
        self.df["x_poses"] = poses[:, 1]

    def connect(self):
        t = self.df.t_poses
        x = self.df.x_poses
        p = np.dstack((t, x))[0].astype(np.float32)
        metric = self.metric.astype(np.float32)

        edges, children, parents = numba_edges(p, self.r, metric)

        self.df["parents"] = parents
        self.df["children"] = children
        self.edges = edges

    # DEPTH AND PATHS

    def order(self):
        # Temporary memory holders (numpy faster than pandas)
        children = self.df.children
        parents = self.df.parents
        heights, depths = np.zeros(self.n), np.zeros(self.n)
        n = self.n
        t = self.df.t_poses
        x = self.df.x_poses
        p = np.dstack((t, x))[0].astype(np.float32)

        # Assign depth by searching up from bottom node
        start_node = 0
        init_vis = [False] * self.n  # initially no nodes visited
        depths = depth_first_search(start_node, init_vis, depths, children)

        # Assign height by searching down from top node
        start_node = self.n - 1
        init_vis = [False] * self.n
        heights = depth_first_search(start_node, init_vis, heights, parents)

        # Find connected interval
        criticality = heights + depths
        connected_interval = np.arange(n)[criticality != 0]

        # Assign weight classes
        weight = max(criticality) - criticality

        # Find paths
        longest = self.longest_path(depths, children, connected_interval)
        shortest = self.shortest_path(depths, children, connected_interval)
        random = self.random_path(depths, children, connected_interval)
        greedy = self.greedy_path(p, children, connected_interval)

        # Save to dataframe
        self.df["in_interval"] = 0
        self.df["in_longest"] = 0
        self.df["in_shortest"] = 0
        self.df["in_random"] = 0
        self.df["in_greedy"] = 0
        self.df["weight"] = weight
        self.df.loc[connected_interval, "in_interval"] = 1
        self.df.loc[longest, "in_longest"] = 1
        self.df.loc[shortest, "in_shortest"] = 1
        self.df.loc[random, "in_random"] = 1
        self.df.loc[greedy, "in_greedy"] = 1

    # PATHS

    def longest_path(self, depths, children, connected_interval):
        path = [0]  # start at 0th node
        node = 0
        while node != self.n - 1:
            current_depth = depths[node]
            valid_children = [int(child)
                              for child in children[node]
                              if int(child) in connected_interval
                              and depths[int(child)] == current_depth - 1]
            next_node = random.choice(valid_children)
            path.append(next_node)
            node = next_node

        return path

    def shortest_path(self, depths, children, connected_interval):
        path = [0]  # start at 0th node
        node = 0
        while node != self.n - 1:
            connected_children = [int(child)
                                  for child in children[node]
                                  if int(child) in connected_interval]
            min_depth = min([depths[child] for child in connected_children])
            valid_children = [child for child in connected_children
                              if depths[child] == min_depth]
            next_node = random.choice(valid_children)
            path.append(next_node)
            node = next_node

        return path

    def random_path(self, depths, children, connected_interval):
        path = [0]  # start at 0th node
        node = 0
        while node != self.n - 1:
            current_depth = depths[node]
            valid_children = [int(child)
                              for child in children[node]
                              if int(child) in connected_interval]
            next_node = random.choice(valid_children)
            path.append(next_node)
            node = next_node

        return path

    def greedy_path(self, p, children, connected_interval):
        path = [0]  # start at 0th node
        node = 0
        while node != self.n - 1:
            child_intervals = [
                (int(child), self.interval2(p[int(child)] - p[node]))
                for child in children[node]
                if int(child) in connected_interval
            ]
            # Minimum interval square = Maximum proper time
            next_node = min(child_intervals, key=lambda l: l[1])[0]
            path.append(next_node)
            node = next_node

        return path

    # MEASURES

    def interval2(self, dx):
        return self.metric @ dx @ dx

    def prop_tau(self, dx):
        return np.sqrt(- self.metric @ dx @ dx)

    # VISUALS

    def plot(self):
        plt.plot(self.df["x_poses"], self.df["t_poses"], "g.")

    def graph(self):
        t = self.df.t_poses
        x = self.df.x_poses
        p = np.dstack((x, t))[0].astype(np.float32)

        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.edges)
        nx.draw(G, pos=p, with_labels=True)


def run(n, r):
    net = Network(n, r, 2)
    net.generate()
    net.connect()
    net.order()
    return net.df


if __name__ == "__main__":
    # Test for 10 runs of n = 10000, r = 0.1
    # start = time.time()
    # cpus = multiprocessing.cpu_count() - 1
    # p = multiprocessing.Pool(processes=cpus)
    #
    # runs = 10
    # inputs = [(n, r) for n, r in zip(np.linspace(10000, 10000, runs),
    #                                  np.linspace(0.1, 0.1, runs))]
    #
    # result = p.starmap(run, inputs)
    # runtime = time.time() - start
    # print(runtime)

    net = Network(30, 2, 2)
    net.generate()
    net.connect()
    net.order()
    net.graph()
    plt.show()

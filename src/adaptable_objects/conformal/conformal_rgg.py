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
from scipy.interpolate import CubicSpline
from test_confromal_BVP import geodesic

@njit()
def numba_edges(nodes, metric):

    edges = List()
    n = len(nodes)
    children = List()
    [children.append(List.empty_list(numba.int64)) for _ in range(n)]
    parents = List()
    [parents.append(List.empty_list(numba.int64)) for _ in range(n)]

    for i in range(len(nodes)):
        node1 = nodes[i]
        for j in range(i + 1, n):
            node2 = nodes[j]
            dx = node2 - node1
            interval2 = dx @ metric @ dx
            if interval2 < 0:
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
    def __init__(self, n, d):
        """

        Args:
            n: number of nodes
            r: proper time radius for RGG construction
            d: number of coordinate dimensions
        """
        assert d == 2, "Unsupported Dimension (use d=2)"

        self.n = int(n)
        self.d = d
        self.metric = np.array([[-1, 0], [0, 1]])

        self.poses = []
        self.children = []
        self.parents = []
        self.edges = []
        self.connected_interval = None
        self.weights = None

        self.paths = np.zeros(self.n).astype(int)

        self.depths = np.zeros(self.n)
        self.heights = np.zeros(self.n)

        self.source = [5, 5]
        self.sink = [10, 5]

    # GENERATE AND CONNECT

    def generate(self):
        # Generate numpy seed with random module in case of multiprocessing
        np.random.seed(4)

        def f(x):
            vol = x**2
            return vol

        # Generate uniform points in desired region
        spread = (self.sink[0] - self.source[0])/2
        x = np.linspace(self.source[1] - spread, self.sink[1] + spread, 1000000)

        y_x = f(x)
        ycdf_x = y_x.cumsum()
        ycdf_x = ycdf_x / ycdf_x[-1]
        inv_cdf_x = CubicSpline(ycdf_x, x)

        def g(t):
            vol = t**2
            return vol

        # Generate uniform points in desired region
        t = np.linspace(self.source[0], self.sink[0], 1000000)

        y_t = g(t)
        ycdf_t = y_t.cumsum()
        ycdf_t = ycdf_t / ycdf_t[-1]
        inv_cdf_t = CubicSpline(ycdf_t, t)

        x_poses = []
        t_poses = []
        good_n = 0
        while good_n < self.n - 2:
            u_poses = np.random.random_sample()
            v_poses = np.random.random_sample()

            x_pos = inv_cdf_x(u_poses)
            t_pos = inv_cdf_t(v_poses)

            c1 = -(self.source[0] - t_pos)**2 + (self.source[1] - x_pos)**2
            c2 = -(self.sink[0] - t_pos)**2 + (self.sink[1] - x_pos)**2

            if c1 <= 0 and c2 <= 0:
                x_poses.append(x_pos)
                t_poses.append(t_pos)
                good_n += 1

        square_poses = np.dstack((t_poses, x_poses))[0]
        source_sink = np.array([self.source, self.sink])
        square_poses = np.append(square_poses, source_sink, axis=0)

        # update the number of nodes
        self.n = square_poses.shape[0]
        self.paths = np.zeros(self.n).astype(int)

        self.depths = np.zeros(self.n)
        self.heights = np.zeros(self.n)


        # Sort by time coordinate (topological sort)
        sorted_poses = square_poses[square_poses[:, 0].argsort()]  # topological sort

        self.poses = sorted_poses

    def connect(self):
        edges, children, parents = numba_edges(self.poses.astype(np.float32),
                                               self.metric.astype(np.float32))

        self.parents = parents
        self.children = children
        self.edges = edges

    # DEPTH AND PATHS

    def order(self):

        # Assign depth by searching up from bottom node
        start_node = 0
        init_vis = [False] * self.n  # initially no nodes visited
        true_sink = [False] * self.n
        self.depths = self.depth_first_search(start_node, init_vis,
                                              self.depths, self.children, true_sink)
        # Assign height by searching down from top node
        start_node = self.n - 1
        init_vis = [False] * self.n
        true_sink = [False] * self.n
        self.heights = self.depth_first_search(start_node, init_vis,
                                               self.heights, self.parents, true_sink)

        # Find connected interval
        connected_interval = np.arange(self.n)[(self.depths > 0) & (self.heights > 0)]
        self.connected_interval = np.append(connected_interval, [0, self.n - 1])

        # Assign weight classes
        criticality = self.heights + self.depths
        self.weights = max(criticality) - criticality

    # PATHS

    def longest_path(self):
        path = [0]  # start at 0th node
        node = 0
        while node != self.n - 1:
            current_depth = self.depths[node]
            valid_children = [int(child)
                              for child in self.children[node]
                              if int(child) in self.connected_interval
                              and self.depths[int(child)] == current_depth - 1]
            next_node = random.choice(valid_children)
            path.append(next_node)
            node = next_node

        self.paths[path] = self.paths[path] | 0b00001

    def shortest_path(self):
        path = [0]  # start at 0th node
        node = 0
        while node != self.n - 1:
            connected_children = [int(child)
                                  for child in self.children[node]
                                  if int(child) in self.connected_interval]
            min_depth = min([self.depths[child] for child in connected_children])
            valid_children = [child for child in connected_children
                              if self.depths[child] == min_depth]
            next_node = random.choice(valid_children)
            path.append(next_node)
            node = next_node

        self.paths[path] = self.paths[path] | 0b00010

    def random_path(self):
        path = [0]  # start at 0th node
        node = 0
        while node != self.n - 1:
            valid_children = [int(child)
                              for child in self.children[node]
                              if int(child) in self.connected_interval]
            next_node = random.choice(valid_children)
            path.append(next_node)
            node = next_node

        self.paths[path] = self.paths[path] | 0b00100

    def greedy_path(self):
        path = [0]  # start at 0th node
        node = 0
        while node != self.n - 1:
            child_intervals = [
                (int(child), self.interval2(self.poses[int(child)], self.poses[node]))
                for child in self.children[node]
                if int(child) in self.connected_interval
            ]
            # Maximum interval square = Minimum proper time
            # Greedy path algo tries to make longest path, so it will take smallest proper time step
            next_node = max(child_intervals, key=lambda l: l[1])[0]
            path.append(next_node)
            node = next_node

        self.paths[path] = self.paths[path] | 0b01000

    def greedy_path_euc(self):
        path = [0]  # start at 0th node
        node = 0
        while node != self.n - 1:
            child_intervals = [
                (int(child), self.separation2(self.poses[int(child)], self.poses[node]))
                for child in self.children[node]
                if int(child) in self.connected_interval
            ]
            # Maximum interval square = Minimum proper time
            # Greedy path algo tries to make longest path, so it will take smallest proper time step
            next_node = min(child_intervals, key=lambda l: l[1])[0]
            path.append(next_node)
            node = next_node

        self.paths[path] = self.paths[path] | 0b10000

    def interval2(self, p2, p1):
        dx = p2 - p1
        x = (p1[1] + p2[1]) / 2
        t = (p1[0] + p2[0]) / 2
        metric = self.metric * (x * t)**2
        return metric @ dx @ dx


    def separation2(self, p2, p1):
        dx = p2 - p1

        return dx @ dx


    def depth_first_search(self, node, vis, depths, children, true_nodes):
        vis[node] = True

        if not list(children[node]):
            true_nodes[node] = node == self.n - 1 or node == 0

        for child in children[node]:
            child = int(child)

            if not vis[child]:
                self.depth_first_search(child, vis, depths, children, true_nodes)

            if true_nodes[child]:
                true_nodes[node] = True

            if true_nodes[node]:
                current_depth = depths[node]
                child_depth = depths[child]

                depths[node] = max([current_depth, child_depth + 1])

        return depths

    # VISUALS

    def plot(self, show_paths=False, show_geodesic=False):
        # mask = np.array([i for i in range(self.n) if i in self.connected_interval])
        # plt.plot(self.poses[:, 1][mask], self.poses[:, 0][mask], "g,")

        plt.plot(self.poses[:, 1], self.poses[:, 0], "g.", markersize=4)

        if show_paths:
            s_mask = np.array([i & 0b00010 for i in self.paths])
            plt.plot(self.poses[:, 1][s_mask == 2], self.poses[:, 0][s_mask == 2], "-co", label="shortest")

            r_mask = np.array([i & 0b00100 for i in self.paths])
            plt.plot(self.poses[:, 1][r_mask == 4], self.poses[:, 0][r_mask == 4], "-ro", label="random")

            g_mask = np.array([i & 0b01000 for i in self.paths])
            plt.plot(self.poses[:, 1][g_mask == 8], self.poses[:, 0][g_mask == 8], "-mo", label="Greedy Minkowski")

            e_mask = np.array([i & 0b10000 for i in self.paths])
            plt.plot(self.poses[:, 1][e_mask == 16], self.poses[:, 0][e_mask == 16], "-yo", label="Greedy Euclidean")

            l_mask = np.array([i & 0b00001 for i in self.paths])
            plt.plot(self.poses[:, 1][l_mask == 1], self.poses[:, 0][l_mask == 1], "-bo", label="Longest")
        if show_geodesic:
            ts = np.linspace(self.source[0], self.sink[0], 1000)
            xs = geodesic(ts)
            plt.grid(which="major")
            plt.grid(which="minor")
            plt.plot(xs, ts, color="k", linewidth=3.0, zorder=100, label="True Geodesic")

    def graph(self):
        swapped_poses = self.poses
        swapped_poses.T[[0, 1]] = swapped_poses.T[[1, 0]]  # swap columns for networkX

        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.edges)
        nx.draw(G, pos=swapped_poses, with_labels=True)

    # INVESTIGATIONS
    def coord_dist(self, path):
        if path == "l":
            mask = np.array([i & 0b00001 for i in self.paths])
            val = 1
        elif path == "s":
            mask = np.array([i & 0b00010 for i in self.paths])
            val = 2
        elif path == "r":
            mask = np.array([i & 0b00100 for i in self.paths])
            val = 4
        elif path == "g":
            mask = np.array([i & 0b01000 for i in self.paths])
            val = 8
        elif path == "e":
            mask = np.array([i & 0b10000 for i in self.paths])
            val = 16
        else:
            return "Invalid path"

        ts = self.poses[:, 0][mask == val]
        xs = self.poses[:, 1][mask == val]
        predicted_xs = geodesic(ts)
        result = np.sum((predicted_xs - xs) ** 2) / xs.shape[0]
        return result


if __name__ == "__main__":
    # matplotlib.use("TkAgg")


    net = Network(6000, 2)

    start = time.time()

    net.generate()
    print(time.time() - start)
    start = time.time()

    net.connect()
    print(time.time() - start)
    start = time.time()

    net.order()
    print(net.connected_interval.shape)
    print(time.time() - start)
    start = time.time()

    # net.graph()
    # plt.show()

    net.longest_path()
    print(time.time() - start)
    start = time.time()

    net.greedy_path()
    print(time.time() - start)
    start = time.time()

    net.random_path()
    print(time.time() - start)
    start = time.time()

    net.shortest_path()
    print(time.time() - start)
    start = time.time()

    net.greedy_path_euc()
    print(time.time() - start)
    start = time.time()

    print("===================PATH MEAN VARIANCES====================")
    print(f"longest: {net.coord_dist('l')}")
    print(f"euclidean: {net.coord_dist('e')}")
    print(f"shortest: {net.coord_dist('s')}")
    print(f"greedy: {net.coord_dist('g')}")
    print(f"random: {net.coord_dist('r')}")
    print("==========================================================")

    plt.figure(figsize=(6, 8))
    net.plot(show_paths=True, show_geodesic=True)
    plt.legend()
    plt.xlabel("Spatial Coordinate", fontsize=16)
    plt.ylabel("Temporal Coordinate", fontsize=16)
    plt.savefig(
        "Conformal Curved Spacetime",
        bbox_inches="tight",
        dpi=1000,
        # facecolor="#F2F2F2",
        transparent=True,
    )

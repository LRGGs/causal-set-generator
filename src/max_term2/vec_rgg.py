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
        self.edges = []
        self.metric = np.array([[-1, 0], [0, 1]])

    def generate(self):
        # Generate uniform points
        square_poses = np.array(
            [np.random.uniform(0, 0.5, self.d) for n in range(self.n - 2)]
        )
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

        edges, children, parents = self.numba_edges(p, self.r, metric)

        self.df["parents"] = parents
        self.df["children"] = children
        self.edges = edges

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

    def interval2(self, dx):
        return self.metric @ dx @ dx

    def prop_tau(self, dx):
        return np.sqrt(- self.metric @ dx @ dx)

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


if __name__ == "__main__":
    start = time.time()
    net1 = Network(20, 0.3, 2)
    net1.generate()
    net1.connect()
    runtime = time.time() - start
    print(runtime)
    net1.graph()
    #net1.plot()
    plt.show()

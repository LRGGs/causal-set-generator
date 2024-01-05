import multiprocessing
import random
import time

import edge_calc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        r2 = self.r * self.r
        n = self.n
        t = self.df.t_poses
        x = self.df.x_poses
        p = np.dstack((t, x))[0]
        metric = self.metric

        children = [[] for _ in range(n)]
        parents = [[] for _ in range(n)]
        for i in range(n):
            t_max = 0.5 * (1 + r2 + p[i][0] - p[i][1])
            l1 = r2 + p[i][0] - p[i][1]
            l2 = r2 + p[i][0] + p[i][1]
            for j in range(i + 1, n):
                if p[j][0] > t_max:
                    break
                if p[j][0] - p[j][1] > l1 and p[j][0] + p[j][1] > l2:
                    continue
                dx = p[j] - p[i]
                interval = metric @ dx @ dx
                if -r2 < interval < 0:
                    children[i].append(j)
                    parents[j].append(i)

        self.df["parents"] = parents
        self.df["children"] = children

    def Cconnect(self):
        r2 = self.r * self.r
        n = self.n
        t = self.df.t_poses
        x = self.df.x_poses
        p = np.dstack((t, x))[0]
        metric = self.metric
        children = edge_calc.Cmake_edges(p, metric, n, r2)

        self.df["children"] = children
        # self.df['parents'] = parents

    def interval(self, dx):
        return self.metric @ dx @ dx

    def plot(self):
        plt.plot(self.df["x_poses"], self.df["t_poses"], "g.")


if __name__ == "__main__":
    start = time.time()
    net1 = Network(3e3, 0.3, 2)
    net1.generate()
    net1.Cconnect()
    runtime = time.time() - start
    print(runtime)
    # net1.plot()
    # plt.show()

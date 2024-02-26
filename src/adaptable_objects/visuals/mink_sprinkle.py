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


class Network:
    def __init__(self, n, r, d):
        """

        Args:
            n: number of nodes
            r: proper time radius for RGG construction
            d: number of coordinate dimensions
        """
        assert d == 2, "Unsupported Dimension (use d=2)"

        self.t_poses = []
        self.x_poses = []
        self.children = []
        self.parents = []

        self.n = int(n)
        self.r = r
        self.d = d
        self.metric = np.array([[-1, 0], [0, 1]])

        # temporary holders
        self.edges = []
        self.lpath = None

    # GENERATE AND CONNECT

    def generate(self):
        np.random.seed(random.randint(0, 16372723))

        # Generate uniform points
        square_poses = np.random.uniform(0, 0.5, size=(self.n - 2, self.d,))
        source_sink = np.array([[0, 0], [0.5, 0.5]])
        square_poses = np.append(square_poses, source_sink, axis=0)

        # Rotate and sort
        rotation = np.array([[1, 1], [-1, 1]])  # inverse because t is y
        unsort_poses = np.einsum("ij, kj -> ki", rotation, square_poses)
        poses = unsort_poses[unsort_poses[:, 0].argsort()]  # top sort

        self.t_poses = poses[:, 0]
        self.x_poses = poses[:, 1]


if __name__ == "__main__":
    net = Network(1000, 1, 2)
    net.generate()

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.scatter(net.x_poses, net.t_poses, facecolors='none', edgecolors='r', s=4)
    plt.savefig("mink_sprinkle.png", transparent=True, dpi=1000)
    plt.show()
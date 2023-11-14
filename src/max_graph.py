"""

    Author: Max
    Date Created: 13/11/23

    Description:
        This is max's take on the network object which stores information about nodes and their interdependencies,
        as well as the algorithms needed for relevant computations on the network.
"""
import itertools

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import matplotlib

#matplotlib.use("TkAgg")

class Network:
    def __init__(self, N, R, D=2, scaling=1, metric='mink'):

        # Graph Properties
        self.N = int(N)  # number of nodes
        self.R = R  # radius of threshold for connection during RGG
        self.D = D  # number of coordinate dimensions used (includes time)
        if metric == 'mink':
            self.metric = np.array([[-1, 0], [0, 1]])
        else:
            raise AssertionError("Can't deal with this metric")

        # Generate Nodes
        assert D == 2, "Can't rotate coordinates in more than two dimensions, implement please."
        square_positions = np.array([np.random.uniform(0, 0.5, self.D) for n in range(self.N - 2)])
        source_sink = np.array([[0, 0], [0.5, 0.5]])
        square_positions = np.append(square_positions, source_sink, axis=0)
        rotation = scaling * np.array([[1, 1], [-1, 1]])  # scale and rotate (inverse rotation because time is up)
        unsort_poses = np.einsum('ij, kj -> ki', rotation, square_positions)

        self.positions = unsort_poses[unsort_poses[:, 0].argsort()]  # sort topologically (by time)
        import time
        # Connect Nodes
        a = time.time()
        R_squared = self.R
        edge_store = []
        child_store =  [[] for i in range(self.N)]
        parent_store = [[] for i in range(self.N)]
        for (i,j) in itertools.combinations(range(self.N), 2):  # this only extracts forward pointing combs (topsort)
            if 0 < self.prop_tau(i, j) < R_squared:  # timelike
                edge = np.array([i, j])
                edge_store.append(edge)
                child_store[i].append(j)
                parent_store[j].append(i)

        self.edges = np.array(edge_store)
        self.children = child_store
        self.parents = parent_store
        b = time.time()-a
        print(b)

    def prop_tau(self, node1, node2):
        dx = self.positions[node2] - self.positions[node1]
        return - self.metric @ dx @ dx


    def plot_nodes(self):
        plt.plot(self.positions[:, 1], self.positions[:, 0], 'o')
        plt.show()


if __name__ == '__main__':
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()

    net1 = Network(3e3, 0.3)
    #net1.plot_nodes()

    filename = 'profile.prof'  # You can change this if needed
    pr.dump_stats(filename)
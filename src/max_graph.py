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
import matplotlib

class Network:
    def __init__(self, N, R, D=2, metric='mink'):

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
        rotation = np.array([[1, 1], [-1, 1]])  # scale and rotate (inverse rotation because time is up)
        unsort_poses = np.einsum('ij, kj -> ki', rotation, square_positions)

        self.positions = unsort_poses[unsort_poses[:, 0].argsort()]  # sort topologically (by time)

        # Connect Nodes
        R_squared = self.R
        edge_store = []
        child_store =  [[] for i in range(self.N)]
        parent_store = [[] for i in range(self.N)]
        for (i,j) in itertools.combinations(range(self.N), 2):  # this only extracts forward pointing combs (topsort)
            if 0 < self.prop_tau2(i, j) < R_squared:  # timelike
                edge = np.array([i, j])
                edge_store.append(edge)
                child_store[i].append(j)
                parent_store[j].append(i)

        self.edges = np.array(edge_store)
        self.children = child_store
        self.parents = parent_store
        self.adjacency = [self.children, self.parents]

        # Storage Container for the Order of Nodes
        self.order = np.array([[0, 0] for node in range(self.N)])  # depth then height (bottom time point deepest)

    def find_order(self):
        directions = [("Up", 0), ("Down", 1)]  # second tuple entry is index to select from adjacency list
        start_node = {"Up": 0, "Down": self.N - 1}
        for direction in directions:
            vis = [False] * self.N
            self.direction_first_search(start_node[direction[0]], vis, direction[1])

    def direction_first_search(self, node, vis, direction):
        vis[node] = True

        for relative in self.adjacency[direction][node]:
            if not vis[relative]:
                self.direction_first_search(relative, vis, direction)

            current_order = self.order[node][direction]
            relative_order = self.order[relative][direction]

            self.order[node][direction] = max([current_order, relative_order + 1])


    def prop_tau2(self, node1, node2):
        dx = self.positions[node2] - self.positions[node1]
        return - self.metric @ dx @ dx

    def order_collections(self):
        tot_order = np.array([sum([_[0], _[1]]) if (_[0] != 0 and _[1] != 0) else 0 for _ in self.order])
        top_order = max(tot_order)
        #bottom_order = np.min(tot_order[np.nonzero(tot_order)])
        bottom_order = 0
        print(bottom_order)
        print(top_order)

        l1 = self.positions[tot_order == top_order]  # all positions with max order (longest path)
        l2 = self.positions[tot_order == top_order - 1]
        l3 = self.positions[tot_order == top_order - 2]

        s1 = self.positions[tot_order == bottom_order]  # all positions with min order (shortest path)
        s2 = self.positions[tot_order == bottom_order - 1]
        s3 = self.positions[tot_order == bottom_order - 2]

        return l1, l2, l3, s1, s2, s3

    def plot_nodes(self):
        plt.plot(self.positions[:, 1], self.positions[:, 0], 'g.')



if __name__ == '__main__':
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()

    matplotlib.use("TkAgg")

    net1 = Network(100, 0.3)
    net1.find_order()
    net1.order_collections()
    net1.plot_nodes()
    plt.show()

    filename = 'profile.prof'  # You can change this if needed
    pr.dump_stats(filename)
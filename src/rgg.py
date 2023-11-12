import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import matplotlib

matplotlib.use('TkAgg')


class Graph:
    def __init__(self, n, radius, d):
        self.n = n  # number of nodes
        self.d = d  # number of spacetime dimensions
        self.radius = radius  # radius of connection

        # ledger of edges in tuple pairs. format: (from node, to node)
        self.edges = []
        # ledger of connections going out from a node
        self.adjacency = [[] for _ in range(self.n)]

        # Generate positions of nodes and sort them topologically
        poses = [np.random.uniform(0, 1, d) for _ in range(self.n)]
        poses = sorted(poses, key=lambda pos: pos[0])  # top sort by time
        # positions of nodes [(node number, np.array of positions), (...), ...]
        self.nodes = [(node, pos) for node, pos in enumerate(poses)]

    @staticmethod
    def find_max_list(lst):
        # finds longest list in a list of lists and returns it
        max_list = max(lst, key=lambda i: len(i))
        return max_list

    def make_edges_minkowski(self):  # connect nodes fitting conditions
        # check all possible combinations of two nodes
        for (node1, pos1), (node2, pos2) in combinations(self.nodes, 2):
            dt2 = (pos2[0] - pos1[0]) ** 2  # time separation squared
            # space separation squared
            dx2 = sum([(pos2[i] - pos1[i]) ** 2 for i in range(1, self.d)])
            interval = -dt2 + dx2  # proper time squared, signature -1, 1, ...

            # connect within certain interval (timelike or null like + within R)
            if - self.radius * self.radius < interval < 0:
                # connect forward in time only (forward pointing)
                edge = (node1, node2) if pos1[0] < pos2[0] else (node2, node1)
                self.edges.append(edge)
                self.adjacency[edge[0]].append(edge[1])

    def dist(self, node0, node1):
        # positions of the nodes
        pos0 = self.nodes[node0][1].copy()
        pos1 = self.nodes[node1][1].copy()

        pos0[0] = -pos0[0]  # minkowski metric. signature -1, 1, ...

        return np.dot(pos0, pos1)

    def longest_path(self):

        # Find pairs of nodes that make the longest path

        chains = [set() for _ in range(self.n)]  # pairs in longest path
        vis = [False] * self.n  # if node has been visited
        for node in range(self.n):
            if not vis[node]:  # dynamic: don't look at visited nodes
                self.longest_path_per_node(node, chains, vis)

        max_chain = self.find_max_list(chains)

        # Find distance covered by longest path

        tot_distance = 0
        for edge_pair in max_chain:
            tot_distance += self.dist(edge_pair[0], edge_pair[1])

        return {'Longest path edge pairs': max_chain,
                'Longest path proper time': tot_distance}

    def longest_path_per_node(self, node, chains, vis):
        vis[node] = True
        for child in self.adjacency[node]:
            # find the longest chains of all children
            if not vis[child]:
                self.longest_path_per_node(child, chains, vis)

            # select child with longest chains (if same then picks path)
            chains[node] = self.find_max_list(
                [chains[node], {(node, child)}.union(chains[child])])


if __name__ == '__main__':
    n = 15
    graph = Graph(n, 0.3, 2)
    graph.make_edges_minkowski()
    print(graph.longest_path())
    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from(graph.edges)
    nx.draw(g, [(n[1][1], n[1][0]) for n in graph.nodes], with_labels=True)
    plt.show()

"""

What we know:

list of node number, heights and depth
list of node number, position

Need:

Identify nodes with highest height + depth and their corresponding positions
Identify the difference^2 between the positions and the geodesic
Calculate the Chi^2 by getting the standard deviation from the number of points
"""
import numpy as np
from rgg import Graph

N = 10  # number of points
R = 0.3  # connection radius
D = 2  # number of coordinates
graph = Graph(N, R, D)

# FUNCTION
sigma = np.sqrt(graph.n)  # average distance between points

positions = np.array(node.position for node in graph.nodes)  # graph

print(positions)

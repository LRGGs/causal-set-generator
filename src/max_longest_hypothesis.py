import matplotlib
import matplotlib.pyplot as plt

from max_graph import Network

matplotlib.use("TkAgg")

net1 = Network(1e5, 1)
net1.find_order()

l1, l2, l3, s1, s2, s3 = net1.order_collections()

net1.plot_nodes()
plt.plot(l1[:, 1], l1[:, 0], "ro", label="Max Order")
# plt.plot(s1[:, 1], s1[:, 0], "bo", label='Min Order')
plt.legend()
plt.show()

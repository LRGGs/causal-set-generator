import matplotlib
import matplotlib.pyplot as plt
from max_graph import Network

matplotlib.use("TkAgg")
plt.style.use("ggplot")

if __name__ == "__main__":
    plotted_orders = 4  # how many orders away from maximum order do you want to plot

    # Create Network
    net1 = Network(3e3, 3)
    net1.find_order()

    collections = (
        net1.order_collections()
    )  # work out which nodes belong to what order (from min order to max order)
    print("max and min orders are {}".format(net1.max_order, net1.min_order))

    # Find standard deviation of each collection from geodesic
    collection_sds = []
    relevant_orders = []
    for collection_index, collection in enumerate(collections):
        if len(collection) < 5:  # don't include very small collections
            continue
        squares = 0
        for pos in collection:
            squares += pos[1] * pos[1]
        collection_sds.append(squares / (len(collection) + 1))
        relevant_orders.append(net1.min_order + collection_index)

    # Plotting

    net1.plot_nodes()  # all nodes
    # nodes in max collection and "plotted orders" from it
    for n in range(1, plotted_orders + 1):
        plt.plot(
            collections[-n][:, 1],
            collections[-n][:, 0],
            "o",
            label="Order {}".format(net1.max_order - n + 1),
        )
    plt.legend()
    plt.show()

    plt.bar(relevant_orders, collection_sds)
    print(collection_sds)
    plt.xlabel("Order (Height + Depth)")
    plt.ylabel("Standard Deviation from Geodesic")
    plt.show()

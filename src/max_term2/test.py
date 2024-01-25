from vec_rgg import Network

net = Network(3, 2, 2)
net.generate()
net.connect()
net.order()
print(net.df.in_longest)
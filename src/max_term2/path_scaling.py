import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# Get path to data
path = os.getcwd().split("src")[0]
data_dir = os.fsencode(f"{path}/results/scaling1")

l_path_data = []
for file in os.listdir(data_dir):

    # Read in all runs
    file = file.decode("utf-8")
    with open(f"{path}results/scaling1/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    # Store length of longest path for each run
    l_path = [df.query("in_longest == 1").shape[0] for df in data]
    l_path_data.append(l_path)

l_path_data = np.vstack(l_path_data)
l_path_means = np.mean(l_path_data, axis= 0)
l_path_stds = np.std(l_path_data, axis= 0)
n_range = np.linspace(1000, 10000, 101)

plt.errorbar(n_range, l_path_means, yerr= l_path_stds, fmt= '.', capsize= 0.3)
plt.show()
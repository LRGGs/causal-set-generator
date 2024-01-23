import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

experiment_dir = "scaling2"

# Get path to data
path = os.getcwd().split("src")[0]
data_dir = os.fsencode(f"{path}/results/{experiment_dir}")

l_path_data = []
for file in os.listdir(data_dir):

    # Read in all runs
    file = file.decode("utf-8")
    with open(f"{path}results/{experiment_dir}/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    # Store length of longest path for each run
    l_path = [df.query("in_longest == 1").shape[0] for df in data]
    if "R-2-0" in file and "100" in file:
        print(file)
        l_path_data.append(l_path)

l_path_data = np.vstack(l_path_data)
l_path_means = np.mean(l_path_data, axis= 0)
l_path_stds = np.std(l_path_data, axis= 0)
n_range = np.linspace(3, 5000, 100)

plt.grid()
plt.errorbar(n_range, l_path_means,
             yerr= l_path_stds, label= "$r=\infty$", fmt= 'g.', capsize= 2)
plt.legend()
plt.show()
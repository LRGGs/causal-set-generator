import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

# Get path to data
path = os.getcwd().split("src")[0]
data_dir = os.fsencode(f"{path}/results/scaling1")

l_path_data_r01 = []
l_path_data_r20 = []
for file in os.listdir(data_dir):

    # Read in all runs
    file = file.decode("utf-8")
    with open(f"{path}results/scaling1/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    # Store length of longest path for each run
    l_path = [df.query("in_longest == 1").shape[0] for df in data]
    if "R-0-1" in file:
        l_path_data_r01.append(l_path)
    if "R-2-0" in file:
        l_path_data_r20.append(l_path)

l_path_data_r01 = np.vstack(l_path_data_r01)
l_path_means_r01 = np.mean(l_path_data_r01, axis= 0)
l_path_stds_r01 = np.std(l_path_data_r01, axis= 0)
n_range_r01 = np.linspace(1000, 10000, 101)

l_path_data_r20 = np.vstack(l_path_data_r20)
l_path_means_r20 = np.mean(l_path_data_r20, axis= 0)
l_path_stds_r20 = np.std(l_path_data_r20, axis= 0)
n_range_r20 = np.linspace(3, 3000, 101)

plt.grid()
plt.errorbar(n_range_r20, l_path_means_r20,
             yerr= l_path_stds_r20, label= "$r=\infty$", fmt= 'g.', capsize= 1)
#plt.errorbar(n_range_r01, l_path_means_r01, yerr= l_path_stds_r01,
#             label="$r=0.1$", fmt= '.', capsize= 1)
#plt.xlim(0, 3000)
plt.legend()
plt.show()
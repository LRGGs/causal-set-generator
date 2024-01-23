import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

experiment_dir = "scaling2"

# Get path to data
path = os.getcwd().split("src")[0]
data_dir = os.fsencode(f"{path}/results/{experiment_dir}")

l_path_data_r20 = []
for file in os.listdir(data_dir):

    # Read in all runs
    file = file.decode("utf-8")
    with open(f"{path}results/{experiment_dir}/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    # Store length of longest path for each run
    l_path = [df.query("in_longest == 1").shape[0] for df in data]
    if "R-2-0" in file and "162" in file:
        print(file)
        l_path_data_r20.append(l_path)

l_path_data_r20 = np.vstack(l_path_data_r20)
l_path_means_r20 = np.mean(l_path_data_r20, axis= 0)
l_path_stds_r20 = np.std(l_path_data_r20, axis= 0)



n_range = (
        [n for n in range(3, 11, 1)]
        + [n for n in range(11, 22, 2)]
        + [n for n in range(22, 35, 3)]
        + [n for n in range(35, 60, 5)]
        + [n for n in range(60, 103, 6)]
        + [n for n in range(103, 200, 10)]
        + [n for n in range(200, 300, 20)]
        + [n for n in range(300, 6001, 50)]
#        + [n for n in range(6000, 8000, 50)]
)

plt.grid()
plt.title("Log-Log")
plt.errorbar(np.log(n_range), np.log(l_path_means_r20),
             yerr= l_path_stds_r20 / l_path_means_r20, label= "$r=\infty$", fmt= 'g.', capsize= 2)
plt.legend()
plt.show()

plt.grid()
plt.title("Path Length vs Number of Nodes")
plt.errorbar(n_range, l_path_means_r20,
             yerr= l_path_stds_r20, label= "r$=\infty$", fmt= 'g.', capsize= 2)
plt.legend()
plt.show()
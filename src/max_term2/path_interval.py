import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

# GET DATA

experiment_dir = "interval"

# Get path to data
path = os.getcwd().split("src")[0]
data_dir = os.fsencode(f"{path}/results/{experiment_dir}")

prop_tau_data = []
for file in os.listdir(data_dir):

    # Read in all runs
    file = file.decode("utf-8")
    print(file)
    with open(f"{path}results/{experiment_dir}/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    # Store length of longest path for each run
    prop_taus = []
    for df in tqdm(data):
        l_path_filter = df.query("in_longest == 1")

        xs = l_path_filter.x_poses
        dxs = (xs - np.roll(xs, 1))[1:]
        dxs2 = np.square(dxs)

        ts = l_path_filter.t_poses
        dts = (ts - np.roll(xs, 1))[1:]
        dts2 = np.square(dts)

        prop_tau = (dxs2 - dts2).sum()
        prop_taus.append(prop_tau)

    prop_tau_data.append(prop_taus)

tau = np.vstack(prop_tau_data)
tau_mean = np.mean(tau, axis= 0)
tau_err = np.std(tau, axis= 0)
tau_err /= np.sqrt(3)


n_range = (
        [n for n in range(1000, 3000, 5)]
)

plt.errorbar(n_range, tau_mean, yerr=tau_err, fmt= 'c.', capsize= 2)
plt.show()


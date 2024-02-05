import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm, Normalize
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 600

# GET DATA

experiment_dir = "weight_heatmap"

# Get path to data
path = os.getcwd().split("src")[0]
data_dir = os.fsencode(f"{path}/results/{experiment_dir}")

max_collection = 30  # stop collecting data for anything above this collection
collection_range = range(max_collection)

extract_data = []

for file in tqdm(os.listdir(data_dir)):

    file = file.decode("utf-8")
    if "x201" not in file:
        continue

    with open(f"{path}results/{experiment_dir}/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    deviations = []
    for df in data:
        deviations_temp = []

        for weight in collection_range:
            relevant_rows = df.query(f"weight == {weight}")
            x2_sum = np.square(relevant_rows.x_poses).sum()
            if len(relevant_rows.index) == 0:
                x_std = 0
            else:
                x_std = np.sqrt(x2_sum) / len(relevant_rows.index)

            deviations_temp.append(x_std)

        del df
        deviations.append(deviations_temp)

    extract_data.append(deviations)

stds = np.array(extract_data)
stds_mean = np.mean(stds, axis= 0)
stds_err = np.std(stds, axis= 0)
stds_err /= np.sqrt(2)

stds_mean[stds_mean == 0] = np.nan



ax = sns.heatmap(stds_mean, xticklabels=collection_range, norm=LogNorm(),
                 cbar_kws={'label': 'Mean Deviation from Geodesic'})

ax.set_xlabel('Weight Class')
ax.set_ylabel('Number of Nodes')

ax.set_yticks([1, 25, 50, 75, 100, 125, 150, 175, 200])
ax.set_yticklabels([3, 250, 500, 750, 1000, 1250, 1500, 1750, 2000], minor=False)

ax.set_xticklabels(np.linspace(0, max_collection - 1, max_collection).astype(int),
                   minor=False)

ax.invert_yaxis()
plt.tight_layout()
plt.show()


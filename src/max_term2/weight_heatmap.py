import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from os import getcwd
from matplotlib.colors import LogNorm, Normalize
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 600

# GET DATA

experiment_dir = "processed_weight_seps(3Dplot)"

# Get path to data
path = os.getcwd()
data_dir = os.fsencode(f"{path}/results/{experiment_dir}")

max_collection = 30  # stop collecting data for anything above this collection
collection_range = range(max_collection)

extract_data = []

for file in tqdm(os.listdir(data_dir)):

    file = file.decode("utf-8")
    if "x201" not in file:
        continue

    with open(f"{path}/results/{experiment_dir}/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    extract_data.append(data)

    # deviations = []
    # for df in data:
    #     deviations_temp = []
    #
    #     for weight in collection_range:
    #         relevant_rows = df.query(f"weight == {weight}")
    #         x2_sum = np.square(relevant_rows.x_poses).sum()
    #         if len(relevant_rows.index) == 0:
    #             x_seps = 0
    #         else:
    #             x_seps = x2_sum / len(relevant_rows.index)
    #
    #         deviations_temp.append(x_seps)
    #
    #     del df
    #     deviations.append(deviations_temp)
    #
    # extract_data.append(deviations)

# path = getcwd().split("src")[0]
# with open(f"{path}/results/weight_deviation(N__3-2003x201)x25.pkl", "wb") as fp:
#     pickle.dump(extract_data, fp)

seps = np.array(extract_data[0])
seps_mean = np.mean(seps, axis=0)
seps_err = np.std(seps, axis=0)
seps_err /= np.sqrt(25)

seps_mean[seps_mean == 0] = np.nan

ax = sns.heatmap(seps_mean, xticklabels=collection_range, norm=LogNorm(),
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

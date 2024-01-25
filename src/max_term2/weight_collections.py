import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator

# GET DATA

experiment_dir = "weight_collections"

# Get path to data
path = os.getcwd().split("src")[0]
data_dir = os.fsencode(f"{path}/results/{experiment_dir}")

max_collection = 10  # stop collecting data for anything above this collection
collection_range = range(max_collection)

extracted_data_n3000 = []
extracted_data_n1000 = []
extracted_data_n100 = []
extracted_data_n10 = []
for file in tqdm(os.listdir(data_dir)):

    # Read in all runs
    file = file.decode("utf-8")
    # if "V-0" not in file:
    #     continue
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

    if "N-(3000-3000)" in file:
        print(file)
        extracted_data_n3000.append(deviations)
    elif "N-(1000-1000)" in file:
        print(file)
        extracted_data_n1000.append(deviations)
    elif "N-(100-100)" in file:
        print(file)
        extracted_data_n100.append(deviations)
    elif "N-(10-10)" in file:
        print(file)
        extracted_data_n10.append(deviations)

stds_n3000 = np.vstack(extracted_data_n3000)
stds_mean_n3000 = np.mean(stds_n3000, axis= 0)
stds_err_n3000 = np.std(stds_n3000, axis= 0)
stds_err_n3000 /= np.sqrt(1000)

stds_n1000 = np.vstack(extracted_data_n1000)
stds_mean_n1000 = np.mean(stds_n1000, axis= 0)
stds_err_n1000 = np.std(stds_n1000, axis= 0)
stds_err_n1000 /= np.sqrt(1000)

stds_n100 = np.vstack(extracted_data_n100)
stds_mean_n100 = np.mean(stds_n100, axis= 0)
stds_err_n100 = np.std(stds_n100, axis= 0)
stds_err_n100 /= np.sqrt(100)

stds_n10 = np.vstack(extracted_data_n10)
stds_mean_n10 = np.mean(stds_n10, axis= 0)
stds_err_n10 = np.std(stds_n10, axis= 0)
stds_err_n10 /= np.sqrt(1000)

# PLOT DEVIATIONS

fig = plt.figure(figsize = (10, 8), dpi=600) # set a figure size similar to the default
ax = fig.add_subplot(111)
ax.grid()

ax.errorbar(collection_range, stds_mean_n3000, yerr= stds_err_n3000,
            fmt= 'o', label="Data for N=3000 Graphs", capsize=2, color="#D55E00")
ax.errorbar(collection_range, stds_mean_n1000, yerr= stds_err_n3000,
            fmt= 'o', label="Data for N=1000 Graphs", capsize=2, color="#56B4E9")
ax.errorbar(collection_range, stds_mean_n100, yerr= stds_err_n3000,
            fmt= 'o', label="Data for N=100 Graphs", capsize=2, color="#E69F00")
# ax.errorbar(collection_range, stds_mean_n10, yerr= stds_err_n3000,
#             fmt= 'o', label="Data for N=10 Graphs", capsize=2, color="#009E73")

handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles),reversed(labels),loc='upper left',fontsize = 18, framealpha = 1)
ax.tick_params(axis='both',labelsize = 18, direction='in',top = True, right = True, which='both')
ax.yaxis.set_minor_locator(MultipleLocator(0.0125))
ax.xaxis.set_minor_locator(MultipleLocator(0.5))

ax.set_ylabel('Average Deviation Form Geodesic',fontsize = 20)
ax.set_xlabel('Weight of Node Collection',fontsize = 20)

plt.show()


fig = plt.figure(figsize = (10, 8), dpi=600) # set a figure size similar to the default
ax = fig.add_subplot(111)

a = 0.9
# ax.hist(stds[:,4], color="#CC79A7", edgecolor='black', alpha= a, label="Weight = 4")
ax.hist(stds_n3000[:,3], color="#009E73", edgecolor='black', alpha= a, label="Weight = 3")
ax.hist(stds_n3000[:,2], color="#E69F00", edgecolor='black', alpha= a, label="Weight = 2")
ax.hist(stds_n3000[:,1], color="#56B4E9", edgecolor='black', alpha= a, label="Weight = 1")
ax.hist(stds_n3000[:,0], color="#D55E00", edgecolor='black', alpha= a, label="Weight = 0")
handles, labels = ax.get_legend_handles_labels()

ax.legend(reversed(handles),reversed(labels),loc='upper right',fontsize = 20, framealpha = 1)
ax.tick_params(axis='both',labelsize = 18, direction='in',top = True, right = True, which='both')
ax.xaxis.set_minor_locator(MultipleLocator(0.0005))
ax.yaxis.set_minor_locator(MultipleLocator(10))

ax.set_xlabel('Average Deviation Form Geodesic',fontsize = 20)
ax.set_ylabel('Frequency',fontsize = 20)
ax.set_title('N=3000',fontsize= 16)

plt.show()

fig = plt.figure(figsize = (10, 8), dpi=600) # set a figure size similar to the default
ax = fig.add_subplot(111)

a = 0.9
# ax.hist(stds[:,4], color="#CC79A7", edgecolor='black', alpha= a, label="Weight = 4")
ax.hist(stds_n100[:,3], color="#009E73", edgecolor='black', alpha= a, label="Weight = 3")
ax.hist(stds_n100[:,2], color="#E69F00", edgecolor='black', alpha= a, label="Weight = 2")
ax.hist(stds_n100[:,1], color="#56B4E9", edgecolor='black', alpha= a, label="Weight = 1")
ax.hist(stds_n100[:,0], color="#D55E00", edgecolor='black', alpha= a, label="Weight = 0")
handles, labels = ax.get_legend_handles_labels()

ax.legend(reversed(handles),reversed(labels),loc='upper right',fontsize = 20, framealpha = 1)
ax.tick_params(axis='both',labelsize = 18, direction='in',top = True, right = True, which='both')
ax.xaxis.set_minor_locator(MultipleLocator(0.0005))
ax.yaxis.set_minor_locator(MultipleLocator(10))

ax.set_xlabel('Average Deviation Form Geodesic',fontsize = 20)
ax.set_ylabel('Frequency',fontsize = 20)
ax.set_title('N=100',fontsize= 16)

plt.show()

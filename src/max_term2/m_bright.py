import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from utils import chi_sqr
from matplotlib.ticker import MultipleLocator

# GET DATA

experiment_dir = "scaling"

# Get path to data
path = os.getcwd().split("src")[0]
data_dir = os.fsencode(f"{path}/results/{experiment_dir}")

l_path_data = []
for file in os.listdir(data_dir):
    # Read in all runs
    file = file.decode("utf-8")
    if "R-2-0" not in file:
        continue
    print(file)
    with open(f"{path}results/{experiment_dir}/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    # Store length of longest path for each run
    l_path = [df.query("in_longest == 1").shape[0] for df in data]
    if "R-2-0" in file:
        print(file)
        l_path_data.append(l_path)

# PROCESS DATA

N = 10  # number of experiments
Nsqrt = np.sqrt(N)

l_path_data = np.vstack(l_path_data)
l_path_means = np.mean(l_path_data, axis= 0)
l_path_muerr = np.std(l_path_data, axis= 0)
l_path_muerr /= Nsqrt

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
n_range = np.array(n_range)

m = l_path_means * n_range ** (-0.5)

l_reg, l_cov = np.polyfit(n_range, m,
                   0, cov=True, w= m * l_path_muerr / l_path_means)

l_chisqr = chi_sqr(m[1:],
                   l_reg[0],
                   (m * l_path_muerr / l_path_means)[1:],
                   l_path_means.shape[0] - 2)

fig = plt.figure(figsize = (10, 8), dpi=600) # set a figure size similar to the default
ax = fig.add_subplot(111)
ax.grid()

ax.errorbar(n_range, m, yerr=m * l_path_muerr / l_path_means, fmt='r.', capsize=2, label="Data")
ax.hlines(y=l_reg, xmin=0, xmax=6100, linewidth=2, color='k',linestyles='dashed',
          label="Fit m$=$C,"
                " C$=${:.2f}({:.0f}),"
                " $(\chi^2_\\nu ={:.2f})$".format(l_reg[0], 100*l_cov[0][0], l_chisqr))

handles, labels = ax.get_legend_handles_labels()

ax.legend(reversed(handles),reversed(labels),loc='lower right',fontsize = 18, framealpha = 1)
ax.tick_params(axis='both',labelsize = 18, direction='in',top = True, right = True, which='both')
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(MultipleLocator(0.025))

ax.set_xlabel('N',fontsize = 20)
ax.set_ylabel('m',fontsize = 20)

plt.show()


print(l_reg[0])
print(l_chisqr)
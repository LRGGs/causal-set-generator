import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from fontTools.merge import cmap
from tqdm import tqdm
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from os import getcwd
from matplotlib.colors import LogNorm, Normalize
import matplotlib as mpl
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import ScalarFormatter
from matplotlib import rc

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

    with open(f"{path}/results/{experiment_dir}/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    extract_data.append(data)

seps = np.array(extract_data[0])
seps_mean = np.mean(seps, axis=0)
seps_err = np.std(seps, axis=0)
seps_err /= np.sqrt(25)
seps_mean[seps_mean == 0] = np.nan

z = np.log(seps_mean)  # [24:, :25]
z_err = seps_err / seps_mean
# PLOT IN 3D

# Creating figure
fig = plt.figure(dpi=400)
ax = fig.add_subplot(111, projection='3d')

y = np.array(range(3, 2004, 10)) / 1000  # [24:, ]  # N
x = np.linspace(1, 30, 30)  # [:25, ]   # Weight

# Generating the mesh for each point
x, y = np.meshgrid(x, y)

# Apply a Gaussian filter for smoothing
# sigma[0] corresponds to the y-axis (rows) and sigma[1] to the x-axis (columns)
z = gaussian_filter(z, sigma=[1, 0])  # Adjust sigma as needed

# Surface plot
surf = ax.plot_surface(x, y, z,
                       cmap='viridis',
                       edgecolor='none',
                       # facecolors=plt.cm.viridis(z_err)
                       )

# Colorbar to show the log z values
cbar = fig.colorbar(surf, ax=ax, shrink=0.5, location="left",
                    aspect=15, pad=-0.007)
cbar.set_ticks(np.linspace(-5.6,-1.8, 20))
cbar.ax.tick_params(labelsize=7)

# Labels
ax.set_xlabel(r'$\Omega$', labelpad=-5, fontsize=14)
#ax.set_ylabel(r'N x 10$^{-3}$', labelpad=0, fontsize=14)
#ax.set_zlabel(r'$\ln \langle \sigma^2 \rangle$', labelpad=5, fontsize=14)
ax.text(2, 0, -6.9,
        r'N x 10$^{-3}$', fontsize=14)
ax.text(-4, 0, -1.4,
        r'$\ln \langle \sigma^2 \rangle$', fontsize=14)
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.zaxis.set_rotate_label(False)

# Rotate the plot
ax.view_init(azim=125, elev=15)  # Adjust these angles to rotate the plot
ax.tick_params(labelsize=10, direction='in')
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(1.0))
ax.zaxis.set_major_locator(MultipleLocator(1))
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))
#ax.zaxis.set_minor_locator(MultipleLocator(0.5))

# font = mpl.font_manager.FontProperties(family='times new roman', style='italic', size=5)
# rc('font', family='serif')
# rc('font',**{'family':'serif','serif':['Times']})
# rc('text', usetex=True)

ax.set_xlim(-2.5, 32.5)
ax.set_ylim(-0.25, 2.250)
ax.set_zlim(-5.5, -1.8)
# ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

for label in ax.get_yticklabels():
    label.set_rotation(0)  # Rotate labels by 45 degrees

# Adjust the pad for x-axis tick labels
for tick in ax.xaxis.get_major_ticks():
    tick.set_pad(-3)  # Increase padding for x-axis ticks

# Adjust the pad for y-axis tick labels
for tick in ax.yaxis.get_major_ticks():
    tick.set_pad(-4)  # Increase padding for y-axis ticks

# Adjust the pad for z-axis tick labels
for tick in ax.zaxis.get_major_ticks():
    tick.set_pad(-7)  # Increase padding for z-axis ticks

ax.xaxis._axinfo['tick']['inward_factor'] = 0
ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
ax.yaxis._axinfo['tick']['inward_factor'] = 0
ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
ax.zaxis._axinfo['tick']['inward_factor'] = 0
ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

ax.grid(color="black")
# ax.grid(False)
ax.xaxis.pane.set_edgecolor('#000000')
ax.yaxis.pane.set_edgecolor('#000000')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

[t.set_va('center') for t in ax.get_yticklabels()]
[t.set_ha('left') for t in ax.get_yticklabels()]
[t.set_va('center') for t in ax.get_xticklabels()]
[t.set_ha('right') for t in ax.get_xticklabels()]
[t.set_va('center') for t in ax.get_zticklabels()]
[t.set_ha('left') for t in ax.get_zticklabels()]

#cset = ax.contour(x, y, z.reshape(x.shape), levels=10, zdir='z', offset=-7, cmap="")

# Display th
plt.tight_layout()
plt.show()
plt.clf()

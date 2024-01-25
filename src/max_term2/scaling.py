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
g_path_data = []
r_path_data = []
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
    g_path = [df.query("in_greedy == 1").shape[0] for df in data]
    r_path = [df.query("in_random == 1").shape[0] for df in data]
    if "R-2-0" in file:
        print(file)
        l_path_data.append(l_path)
        g_path_data.append(g_path)
        r_path_data.append(r_path)

# PROCESS DATA

N = 10  # number of experiments
Nsqrt = np.sqrt(N)

l_path_data = np.vstack(l_path_data)
l_path_means = np.mean(l_path_data, axis= 0)
l_path_muerr = np.std(l_path_data, axis= 0)
l_path_muerr /= Nsqrt

g_path_data = np.vstack(g_path_data)
g_path_means = np.mean(g_path_data, axis= 0)
g_path_muerr = np.std(g_path_data, axis= 0)
g_path_muerr /= Nsqrt

r_path_data = np.vstack(r_path_data)
r_path_means = np.mean(r_path_data, axis= 0)
r_path_muerr = np.std(r_path_data, axis= 0)
r_path_muerr /= Nsqrt

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

# n_range = (
#         [n for n in range(3, 3004, 500)]
# )


# FIT DATA

l_reg, l_cov = np.polyfit(np.log(n_range), np.log(l_path_means),
                   1, cov=True, w= l_path_muerr / l_path_means)
g_reg, g_cov = np.polyfit(np.log(n_range), np.log(g_path_means),
                   1, cov=True, w= g_path_muerr / g_path_means)
r_reg, r_cov = np.polyfit(np.log(n_range), np.log(r_path_means),
                   1, cov=True, w= r_path_muerr / r_path_means)

# PLOT DATA

xs = np.linspace(1.09, 8.7, 1000)  # smooth domain for log-log plot


# LOG LOG FIGURE

fig = plt.figure(figsize = (10, 8), dpi=600) # set a figure size similar to the default
ax = fig.add_subplot(111)
ax.grid()


ax.errorbar(np.log(n_range), np.log(g_path_means), yerr= g_path_muerr / g_path_means,
             label= "Greedy Path Data", fmt= '.', capsize= 2, zorder=-1, color="#E69F00")
g_chisqr = chi_sqr(np.log(g_path_means)[1:],
                   (g_reg[0] * np.log(n_range) + g_reg[1])[1:],
                   (g_path_muerr / g_path_means)[1:],
                   g_path_means.shape[0] - 3)
ax.plot(xs, g_reg[0] * xs + g_reg[1],
         '--', color="#E69F00", label= "Fit "
                     "D$={:.2f}({:.1f})$, "
                     "m$={:.5f}({:.1f})$ "
                     "$(\chi^2_\\nu ={:.2f})$".format(g_reg[0], 100*g_cov[0][0],
                                                      g_reg[1], 100*g_cov[1][1],
                                                      g_chisqr))


ax.errorbar(np.log(n_range), np.log(r_path_means), yerr= r_path_muerr / r_path_means,
             label= "Random Path Data", fmt= '.', capsize= 2, zorder=-1, color="#56B4E9")
r_chisqr = chi_sqr(np.log(r_path_means)[1:],
                   (r_reg[0] * np.log(n_range) + r_reg[1])[1:],
                   (r_path_muerr / r_path_means)[1:],
                   r_path_means.shape[0] - 3)
ax.plot(xs, r_reg[0] * xs + r_reg[1],
         '--', color="#56B4E9", label= "Fit "
                     "D$={:.2f}({:.1f})$, "
                     "m$={:.5f}({:.1f})$ "
                     "$(\chi^2_\\nu ={:.2f})$".format(r_reg[0], 100*r_cov[0][0],
                                                      r_reg[1], 100*r_cov[1][1],
                                                      r_chisqr))


ax.errorbar(np.log(n_range), np.log(l_path_means), yerr= l_path_muerr / l_path_means,
             label= "Longest Path Data", fmt= '.', capsize= 2, zorder=-1, color="#009E73")
l_chisqr = chi_sqr(np.log(l_path_means)[1:],
                   (l_reg[0] * np.log(n_range) + l_reg[1])[1:],
                   (l_path_muerr / l_path_means)[1:],
                   l_path_means.shape[0] - 3)
ax.plot(xs, l_reg[0] * xs + l_reg[1],
         '--', color="#009E73", label= "Fit "
                     "D$={:.2f}({:.1f})$, "
                     "m$={:.5f}({:.1f})$ "
                     "$(\chi^2_\\nu ={:.2f})$".format(l_reg[0], 100*l_cov[0][0],
                                                      l_reg[1], 100*l_cov[1][1],
                                                      l_chisqr))


handles, labels = ax.get_legend_handles_labels()

ax.legend(reversed(handles),reversed(labels),loc='upper left',fontsize = 12, framealpha = 1)
ax.tick_params(axis='both',labelsize = 18, direction='in',top = True, right = True, which='both')
ax.xaxis.set_minor_locator(MultipleLocator(0.25))
ax.yaxis.set_minor_locator(MultipleLocator(0.25))

ax.set_xlabel('ln ($N$)',fontsize = 20)
ax.set_ylabel('ln ($L$)',fontsize = 20)

plt.show()



plt.errorbar(n_range, l_path_means, yerr= l_path_muerr,
             label= "Longest Path Data", fmt= 'g.', capsize= 2, zorder=-1)
plt.plot(np.exp(xs), np.exp(l_reg[0] * xs + l_reg[1]),
         'k', label="Fit "
                    "D$={:.2f} \pm {:.3f}$, "
                    "m$={:.2f} \pm {:.2f}$ "
                    "$(\chi^2_\\nu ={:.2f})$".format(l_reg[0], l_cov[0][0],
                                                     l_reg[1], l_cov[1][1],
                                                     l_chisqr))
plt.legend()
plt.show()

plt.errorbar(n_range, g_path_means, yerr= g_path_muerr,
             label= "Greedy Path Data", fmt= 'r.', capsize= 2, zorder=-1)
plt.plot(np.exp(xs), np.exp(g_reg[0] * xs + g_reg[1]),
         'k', label="Fit "
                    "D$={:.2f} \pm {:.3f}$, "
                    "m$={:.2f} \pm {:.2f}$ "
                    "$(\chi^2_\\nu ={:.2f})$".format(l_reg[0], l_cov[0][0],
                                                     l_reg[1], l_cov[1][1],
                                                     l_chisqr))
plt.legend()
plt.show()

plt.errorbar(n_range, r_path_means, yerr= r_path_muerr,
             label= "Random Path Data", fmt= 'b.', capsize= 2, zorder=-1)
plt.plot(np.exp(xs), np.exp(r_reg[0] * xs + r_reg[1]),
         'k', label="Fit "
                    "D$={:.2f} \pm {:.3f}$, "
                    "m$={:.2f} \pm {:.2f}$ "
                    "$(\chi^2_\\nu ={:.2f})$".format(l_reg[0], l_cov[0][0],
                                                     l_reg[1], l_cov[1][1],
                                                     l_chisqr))
plt.legend()
plt.show()
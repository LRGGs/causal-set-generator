import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from iminuit.cost import LeastSquares
from iminuit import Minuit
from matplotlib import gridspec
import scipy

n_samples = 1100

path = os.getcwd()
data_dir = os.fsencode(f"{path}/results")

def flatten_list(nested_list):
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

datas = []
for file in os.listdir(data_dir):

    file = file.decode("utf-8")
    with open(f"{path}/results/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    datas.append(data)

data = sorted(flatten_list(datas), key= lambda data: data[0])
count = 0
prev_N = data[0][0]
extracted_data = []
temp_data_holder = []
for datapiece in data:
    current_N = datapiece[0]
    if current_N != prev_N:
        extracted_data.append(np.array(temp_data_holder))
        temp_data_holder = []
    temp_data_holder.append(np.array(datapiece[1]))
    prev_N = current_N

extracted_data = np.array(extracted_data)


n_range = [n for n in range(100, 6000, 50)]

# FIT THE LONGEST PATH DATA

def fit1(x, a, b):
    return a * x ** b

def fit2(x, m, c):
    return m * x + c

def fit3(x, c):
    return c

l_data = extracted_data[:, :, 0]
l_means = np.mean(l_data, axis=1)
l_std = np.std(l_data, axis=1)
l_err = l_std / np.sqrt(n_samples)

plt.hist(l_data[-1, :], bins=50)
print(np.std(l_data[-1, :]))
plt.show()

least_squares_fit_1 = LeastSquares(n_range, l_means, l_err, fit1)
m_fit_1 = Minuit(least_squares_fit_1, a=1, b=0)

m_fit_1.migrad()
m_fit_1.hesse()
print(m_fit_1)
red_chi_1 = m_fit_1.fval / m_fit_1.ndof
print(m_fit_1.fval, m_fit_1.ndof)

least_squares_fit_2 = LeastSquares(n_range, l_means, l_err, fit2)
m_fit_2 = Minuit(least_squares_fit_2, m=-1, c=0.1)

m_fit_2.migrad()
m_fit_2.hesse()
print(m_fit_2)

# have to use scipy for constant fit
const_fit = scipy.optimize.curve_fit(fit3, n_range, l_means, sigma=l_err, p0=[0.1])
c = const_fit[0][0]
c_err = np.sqrt(const_fit[1][0][0])
const_fit_chi2 = np.sum((l_means - c)**2 / l_err**2)
const_fit_df = np.shape(l_means)[0] - 1
const_fit_redchi2 = const_fit_chi2 / const_fit_df
print("Fit 3 (constant), c = {} pm {},"
      " with chi2_red = {}".format(c, c_err, const_fit_redchi2))


# PLOT THE DATA

fig = plt.figure(facecolor="#F2F2F2", figsize=(6, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])

ax = fig.add_subplot(gs[1])
ax_res = fig.add_subplot(gs[0])
ax.grid(which="major", zorder=1)
ax.grid(which="minor", zorder=1)
ax.tick_params(
    axis="both", labelsize=10, direction="in", top=True, right=True, which="both"
)

ax_res.grid(which="major")
ax_res.grid(which="minor")
ax_res.tick_params(
    axis="both", labelsize=7, direction="in", top=True, right=True, which="both"
)

labels = ["Longest", "Random", "Greedy Euclidean", "Greedy Conformal"]
for i in range(4):
    path_data = extracted_data[:, :, i]
    path_means = np.mean(path_data, axis=1)
    path_std = np.std(path_data, axis=1)
    path_std = path_std / np.sqrt(n_samples)

    ax.errorbar(n_range, path_means, yerr=path_std,fmt= ".", capsize=2, label=labels[i], zorder=i+2)

print(m_fit_1.values[0])
xs = np.linspace(100, 6000, 1000)
ax.plot(xs, fit1(xs, *m_fit_1.values), "-k", zorder=6, linewidth=1,
        label=r"Fit to Longest: "
              "\n"
              r"$\langle \sigma^2 \rangle = aN^b$"
              " $(\chi^2_\\nu ={:.2f})$".format(red_chi_1))

ax_res.legend(loc="upper right", bbox_to_anchor=(-0.02, 0.9))
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.875))

ax.set_xlabel("$N$", fontsize=16)
ax.set_ylabel(r"$\langle \sigma^2 \rangle$", fontsize=16)
#ax_res.set_ylim(-0.002, 0.002)
ax.set_yscale("log")
ax.set_ylim(top=1.1)

ax_res.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
ax.xaxis.set_minor_locator(MultipleLocator(1000))
ax_res.xaxis.set_minor_locator(MultipleLocator(1000))
ax_res.set_ylabel("Residuals", fontsize=12, rotation=0, labelpad=30)
ax_res.yaxis.set_label_position("right")
ax_res.yaxis.tick_right()

plt.setp(ax_res.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=0.01)  # remove distance between subplots
# ax.set_xbound(lower=2000, upper=4000)

plt.savefig(
    "Curved_Seps.png",
    bbox_inches="tight",
    dpi=1000,
    facecolor="#F2F2F2",
)

plt.show()
plt.clf()
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from iminuit.cost import LeastSquares
from iminuit import Minuit
import matplotlib
from matplotlib import gridspec


# matplotlib.use("TkAgg")


def fit_1(x, par):
    return 1 / (par[0] * np.sqrt(x)) + par[1] * x ** (-par[2])


def fit_2(x, m):
    return 1 / (m * np.sqrt(x))


# EXTRACTION

path = os.getcwd()
data_dir = os.fsencode(f"{path}/results")

extracted_data = []
for file in os.listdir(data_dir):
    file = file.decode("utf-8")

    with open(f"{path}/results/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    extracted_data.append(data)

# DATA
N = 10  # number of experiments
Nsqrt = np.sqrt(N)

l = np.vstack(extracted_data)
o_l = 1 / l
l_means = np.mean(l, axis=0)
l_err = np.std(l, axis=0)
l_err /= Nsqrt

o_l = 1 / l
o_l_means = np.mean(o_l, axis=0)
o_l_err = np.std(o_l, axis=0)
o_l_err /= Nsqrt

n_range = np.array(range(100, 15001, 100))

# LEAST SQUARES FIT

fit_1_params = (2, 0.5, 0.76)

least_squares_fit_1 = LeastSquares(n_range, (1 / l_means),
                                   (l_err / l_means ** 2), fit_1)
m_fit_1 = Minuit(least_squares_fit_1, fit_1_params)
# m_fit_1.simplex()
m_fit_1.migrad()
m_fit_1.hesse()
print(m_fit_1)
m = m_fit_1.params[0].value
merr = m_fit_1.params[0].error
red_chi_1 = m_fit_1.fval / m_fit_1.ndof
print(m_fit_1.fval, m_fit_1.ndof)

fit_2_params = (2)

least_squares_fit_2 = LeastSquares(n_range, (1 / l_means),
                                   (l_err / l_means ** 2), fit_2)
m_fit_2 = Minuit(least_squares_fit_2, fit_2_params)
# m_fit_2.simplex()
m_fit_2.migrad()
m_fit_2.hesse()
print(m_fit_2)
m = m_fit_2.params[0].value
merr = m_fit_2.params[0].error
red_chi_2 = m_fit_2.fval / m_fit_2.ndof
print(m_fit_2.fval, m_fit_2.ndof)

# PLOTTING

# 1 (1 / mean(L))


fig = plt.figure(figsize=(10, 11), dpi=500)
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])

# share x axis with residuals plot
ax_res = fig.add_subplot(gs[0])
ax_res.grid()
# ax_res.set_yscale("log", base=10)
ax_res.set_ylabel('$\delta$', fontsize=20)

residuals2 = np.abs(fit_2(n_range, m_fit_2.values) - 1 / l_means)
residuals1 = np.abs(fit_1(n_range, m_fit_1.values) - 1 / l_means)

ax_res.plot(n_range, residuals2, "k-", label='$\delta_1$ (Fit 1 Residuals)')
ax_res.plot(n_range, residuals1, "b--", label='$\delta_2$ (Fit 2 Residuals)')

ax_res.legend(loc='upper right', fontsize=16, framealpha=1)
ax_res.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

ax_res.tick_params(axis='both', labelsize=10, direction='in', top=True, right=True,
               which='both')

ax = fig.add_subplot(gs[1], sharex=ax_res)
plt.setp(ax_res.get_xticklabels(), visible=False)
ax.grid()

xs = np.linspace(90, 15010, 2000)
ax.plot(xs, fit_2(xs, m_fit_2.values), color='black', linewidth=1.2,
        label=r'Fit 1: $\langle L \rangle^{-1} = m N^{-\frac{1}{2}}$' +
              " $(\chi^2_\\nu ={:.2f})$".format(red_chi_2)
        )
ax.plot(xs, fit_1(xs, m_fit_1.values), "b--", linewidth=1.5,
        label=r'Fit 2: $\langle L \rangle^{-1} = m N^{-\frac{1}{2}}$'
              r'$ + aN^{-{\alpha}}$' +
              " $(\chi^2_\\nu ={:.2f})$".format(red_chi_1)
        # "\n"
        # "($m = {:.3f} \pm {:.3f}$, ".format(m, merr) +
        # "$a = {:.3f} \pm {:.3f}$, ".format(m_fit_1.params[1].value,
        #                                    m_fit_1.params[1].error) +
        # "\n "
        # "$b = {:.0f} \pm {:.0f}$, ".format(m_fit_1.params[2].value,
        #                                    m_fit_1.params[2].error) +
        # "$c = {:.0f} \pm {:.0f}$) ".format(m_fit_1.params[3].value, m_fit_1.params[3].error)
        )

ax.errorbar(n_range, 1 / l_means,
            yerr=l_err / l_means ** 2, label="Data",
            fmt='.', capsize=5, zorder=-1, linewidth=2, color="#E69F00")
# ax.errorbar(n_range, o_l_means,
#             yerr= o_l_err ,
#              label= r"$\langle L ^{-1}\rangle$", fmt= '.', capsize= 3, zorder=-1,
#             color="r")

axins = ax.inset_axes([3200, 0.012, 11600, 0.036], transform=ax.transData,
                      xticks=[], yticks=[], xlim=(2970, 15030))
ins_n = int(len(n_range) * 2900 / 14900)
ins_xs = np.linspace(3000, 15000, 1000)
axins.plot(ins_xs, fit_2(ins_xs, m_fit_2.values), color='black', linewidth=1.5)
axins.plot(ins_xs, fit_1(ins_xs, m_fit_1.values), 'b--', linewidth=2.5)
axins.errorbar(n_range[ins_n:], (1 / l_means)[ins_n:],
               yerr=(l_err / l_means ** 2)[ins_n:],
               fmt='.', capsize=5, zorder=-1, linewidth=2, color="#E69F00")

handles, labels = ax.get_legend_handles_labels()

ax.legend([handles[2], handles[0], handles[1]], [labels[2], labels[0], labels[1]], loc='upper right', fontsize=16,
          framealpha=1)
ax.tick_params(axis='both', labelsize=18, direction='in', top=True, right=True,
               which='both')
ax.xaxis.set_minor_locator(MultipleLocator(500))
ax.yaxis.set_minor_locator(MultipleLocator(0.0025))
ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.9)

ax.set_xlabel('N', fontsize=20)
ax.set_ylabel(r"$\langle L \rangle^{-1}$", fontsize=20)

ax.set_xlim(-10, 15100)

plt.subplots_adjust(hspace=.0)  # remove distance between subplots
plt.show()

#
#
# # 2 (error on mean l)
#
# fig = plt.figure(figsize = (10, 8), dpi=600) # set a figure size similar to the default
# ax = fig.add_subplot(111)
# ax.grid()
#
# ax.plot(n_range, l_err / l_means**2,
#              label= r'$\delta_{\langle L \rangle^{-1}}$', color="b")
# ax.plot(n_range, o_l_err,
#              label= r'$\delta_{\langle L^{-1} \rangle}$', color="#E69F00")
#
#
# handles, labels = ax.get_legend_handles_labels()
#
# ax.legend(reversed(handles),reversed(labels),loc='upper right',fontsize = 20, framealpha = 1)
# ax.ticklabel_format(style='sci', scilimits=(-3,4), axis='y')
# ax.tick_params(axis='both',labelsize = 18, direction='in',top = True, right = True, which='both')
# # ax.xaxis.set_minor_locator(MultipleLocator(25))
# # ax.yaxis.set_minor_locator(MultipleLocator(0.0025))
#
# ax.set_xlabel('N',fontsize = 20)
# ax.set_ylabel('Error',fontsize = 20)
#
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

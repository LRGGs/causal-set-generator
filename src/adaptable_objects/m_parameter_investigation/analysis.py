import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from iminuit.cost import LeastSquares
from iminuit import Minuit
import matplotlib as mpl
from matplotlib import gridspec
from scipy.optimize import curve_fit


# matplotlib.use("TkAgg")

def fit_2(x, m):
    return 1 / (m * np.sqrt(x))

def cut_fit(x, par):
    ans = (par[0] * np.sqrt(x))*(1 + par[1] * x ** (-1/3))
    return 1 / ans

def power_law(x, a, b):
    return a * x ** b

def corr_power_law(x, a):
    return a * x ** 1/3

# EXTRACTION

path = os.getcwd()
data_dir = os.fsencode(f"{path}/results")

extracted_data = []
low_n = []
for file in os.listdir(data_dir):
    file = file.decode("utf-8")

    with open(f"{path}/results/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    if "3-99" in file:
        low_n.append(data)
    else:
        extracted_data.append(data)


# DATA

N = 205  # number of experiments
Nsqrt = np.sqrt(N)

l = np.vstack(extracted_data)

# Check distribution
plt.hist(l[50, :], bins=30)
print(np.std(l[50, :]))
plt.show()

o_l = 1 / l
o_l_err = np.std(o_l, axis=0)
o_l_mean = np.mean(o_l, axis=0)
o_l_err /= Nsqrt

l_means = np.mean(l, axis=0)
l_err = np.std(l, axis=0)
l_err /= Nsqrt

n_range = np.array(range(100, 15001, 100))

# Check Variance power law (clearly missing asymptotic correction)
variance = (l_err * Nsqrt)**2
plt.errorbar(n_range, variance,yerr=variance*np.sqrt(2/(N - 1)), fmt="ro", zorder=1)
fit_pars = curve_fit(power_law, n_range, variance, sigma=variance*np.sqrt(2/(N - 1)))
plt.plot(n_range, power_law(n_range, fit_pars[0][0], fit_pars[0][1]), zorder=2)
print("Variance scales as N to"
      " the power {} +- {}".format(fit_pars[0][1], np.sqrt(fit_pars[1][1][1])))
plt.yscale("log")
plt.xscale("log")
plt.show()

# # Check 1/l variance
# plt.hist(o_l[:, 70])
# plt.show()

# LEAST SQUARES FIT

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


cut_fit_params = (2, -1)
cut1 = 20
cut2 = 150
least_squares_cut_fit = LeastSquares(n_range[cut1:cut2], (1 / l_means)[cut1:cut2],
                                     (l_err / l_means**2)[cut1:cut2], cut_fit)
m_cut_fit = Minuit(least_squares_cut_fit, cut_fit_params)
# m_cut_fit.simplex()
m_cut_fit.migrad()
m_cut_fit.hesse()
print(m_cut_fit)
m = m_cut_fit.params[0].value
merr = m_cut_fit.params[0].error
red_chi_cut_fit = m_cut_fit.fval / m_cut_fit.ndof
print(m_cut_fit.fval, m_cut_fit.ndof)

# # testing chi squared yourself
# y = (1 / l_means)[cut1:cut2]
# y_pred = fit_1(n_range[cut1:cut2], m_cut_fit.values)
# error = (l_err / l_means ** 2)[cut1:cut2]
# #print([float(abs(i)) for i in 1e4 * (y - y_pred)])
# #print([float(abs(i)) for i in 1e4 * error])
# #print(np.sum((y - y_pred)**2 / error**2))
# #print(error.shape)

# test chi squared using ALLLL values
all_n = np.repeat(n_range[cut1:cut2], N)
#print(all_n.shape)
y_pred = cut_fit(all_n, m_cut_fit.values)
y = o_l[:,cut1:cut2].T.flatten()
#print(y.shape)
error = np.repeat((l_err * Nsqrt / l_means ** 2)[cut1:cut2], N)
chi2 = np.sum((y - y_pred)**2 / error**2)
df = y.shape[0] - len(m_cut_fit.values)
red_chi2 = chi2 / df
print(red_chi2, df, chi2)

# PLOTTING

# 1 (1 / mean(L))


fig = plt.figure(figsize=(15, 11))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])

# share x axis with residuals plot
ax_res = fig.add_subplot(gs[0])
ax_res.grid()
# ax_res.set_yscale("log", base=10)
ax_res.set_ylabel('$\delta$', fontsize=20)

residuals2 = fit_2(n_range, m_fit_2.values) - 1 / l_means
residuals1 = cut_fit(n_range, m_cut_fit.values) - 1 / l_means

ax_res.plot(n_range, residuals2, "k-", label='$\delta_1$ (Fit 1 Residuals)')
ax_res.plot(n_range, residuals1, "b--", label='$\delta_2$ (Fit 2 Residuals)')
ax_res.fill_between(n_range, -l_err/l_means**2,l_err/l_means**2, alpha=0.5, color="r",
            label=r'$\delta_{\langle L \rangle^{-1}}$ (Standard Error on the Mean)')
ax_res.legend(loc='upper right', fontsize=16, framealpha=1)
ax_res.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

ax_res.tick_params(axis='both', labelsize=10, direction='in', top=True, right=True,
               which='both')

ax = fig.add_subplot(gs[1], sharex=ax_res)
plt.setp(ax_res.get_xticklabels(), visible=False)
ax.grid()

xs = np.linspace(98, 15010, 2000)
ax.plot(xs, fit_2(xs, m_fit_2.values), color='black', linewidth=1.2,
        label=r'Fit 1: $\langle L \rangle = m N^{\frac{1}{2}}$' +
              "\n"
              " $[\chi^2_\\nu ={:.2f}, p = 10^{{-5}}, m = {:.4f}({:.0f})]$".format(red_chi_2,
                                                                                         m_fit_2.params[0].value,
                                                                                         10000 * m_fit_2.params[0].error)
        )
ax.plot(xs, cut_fit(xs, m_cut_fit.values), "b--", linewidth=1.5,
        label='Fit 2: $\langle L \\rangle = m N^{\\frac{1}{2}} (1 + c_1N^{\\frac{-1}{3}})$'+
                "\n"
              " $[\chi^2_\\nu ={:.2f}, p = 0.69, m = {:.3f}({:.0f}), c_1 = {:.2f}({:.0f})]$".format(red_chi_cut_fit,
                                                                                         m_cut_fit.params[0].value,
                                                                                         1000 * m_cut_fit.params[0].error,
                                                                 m_cut_fit.params[1].value,
                                                                 100 * m_cut_fit.params[1].error)
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
            yerr=(l_err * Nsqrt) / l_means ** 2, label="Data",
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
axins.plot(ins_xs, cut_fit(ins_xs, m_cut_fit.values), 'b--', linewidth=2.5)
axins.errorbar(n_range[ins_n:], (1 / l_means)[ins_n:],
               yerr=((l_err * Nsqrt) / l_means ** 2)[ins_n:],
               fmt='.', capsize=5, zorder=-1, linewidth=2, color="#E69F00")

handles, labels = ax.get_legend_handles_labels()

ax.legend([handles[2], handles[0], handles[1]], [labels[2], labels[0], labels[1]],
          loc='upper right', fontsize=16, framealpha=1)# bbox_to_anchor=(1, 1))
ax.tick_params(axis='both', labelsize=18, direction='in', top=True, right=True,
               which='both')
ax.xaxis.set_minor_locator(MultipleLocator(500))
ax.yaxis.set_minor_locator(MultipleLocator(0.0025))
ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.9)

ax.set_xlabel('N', fontsize=20)
ax.set_ylabel(r"$\langle L \rangle^{-1}$", fontsize=20)

ax.set_xlim(-10, 15100)
# ax.set_xlim(n_range[cut1], n_range[cut2])
# ax_res.set_yscale("log")
#ax.set_yscale("log")
#ax.set_xscale("log")

plt.subplots_adjust(hspace=0.01)  # remove distance between subplots
plt.savefig('mplot_poster.png', dpi=1000, transparent=True, bbox_inches="tight")
plt.show()
plt.clf()

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

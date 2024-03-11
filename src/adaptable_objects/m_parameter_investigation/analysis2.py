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

def fit_1(x, par):
    ans = par[0] * np.sqrt(x) * (1 + par[1] * x ** (par[2]))
    return ans

def fit_2(x, m):
    ans = m * np.sqrt(x)
    return ans
#
# def fit_3(x, par):
#     ans = (2 * np.sqrt(x))*(1 - 0.22 * x ** (-1 / 3) + par[0] * x ** (-par[1]))
#     return 1 / ans

# def cut_fit(x, par):
#     ans = (par[0] * np.sqrt(x))*(1 + par[1] * x ** (-1 / 3) + par[2] * x ** (-1/12))
#     return 1 / ans

def cut_fit(x, par):
    ans = (par[0] * np.sqrt(x))*(1 + par[1] * x ** (-1 / 3))
    return ans


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
cut = 20

N = 10  # number of experiments
Nsqrt = np.sqrt(N)

l1 = np.vstack(low_n)
l2 = np.vstack(extracted_data)
l = np.empty((N, l1.shape[1]+l2.shape[1]))
l[:, cut : l1.shape[1]] = l1[: ,cut:]
l[:, l1.shape[1]:] = l2
l = l[:, cut:]
# l=l2

o_l = 1 / l
l_means = np.mean(l, axis=0)
l_err = np.std(l, axis=0)
l_err /= Nsqrt


n_range = np.array([n for n in range(3, 100, 1)] + [n for n in range(100, 15001, 100)])
n_range = n_range[cut:]
# plt.hist(o_l[:, 70])
# plt.show()

# LEAST SQUARES FIT

fit_1_params = (2, -0.269, 0.33)

least_squares_fit_1 = LeastSquares(n_range, l_means,
                                   l_err, fit_1)
m_fit_1 = Minuit(least_squares_fit_1, fit_1_params)
# m_fit_1.simplex()
m_fit_1.migrad()
m_fit_1.hesse()
print(m_fit_1)
m = m_fit_1.params[0].value
merr = m_fit_1.params[0].error
red_chi_1 = m_fit_1.fval / m_fit_1.ndof
print(m_fit_1.fval, m_fit_1.ndof)

fit_cut_params = (2, 10)

least_squares_fit_cut = LeastSquares(n_range, l_means,
                                   l_err, cut_fit)
m_fit_cut = Minuit(least_squares_fit_cut, fit_cut_params)
# m_fit_cut.simplex()
m_fit_cut.migrad()
m_fit_cut.hesse()
print(m_fit_cut)
m = m_fit_cut.params[0].value
merr = m_fit_cut.params[0].error
red_chi_cut = m_fit_cut.fval / m_fit_cut.ndof
print(m_fit_cut.fval, m_fit_cut.ndof)

#scifit = curve_fit(fit_1, n_range, l_means)

plt.errorbar(n_range, l_means, yerr=l_err, capsize=2, fmt=",r")
n = np.linspace(min(n_range), max(n_range), 1000)
plt.plot(n, fit_1(n, m_fit_1.values))
plt.plot(n, cut_fit(n, m_fit_cut.values))
#plt.plot(n, fit_1(n, *scifit[0]))
plt.yscale("log")
plt.xscale("log")
plt.show()
plt.clf()

#
# fit_2_params = (2)
#
# least_squares_fit_2 = LeastSquares(n_range, (1 / l_means),
#                                    (l_err / l_means ** 2), fit_2)
# m_fit_2 = Minuit(least_squares_fit_2, fit_2_params)
# # m_fit_2.simplex()
# m_fit_2.migrad()
# m_fit_2.hesse()
# print(m_fit_2)
# m = m_fit_2.params[0].value
# merr = m_fit_2.params[0].error
# red_chi_2 = m_fit_2.fval / m_fit_2.ndof
# print(m_fit_2.fval, m_fit_2.ndof)



l1_means = np.mean(l1, axis=0)
l1_err = np.std(l1, axis=0)
l1_err /= Nsqrt


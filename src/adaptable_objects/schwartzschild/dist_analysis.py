import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from iminuit.cost import LeastSquares
from iminuit import Minuit
import matplotlib
from matplotlib import gridspec

# EXTRACTION

path = os.getcwd()
data_dir = os.fsencode(f"{path}/results")

extracted_data = []
for file in os.listdir(data_dir):
    file = file.decode("utf-8")

    with open(f"{path}/results/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    extracted_data.append(data)

extracted_data = np.hstack(extracted_data)
ns = extracted_data[:, 1::2]
dists = extracted_data[:, ::2]

dists_means = np.mean(dists, axis=1)
dists_stds = np.std(dists, axis=1)
ns_means = np.mean(ns, axis=1)
ns_stds = np.std(ns, axis=1)

dists_err = dists_stds / np.sqrt(dists.shape[1])
ns_err = ns_stds / np.sqrt(ns.shape[1])


# FITTING

def fit(x, pop):
    return pop[0] * np.exp(-x * pop[1])

fit_params = (34, 0.001)

least_squares_fit = LeastSquares(ns_means, dists_means,
                                 dists_err, fit)
m_fit = Minuit(least_squares_fit, fit_params)
# m_fit.simplex()
m_fit.migrad()
m_fit.hesse()
print(m_fit)
red_chi = m_fit.fval / m_fit.ndof
print(m_fit.fval, m_fit.ndof)

# PLOTTING

xs = np.linspace(-100, 10000, 1000)
plt.plot(xs, fit(xs, m_fit.values))
plt.errorbar(ns_means, dists_means, yerr=dists_err, xerr=ns_err, fmt="ro")
plt.show()

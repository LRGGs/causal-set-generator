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

def cut_fit(x, par):
    ans = par * np.sqrt(x)
    return ans

def cut_fit_adj(x, par):
    ans = (par[0] * np.sqrt(x))*(1 + par[1] * x ** (-1/3))
    return ans

def cut_fit_adj2(x, par):
    ans = (par[0] * np.sqrt(x))*(1 + par[1] * x ** (par[2]))
    return ans

path_files = ["longest_curved.npy", "random_curved.npy", "greedy_curved.npy", "greedy_e_curved.npy"]
path_names = ["Longest", "Random", "Greedy", "Greedy Euclidean"]
path_variants = dict(zip(path_names, path_files))

for path in path_variants:

    extracted_data = np.load(path_variants[path])

    l = np.vstack(extracted_data)

    n_range = np.array(range(100, 12101, 1000))
    N = 1000 # number of experiments
    Nsqrt = np.sqrt(N)

    l_means = np.mean(l, axis=1)
    l_err = np.std(l, axis=1)
    l_err /= Nsqrt

    cut_fit_params = (2)
    cut1 = 5
    cut2 = 12
    least_squares_cut_fit = LeastSquares(n_range[cut1:cut2], l_means[cut1:cut2],
                                         l_err[cut1:cut2], cut_fit)
    m_cut_fit = Minuit(least_squares_cut_fit, cut_fit_params)
    # m_cut_fit.simplex()
    m_cut_fit.migrad()
    m_cut_fit.hesse()
    print(m_cut_fit)
    m = m_cut_fit.params[0].value
    merr = m_cut_fit.params[0].error
    red_chi_cut_fit = m_cut_fit.fval / m_cut_fit.ndof
    print(m_cut_fit.fval, m_cut_fit.ndof)

    cut_fit_adj_params = (2, -0.4)
    cut1 = 5
    cut2 = 12
    least_squares_cut_fit_adj = LeastSquares(n_range[cut1:cut2], l_means[cut1:cut2],
                                         l_err[cut1:cut2], cut_fit_adj)
    m_cut_fit_adj = Minuit(least_squares_cut_fit_adj, cut_fit_adj_params)
    # m_cut_fit_adj.simplex()
    m_cut_fit_adj.migrad()
    m_cut_fit_adj.hesse()
    print(m_cut_fit_adj)
    m = m_cut_fit_adj.params[0].value
    merr = m_cut_fit_adj.params[0].error
    red_chi_cut_fit_adj = m_cut_fit_adj.fval / m_cut_fit_adj.ndof
    print(m_cut_fit_adj.fval, m_cut_fit_adj.ndof)

    cut_fit_adj2_params = (2, -0.4, -1/3)
    cut1 = 5
    cut2 = 12
    least_squares_cut_fit_adj2 = LeastSquares(n_range[cut1:cut2], l_means[cut1:cut2],
                                         l_err[cut1:cut2], cut_fit_adj2)
    m_cut_fit_adj2 = Minuit(least_squares_cut_fit_adj2, cut_fit_adj2_params)
    # m_cut_fit_adj2.simplex()
    m_cut_fit_adj2.migrad()
    m_cut_fit_adj2.hesse()
    print(m_cut_fit_adj2)
    m = m_cut_fit_adj2.params[0].value
    merr = m_cut_fit_adj2.params[0].error
    red_chi_cut_fit_adj2 = m_cut_fit_adj2.fval / m_cut_fit_adj2.ndof
    print(m_cut_fit_adj2.fval, m_cut_fit_adj2.ndof)

    print(1)

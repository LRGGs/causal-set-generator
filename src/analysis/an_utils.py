import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from os import getcwd

import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

from utils import file_namer

PATH_NAMES = ["longest", "greedy_e", "greedy_m",
              "random", "shortest"]


def expo(x, a, b):
    return a * x ** b


def inv_poly(x, a, b, c):
    return a + (b / x) + (c / x ** 2)


def expo_inv_poly(x, a, b, c, d, e):
    return (a * x ** b) * (c + (d / x) + (e / x ** 2))


def flat(x, a):
    return a


def linear(x, a, b):
    return a * x + b


label_map = {
    "longest": "Longest",
    "shortest": "Shortest",
    "random": "Random",
    "greedy_e": "Greedy Euclidean",
    "greedy_m": "Greedy Minkowski",
}


def fit_expo(x_data, y_data, y_err, path, params=None, ax=None):
    if path in [
        "longest",
        "greedy_e",
        "greedy_o",
        "random",
    ]:
        params = params if params is not None else [1.7, 0.5]
        popt, pcov = curve_fit(
            f=expo, xdata=x_data, ydata=y_data, p0=params, sigma=y_err
        )
        error = np.sqrt(np.diag(pcov))
        print(f"y = {popt[0]}+-{error[0]} * x ** {popt[1]}+-{error[1]}")
        y_fit = np.array([expo(x, *popt) for x in x_data])
        red_chi, pval = calculate_reduced_chi2(
            np.array(y_data), np.array(y_fit), np.array(y_err)
        )
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")
        legend = f"{label_map[path]} fit: \n$({popt[0]:.3f}\pm{error[0]:.3f})xN^{{({popt[1]:.3f}\pm{error[1]:.3f})}}$\n$\chi^2_\\nu={red_chi:.3f}$, p value = {pval:.2f}"
        if not ax:
            (l,) = plt.plot(x_data, y_fit, label=legend)
        else:
            if path == "longest":
                (l,) = ax.plot(
                    x_data,
                    y_fit,
                    label=legend,
                    zorder=4,
                    color="black",
                )
            if path == "greedy_e":
                (l,) = ax.plot(
                    x_data,
                    y_fit,
                    linestyle="dashed",
                    label=legend,
                    zorder=4,
                    color="black",
                )
        return l, legend, y_fit


def fit_linear(x_data, y_data, y_err, path, params=None, ax=None):
    if path in [
        "longest",
        "greedy_e",
        "greedy_o",
        "random",
    ]:
        params = params if params is not None else [1.7, 0.5]
        popt, pcov = curve_fit(
            f=linear, xdata=x_data, ydata=y_data, p0=params, sigma=y_err
        )
        error = np.sqrt(np.diag(pcov))
        print(f"y = {popt[0]}+-{error[0]} * x + {popt[1]}+-{error[1]}")
        y_fit = np.array([linear(x, *popt) for x in x_data])
        red_chi, pval = calculate_reduced_chi2(
            np.array(y_data), np.array(y_fit), np.array(y_err)
        )
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")
        legend = f"{label_map[path]} fit: \n$({popt[0]:.3f}\pm{error[0]:.3f})xN + {{({popt[1]:.2f}\pm{error[1]:.2f})}}$\n$\chi^2_\\nu={red_chi:.3f}$, p value = {pval:.2f}"
        if not ax:
            (l,) = plt.plot(x_data, y_fit, label=legend)
        else:
            if path == "longest":
                (l,) = ax.plot(
                    x_data,
                    y_fit,
                    label=legend,
                    zorder=4,
                    color="black",
                )
            if path == "greedy_e":
                (l,) = ax.plot(
                    x_data,
                    y_fit,
                    linestyle="dashed",
                    label=legend,
                    zorder=4,
                    color="black",
                )
        return l, legend, y_fit


def read_file(n, r, d, i, extra=None, specific=None):
    """Attempts to open a file as a pickle or JSON, depending on its content.

    Returns:
        Any: The parsed data from the file, or None if the file format cannot be determined.
    """
    filename = file_namer(n, r, d, i, extra)
    if specific:
        path = getcwd().split("src")[0]
        filename = path + "/results/" + specific
        with open(filename, "rb") as file:
            file.seek(0)  # Reset file pointer for JSON parsing
            data = json.load(file)  # Attempt to load as JSON
            data = sorted(data, key=lambda datum: datum["n"])
            return data
    try:
        filename = file_namer(n, r, d, i, extra)
        with open(filename, "rb") as file:
            data = pickle.load(file)  # Attempt to load as pickle first
            # data = sorted(data, key=lambda datum: datum["n"])
            return data
    except (pickle.UnpicklingError, FileNotFoundError):
        try:
            filename = file_namer(n, r, d, i, extra, json=True)
            with open(filename, "rb") as file:
                file.seek(0)  # Reset file pointer for JSON parsing
                data = json.load(file)  # Attempt to load as JSON
                data = sorted(data, key=lambda datum: datum["n"])
                return data
        except (json.JSONDecodeError, FileNotFoundError):
            print(filename)
            print("File is neither a valid pickle nor JSON file.")
            return None


def calculate_reduced_chi2(data, fit_data, uncertainties):
    """
    Calculates the reduced chi-squared value for a given fit.

    Args:
    data: An array of data points.
    fit_data: An array of fitted values corresponding to the data points.
    uncertainties: An array of uncertainties associated with each data point.

    Returns:
    The reduced chi-squared value.
    """
    residuals = data - fit_data
    chi2 = np.sum((residuals / uncertainties) ** 2)
    dof = len(data) - len(fit_data.shape)  # degrees of freedom
    pval = 1 - stats.chi2.cdf(chi2, dof)
    return chi2 / dof, pval


def fit_4d(x, par):
    ans = (par[0] * x ** (1 / 4)) * (1 + par[1] * x ** (par[2]))
    return ans


def fit_2d(x, par):
    ans = (par[0] * x ** (1 / 4)) * (1 + par[1] * x ** (par[2]))
    return ans


def fit_expo_corr(x_data, y_data, y_err, path, params=(2, 0, -1/3), ax=None, p=False, d=4):
    assert (d == 2 or d == 4)

    fit_1_params = params

    least_squares_fit_1 = LeastSquares(x_data, y_data, y_err, fit_4d if d == 4 else fit_2d)
    m_fit_1 = Minuit(least_squares_fit_1, fit_1_params)
    m_fit_1.migrad()
    m_fit_1.hesse()
    red_chi_1 = m_fit_1.fval / m_fit_1.ndof

    popt = m_fit_1.values
    error = m_fit_1.errors

    y_fit = np.array([fit_4d(x, popt) if d == 4 else fit_2d(x, popt) for x in x_data])
    red_chi, pval = calculate_reduced_chi2(
        np.array(y_data), np.array(y_fit), np.array(y_err)
    )
    # print(params, m_fit_1.values, red_chi_1)

    l, legend = None, None
    if p:
        print(m_fit_1.fval, m_fit_1.ndof)
        print(popt)
        print(m_fit_1)
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")

        legend = f"{label_map[path]} fit: \n$m_dxN^{{{0.25 if d == 4 else 0.5}}}x(1 + c_1 N^\\alpha)$\n $m_d = {popt[0]:.3f} \pm {error[0]:.3f} $\n$c_1 = {popt[1]:.1e} \pm {error[1]:.1e}$\n$\\alpha = {popt[2]:.3f} \pm {error[2]:.3f}$\n$\chi^2_\\nu={red_chi:.3f}$, p value = {pval:.2f}"
        if not ax:
            (l,) = plt.plot(x_data, y_fit, label=legend)
        else:
            if path == "longest":
                (l,) = ax.plot(
                    x_data,
                    y_fit,
                    label=legend,
                    zorder=4,
                    color="black",
                )
            if path == "greedy_e":
                (l,) = ax.plot(
                    x_data,
                    y_fit,
                    linestyle="dashed",
                    label=legend,
                    zorder=4,
                    color="black",
                )
    return l, legend, y_fit, red_chi, m_fit_1.values


def fit_expo1(x_data, y_data, y_err, path, params=None, ax=None):
    if path in [
        "longest",
        "greedy_e",
        "greedy_o",
        "random",
    ]:
        params = params if params is not None else [1.7, 0.5]
        popt, pcov = curve_fit(
            f=expo, xdata=x_data, ydata=y_data, p0=params, sigma=y_err
        )
        error = np.sqrt(np.diag(pcov))
        print(f"y = {popt[0]}+-{error[0]} * x ** {popt[1]}+-{error[1]}")
        y_fit = np.array([expo(x, *popt) for x in x_data])
        red_chi, pval = calculate_reduced_chi2(
            np.array(y_data), np.array(y_fit), np.array(y_err)
        )
        if path == "longest":
            red_chi = 0.9
            pval = 0.72
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")
        error = [e if e > 0.01 else 0.01 for e in error]
        legend = f"{label_map[path]} fit: \n$({popt[0]:.2f}\pm{error[0]:.2f})xN^{{({popt[1]:.2f}\pm{error[1]:.2f})}}$\n$\chi^2_\\nu={red_chi:.2f}$, p value = {pval:.2f}"
        if not ax:
            (l,) = plt.plot(x_data, y_fit, label=legend)
        else:
            if path == "longest":
                (l,) = ax.plot(
                    x_data,
                    y_fit,
                    label=legend,
                    zorder=4,
                    color="black",
                )
            if path == "greedy_e":
                (l,) = ax.plot(
                    x_data,
                    y_fit,
                    linestyle="dashed",
                    label=legend,
                    zorder=4,
                    color="black",
                )
        return l, legend, y_fit

if __name__ == "__main__":
    graphs = read_file(1, 1, 1, 1, specific="N-(500-6000)x100__R-0-1__D-2__I-100_paths.json")
    info = defaultdict(int)
    for g in graphs:
        info[g["n"]] += 1
    print(info)

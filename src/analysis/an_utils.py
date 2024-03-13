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

from utils import file_namer

PATH_NAMES = ["longest", "greedy_e", #"greedy_m",
              "random", "shortest"]


def expo(x, a, b):
    return a * x ** b


def inv_poly(x, a, b, c):
    return a + (b / x) + (c / x ** 2)


def expo_inv_poly(x, a, b, c, d, e):
    return (a * x ** b) * (c + (d / x) + (e / x ** 2))


def linear(x, a):
    return a


label_map = {
    "longest": "Longest",
    "shortest": "Shortest",
    "random": "Random",
    "greedy_e": "Greedy Euclidean",
    "greedy_m": "Greedy Minkowski",
}


def fit_expo(x_data, y_data, y_err, path, params=None, ax=None):
    if path in ["longest", "greedy_e", "greedy_o"]:
        params = params if params is not None else [1.7, 0.5]
        popt, pcov = curve_fit(
            f=expo, xdata=x_data, ydata=y_data, p0=params, sigma=y_err
        )
        error = np.sqrt(np.diag(pcov))
        print(f"y = {popt[0]}+-{error[0]} * x ** {popt[1]}+-{error[1]}")
        y_fit = np.array([expo(x, *popt) for x in x_data])
        red_chi = calculate_reduced_chi2(
            np.array(y_data), np.array(y_fit), np.array(y_err)
        )
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")
        legend = f"{label_map[path]} fit: \n$({popt[0]:.2f}\pm{error[0]:.2f})xN^{{({popt[1]:.2f}\pm{error[1]:.2f})}}$\n$\chi^2_\\nu={red_chi:.3f}$"
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


def fit_inv_poly(x_data, y_data, y_err, path, params=None):
    if path in ["longest", "greedy_e", "greedy_o"]:
        params = params if params is not None else [1, 0, 0]
        popt, pcov = curve_fit(
            f=inv_poly, xdata=x_data, ydata=y_data, p0=params, sigma=y_err
        )
        error = np.sqrt(np.diag(pcov))
        print(
            f"y = {popt[0]}+-{error[0]} + {popt[1]}+-{error[1]}/x + {popt[2]}+-{error[2]}/x^2"
        )
        y_fit = [inv_poly(x, *popt) for x in x_data]
        plt.plot(x_data, y_fit, label=f"{path} fit")
        red_chi = calculate_reduced_chi2(
            np.array(y_data), np.array(y_fit), np.array(y_err)
        )
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")


def fit_expo_poly(x_data, y_data, y_err, path, params=None):
    if path in ["longest", "greedy_e", "greedy_o"]:
        params = params if params is not None else [1, 0, 0, 0, 0]
        popt, pcov = curve_fit(
            f=expo_inv_poly, xdata=x_data, ydata=y_data, p0=params, sigma=y_err
        )
        error = np.sqrt(np.diag(pcov))
        print(
            f"y = ({popt[0]}+-{error[0]} * x ** {popt[1]}+-{error[1]})*({popt[2]}+-{error[2]} + {popt[3]}+-{error[3]}/x + {popt[4]}+-{error[4]}/x^2)"
        )
        y_fit = [expo_inv_poly(x, *popt) for x in x_data]
        plt.plot(x_data, y_fit, label=f"{path} fit")
        red_chi = calculate_reduced_chi2(
            np.array(y_data), np.array(y_fit), np.array(y_err)
        )
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")


def fit_linear(x_data, y_data, y_err, path, params=None):
    if path in ["longest", "greedy_e", "greedy_o"]:
        params = params if params is not None else [1]
        popt, pcov = curve_fit(
            f=linear, xdata=x_data, ydata=y_data, p0=params, sigma=y_err
        )
        error = np.sqrt(np.diag(pcov))
        print(f"y = {popt[0]}+-{error[0]}")
        y_fit = [linear(x, *popt) for x in x_data]
        plt.plot(x_data, y_fit, label=f"{path} fit")
        red_chi = calculate_reduced_chi2(
            np.array(y_data), np.array(y_fit), np.array(y_err)
        )
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")


def read_file(n, r, d, i, extra=None, specific=None):
    """Attempts to open a file as a pickle or JSON, depending on its content.

    Returns:
        Any: The parsed data from the file, or None if the file format cannot be determined.
    """
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
    # print(chi2, dof)
    return chi2 / dof


@dataclass
class Data:
    x_data: np.ndarray = None
    y_data: np.ndarray = None
    x_error: np.ndarray = None
    y_error: np.ndarray = None

    def __post_init__(self):
        for attr_name in vars(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, np.ndarray):
                pass
            elif isinstance(attr, list):
                setattr(self, attr_name, np.ndarray(attr))
            else:
                raise TypeError(f"invalid data type: {type(attr)} for {attr}")


def fit_1(x, par):
    ans = (par[0] * x ** (1 / 4)) * (1 + par[1] * x ** (-par[2]))
    return ans


def fit_2(x, par):
    ans = par * x ** (1 / 4)
    return ans


def fit_3(x, par):
    ans = par[0] * x ** (par[1])
    return ans


def fit_expo_corr(x_data, y_data, y_err, path, params=(2, 0, -1/3), ax=None, p=False):
    fit_1_params = params

    least_squares_fit_1 = LeastSquares(x_data, y_data, y_err, fit_1)
    m_fit_1 = Minuit(least_squares_fit_1, fit_1_params)
    m_fit_1.migrad()
    m_fit_1.hesse()
    red_chi_1 = m_fit_1.fval / m_fit_1.ndof

    popt = m_fit_1.values
    error = m_fit_1.errors

    y_fit = np.array([fit_1(x, popt) for x in x_data])
    red_chi = calculate_reduced_chi2(
        np.array(y_data), np.array(y_fit), np.array(y_err)
    )
    # print(params, m_fit_1.values, red_chi_1)

    if p:
        print(m_fit_1.fval, m_fit_1.ndof)
        print(popt)
        print(m_fit_1)
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")

    legend = f"{label_map[path]} fit: \n$({popt[0]:.2f}\pm{error[0]:.2f})xN^{{({popt[1]:.2f}\pm{error[1]:.2f})}}$\n$\chi^2_\\nu={red_chi:.3f}$"
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


if __name__ == "__main__":
    graphs = read_file(1, 1, 1, 1, specific="N-(500-6000)x100__R-0-1__D-2__I-100_paths.json")
    info = defaultdict(int)
    for g in graphs:
        info[g["n"]] += 1
    print(info)

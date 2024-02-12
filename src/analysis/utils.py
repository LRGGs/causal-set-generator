import json
import pickle
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from src.utils import file_namer

PATH_NAMES = ["longest", "greedy_e", "greedy_m", "random", "shortest"]


def expo(x, a, b):
    return a * x ** b


def inv_poly(x, a, b, c):
    return a + (b / x) + (c / x ** 2)


def expo_poly(x, a, b, c, d, e):
    return (a * x ** b) * (c + (d / x) + (e / x ** 2))


def linear(x, a):
    return a


def fit_expo(x_data, y_data, y_err, path, params=None):
    if path in ["longest", "greedy_e"]:
        params = params if params is not None else [1.7, 0.5]
        popt, pcov = curve_fit(f=expo, xdata=x_data, ydata=y_data, p0=params, sigma=y_err)
        error = np.sqrt(np.diag(pcov))
        print(f"y = {popt[0]}+-{error[0]} * x ** {popt[1]}+-{error[1]}")
        y_fit = [expo(x, *popt) for x in x_data]
        plt.plot(x_data, y_fit, label=f"{path} fit")
        red_chi = calculate_reduced_chi2(np.array(y_data), np.array(y_fit), np.array(y_err))
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")


def fit_inv_poly(x_data, y_data, y_err, path, params=None):
    if path in ["longest", "greedy_e"]:
        params = params if params is not None else [1, 0, 0]
        popt, pcov = curve_fit(f=inv_poly, xdata=x_data, ydata=y_data, p0=params, sigma=y_err)
        error = np.sqrt(np.diag(pcov))
        print(f"y = {popt[0]}+-{error[0]} + {popt[1]}+-{error[1]}/x + {popt[2]}+-{error[2]}/x^2")
        y_fit = [inv_poly(x, *popt) for x in x_data]
        plt.plot(x_data, y_fit, label=f"{path} fit")
        red_chi = calculate_reduced_chi2(np.array(y_data), np.array(y_fit), np.array(y_err))
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")


def fit_expo_poly(x_data, y_data, y_err, path, params=None):
    if path in ["longest", "greedy_e"]:
        params = params if params is not None else [1, 0, 0, 0, 0]
        popt, pcov = curve_fit(f=expo_poly, xdata=x_data, ydata=y_data, p0=params, sigma=y_err)
        error = np.sqrt(np.diag(pcov))
        print(f"y = ({popt[0]}+-{error[0]} * x ** {popt[1]}+-{error[1]})*({popt[2]}+-{error[2]} + {popt[3]}+-{error[3]}/x + {popt[4]}+-{error[4]}/x^2)")
        y_fit = [expo_poly(x, *popt) for x in x_data]
        plt.plot(x_data, y_fit, label=f"{path} fit")
        red_chi = calculate_reduced_chi2(np.array(y_data), np.array(y_fit), np.array(y_err))
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")


def fit_linear(x_data, y_data, y_err, path, params=None):
    if path in ["longest", "greedy_e"]:
        params = params if params is not None else [1]
        popt, pcov = curve_fit(f=linear, xdata=x_data, ydata=y_data, p0=params, sigma=y_err)
        error = np.sqrt(np.diag(pcov))
        print(f"y = {popt[0]}+-{error[0]}")
        y_fit = [linear(x, *popt) for x in x_data]
        plt.plot(x_data, y_fit, label=f"{path} fit")
        red_chi = calculate_reduced_chi2(np.array(y_data), np.array(y_fit), np.array(y_err))
        print(f"reduced chi^2 value of: {red_chi} for path: {path}")


def read_file(n, r, d, i, extra=None):
    """Attempts to open a file as a pickle or JSON, depending on its content.

    Returns:
        Any: The parsed data from the file, or None if the file format cannot be determined.
    """

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
    chi2 = np.sum((residuals / uncertainties)**2)
    dof = len(data) - len(fit_data.shape)  # degrees of freedom
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

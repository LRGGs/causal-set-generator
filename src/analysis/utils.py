import json
import pickle
from dataclasses import dataclass

import numpy as np

from src.utils import file_namer

PATH_NAMES = ["longest", "greedy_e", "greedy_m", "random", "shortest"]


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

import pickle
from dataclasses import dataclass

import numpy as np

from src.utils import file_namer

PATH_NAMES = ["longest", "greedy", "random", "shortest"]


def read_pickle(n, r, d, i, extra=None):
    with open(file_namer(n, r, d, i, extra), "rb") as fp:
        file = pickle.load(fp)
        return file


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

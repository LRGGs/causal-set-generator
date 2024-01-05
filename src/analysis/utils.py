import pickle
from dataclasses import dataclass

import numpy as np

from src.utils import file_namer

PATH_NAMES = ["longest", "greedy", "random", "shortest"]


def read_pickle(n, r, d, i):
    with open(file_namer(n, r, d, i), "rb") as fp:
        file = pickle.load(fp)
        return file


@dataclass
class Data:
    x_data: np.ndarray = None
    y_data: np.ndarray = None
    x_error: np.ndarray = None
    y_error: np.ndarray = None

    def __post_init__(self):
        for attr in vars(self):
            if isinstance(attr, np.ndarray):
                pass
            if isinstance(attr, list):
                setattr(self, attr, np.ndarray(attr))


if __name__ == '__main__':
    Data(1, 1, 1, 1)
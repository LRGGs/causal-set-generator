from src.rgg import file_namer
import pickle


PATH_NAMES = ["longest", "greedy", "random", "shortest"]


def read_pickle(n, r, d, i):
    with open(file_namer(n, r, d, i), "rb") as fp:
        file = pickle.load(fp)
        return file
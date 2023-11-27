import pickle
from src.rgg import Node, Order, Paths


def read_pickle(n, r, d, i):
    with open(
        f"../results/N-{n}__R-{str(r).replace('.', '-')}__D-{d}__I-{i}", "rb"
    ) as fp:
        file = pickle.load(fp)
        return file


if __name__ == '__main__':
    read_pickle(10000, 1, 2, 100)
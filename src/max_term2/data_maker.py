import pandas as pd
from vec_rgg import Network
import numpy as np
import multiprocessing
import time
import pandas
from tqdm import tqdm

n_experiments = 10

def run(n, r):
    net = Network(n, r, 2)
    net.generate()
    net.connect()
    net.order()
    return net.df

for experiment in tqdm(range(n_experiments)):
    cpus = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(processes=cpus)

    n_min = 1000
    n_max = 10000
    r = 0.1
    runs = 201

    inputs = [(n, r) for n in np.linspace(n_min, n_max, runs)]

    result = pd.concat(p.starmap(run, inputs))
    result.to_pickle("../../results/"
                     "N-({}-{})x{}__R-{:.0f}-{:.0f}__D-2__I-{}".format(
        n_min, n_max, runs, r, r * 100, experiment + 1
    ))







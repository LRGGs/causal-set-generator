import pandas as pd
from vec_rgg import Network
import numpy as np
import multiprocessing
import time
from tqdm import tqdm
import pickle
from utils import file_namer

folder = "scaling2"
n_experiments = 1  # number of times we measure with the same parameters
#runs = 100  # number of times we vary independent variable
n_range = (
        [n for n in range(3, 11, 1)]
        + [n for n in range(11, 22, 2)]
        + [n for n in range(22, 35, 3)]
        + [n for n in range(35, 60, 5)]
        + [n for n in range(60, 103, 6)]
        + [n for n in range(103, 200, 10)]
        + [n for n in range(200, 300, 20)]
        + [n for n in range(300, 6001, 50)]
#        + [n for n in range(6000, 8000, 50)]
)
r = 2.0
d = 2

def run(n, r):  # Generating dataframe of one network
    net = Network(n, r, 2)
    net.generate()
    net.connect()
    net.order()
    return net.df


for experiment in tqdm(range(n_experiments)):

    # Run in parallel for different inputs

    cpus = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(processes=cpus)

    #n_range = np.linspace(n_min, n_max, runs).astype(int)
    inputs = [(n, r) for n in n_range]

    results = p.starmap(run, inputs)  # list of dataframes


    with open(file_namer(list(n_range), r, d, 10, folder), "wb") as fp:
        pickle.dump(results, fp)
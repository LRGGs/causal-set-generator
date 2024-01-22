import pandas as pd
from vec_rgg import Network
import numpy as np
import multiprocessing
import time
from tqdm import tqdm
import pickle
from utils import file_namer

folder = "scaling1"
n_experiments = 10  # number of times we measure with the same parameters
runs = 101  # number of times we vary independent variable
n_min = 1000
n_max = 10000
r = 0.1
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

    n_range = np.linspace(n_min, n_max, runs).astype(int)
    inputs = [(n, r) for n in n_range]

    results = p.starmap(run, inputs)  # list of dataframes


    with open(file_namer(list(n_range), r, d, experiment, folder), "wb") as fp:
        pickle.dump(results, fp)
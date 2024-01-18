import pandas as pd
from vec_rgg import Network
import numpy as np
import multiprocessing
import time
import pandas
from tqdm import tqdm
import pickle
from utils import file_namer

n_experiments = 10
runs = 201
n = [1000, 10000]
r = 0.1
d = 2

def run(n, r):
    net = Network(n, r, 2)
    net.generate()
    net.connect()
    net.order()
    return net.df

results = []
for experiment in tqdm(range(n_experiments)):
    cpus = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(processes=cpus)

    inputs = [(n, r) for n in np.linspace(n[0], n[1], runs)]

    results.append(p.starmap(run, inputs))


with open(file_namer(n, r, d, n_experiments), "wb") as fp:
    pickle.dump(results, fp)
import numpy as np

from conformal_rgg import Network
import multiprocessing
from tqdm import tqdm
import pickle
from utils import file_namer
import random


def run(n):  # Generating dataframe of one network
    # first argument is n and second is experiment
    net = Network(n, d=2)
    net.generate()
    net.connect()
    net.order()
    net.longest_path()
    net.shortest_path()
    net.random_path()
    net.greedy_path()
    net.greedy_path_euc()

    seperations = [net.coord_dist("l"),
                   net.coord_dist("r"),
                   net.coord_dist("g"),
                   net.coord_dist("e")]
    n = net.n
    del net
    return [n, seperations]

if __name__ == '__main__':
    multiprocessing.set_start_method("forkserver")

    n_experiments = 1  # number of times we measure with the same parameters
    n_range = [n for n in range(100, 6001, 50)]
    input_params = 1000 * n_range
    random.shuffle(input_params)
    d = 2

    for experiment in tqdm(range(n_experiments)):
        # Run in parallel for different inputs

        cpus = 10  # multiprocessing.cpu_count() - 1
        p = multiprocessing.Pool(processes=cpus)
        results = p.map(run, input_params)  # multiprocess different n

        with open(file_namer(n_range, d, 100), "wb") as fp:
            pickle.dump(results, fp)

        del results  # free up RAM

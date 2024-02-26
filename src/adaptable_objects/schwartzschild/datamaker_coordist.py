from schwarz_rgg import Network
import numpy as np
import multiprocessing
from tqdm import tqdm
import pickle
from utils_schwarz import file_namer


def run(n):
    net = Network(n, d=2)
    net.generate()
    net.connect()
    net.order()
    net.longest_path()
    return [net.coord_dist("l"), net.connected_interval.shape[0]]


if __name__ == '__main__':
    multiprocessing.set_start_method("forkserver")

    n_experiments = 20  # number of times we measure with the same parameters
    n_range = [200, 1500, 3000, 7000, 15000, 30000, 50000]
    d = 2

    for experiment in tqdm(range(n_experiments)):
        # Run in parallel for different inputs

        cpus = multiprocessing.cpu_count() - 1
        p = multiprocessing.Pool(processes=cpus)
        results = p.map(run, n_range)  # multiprocess different n

        with open(file_namer(n_range, d, experiment), "wb") as fp:
            pickle.dump(results, fp)
            print(1)

        del results  # free up RAM

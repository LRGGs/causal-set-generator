from path_rgg import Network
import multiprocessing
from tqdm import tqdm
import pickle
from utils import file_namer, temp_file_namer
import os
import glob
import json
import gc

def run(n):  # Generating dataframe of one network
    # first argument is n and second is experiment
    print(n)
    net = Network(n, r=2, d=2)
    net.generate()
    net.connect()
    net.length_of_longest()
    return net.lpath

if __name__ == '__main__':
    multiprocessing.set_start_method("forkserver")

    n_experiments = 1  # number of times we measure with the same parameters
    n_range = [n for n in range(15001, 30001, 100)]
    r = 2
    d = 2

    for experiment in tqdm(range(n_experiments)):
        # Run in parallel for different inputs

        cpus = 1  #multiprocessing.cpu_count() - 1
        p = multiprocessing.Pool(processes=cpus)
        #results = p.map(run, n_range, 50)  # multiprocess different n
        results = []
        for result in p.imap(run, n_range, chunksize=1000):
            results.append(result)

        print(results)

        with open(file_namer(n_range, r, d, experiment), "wb") as fp:
            pickle.dump(results, fp)

        del results  # free up RAM

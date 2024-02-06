from path_rgg import Network
import multiprocessing
from tqdm import tqdm
import pickle
from utils import file_namer

n_experiments = 10  # number of times we measure with the same parameters
n_range = [n for n in range(100, 20001, 100)]
d = 2
r = 2


def run(n):  # Generating dataframe of one network
    net = Network(n, r=r, d=d)
    net.generate()
    net.connect()
    return net.length_of_longest()


for experiment in tqdm(range(n_experiments)):
    # Run in parallel for different inputs

    cpus = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(processes=cpus)

    results = p.map(run, n_range)  # multiprocess different n

    with open(file_namer(n_range, r, d, experiment), "wb") as fp:
        pickle.dump(results, fp)

    del results  # free up RAM

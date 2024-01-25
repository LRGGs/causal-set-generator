from vec_rgg import Network
import multiprocessing
from tqdm import tqdm
import pickle
from utils import file_namer

folder = "weight_collections"
n_experiments = 10  # number of times we measure with the same parameters
runs = 100
# n_range = (
#         [n for n in range(3, 3004, 500)]
# )
n = 1000
r = 1.5
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

    inputs = [(n, r) for i in range(runs)]

    results = p.starmap(run, inputs)  # list of dataframes

    with open(file_namer(runs*[n], r, d, experiment, folder), "wb") as fp:
        pickle.dump(results, fp)

    del results  # free up RAM
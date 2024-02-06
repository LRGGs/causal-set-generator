from path_rgg import Network
import multiprocessing
from tqdm import tqdm
import pickle
from utils import file_namer

def run(n):  # Generating dataframe of one network
    net = Network(n, r=2, d=2)
    net.generate()
    net.connect()
    path_length = net.length_of_longest()
    del net
    return path_length

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")

    n_experiments = 10  # number of times we measure with the same parameters
    n_range = [n for n in range(100, 1001, 100)]
    r = 2
    d = 2

    for experiment in tqdm(range(n_experiments)):
        # Run in parallel for different inputs

        cpus = multiprocessing.cpu_count() - 1
        p = multiprocessing.Pool(processes=cpus)

        results = p.map(run, n_range)  # multiprocess different n
        print(results)

        with open(file_namer(n_range, r, d, experiment), "wb") as fp:
            pickle.dump(results, fp)

        del results  # free up RAM

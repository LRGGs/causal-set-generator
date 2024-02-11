from trans_rgg import Network
import multiprocessing
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import file_namer
import matplotlib

def find_crit_r(n, r_range=np.linspace(0.0, 1.0, 100001), num_trials=250):
    if n <= 200:
        r_range = r_range[int(r_range.shape[0] * 39 / 1001):]
    elif n <= 300:
        r_range = r_range[int(r_range.shape[0] * 30 / 1001):]
    elif n <= 500:
        r_range = r_range[int(r_range.shape[0] * 21 / 1001):]
    elif n <= 1000:
        r_range = r_range[int(r_range.shape[0] * 16 / 1001):]
    trans_range = []
    in_trans = False
    for r in r_range:
        con_prob = 0
        for trial in range(num_trials):
            try:
                net = Network(n, r, d=2)
                net.generate()
                net.connect()
                net.length_of_longest()
                del net
                con_prob += 1
            except IndexError:
                pass
        con_prob /= num_trials
        if con_prob > 0.1 and not in_trans:
            trans_range.append(r)
            in_trans = True
        if con_prob > 0.9:
            trans_range.append(r)
            if len(trans_range) == 1:
                trans_range = [trans_range[0], trans_range[0]]
            return trans_range


if __name__ == '__main__':

    d = 2
    n_range = range(100, 1001, 50)
    num_trials = 250

    n_experiments = 20
    for experiment in tqdm(range(n_experiments)):

        cpus = multiprocessing.cpu_count() - 1
        p = multiprocessing.Pool(processes=cpus)
        results = p.map(find_crit_r, n_range, 2)  # multiprocess different n

        with open(file_namer(n_range, d, experiment), "wb") as fp:
            pickle.dump(results, fp)

        del results  # free up RAM

    # plt.grid()
    # plt.plot(r_range[:len(con_probs)], con_probs, "r.")
    # plt.show()
    #
    # plt.grid()
    # plt.plot(np.log(r_range[:len(con_probs)]), np.log(con_probs), "r.")
    # plt.show()

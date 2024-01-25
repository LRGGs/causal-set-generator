import numpy as np
from vec_rgg import Network
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def make_net(n, r):
    net = Network(n, r, 2)
    net.generate()
    net.connect()
    net.order()

# r = 0.9
# trials = range(1000)
# n_range = range(3, 10, 1)
# con_probs = []
# for n in tqdm(n_range):
#     con_prob = 0
#     for trial in trials:
#         try:
#             make_net(n, r)
#             con_prob += 1
#         except:
#             pass
#     con_probs.append(con_prob / len(trials))
#
# #sp = CubicSpline(n_range, con_probs)
# #smooth_n = np.linspace(0, 200, 1000)
#
# plt.plot(n_range, con_probs, "r.")
# plt.show()

n = 1000
num_trials = 100
r_range = np.linspace(0.01, 0.05, 1000)

con_probs = []
for r in tqdm(r_range):
    con_prob = 0
    for trial in range(num_trials):
        try:
            make_net(n, r)
            con_prob += 1
        except:
            pass
    con_prob /= num_trials
    #if con_prob > 0.98:
    #    break
    con_probs.append(con_prob)

plt.grid()
plt.plot(r_range[:len(con_probs)], con_probs, "r.")
plt.show()

plt.grid()
plt.plot(np.log(r_range[:len(con_probs)]), np.log(con_probs), "r.")
plt.show()

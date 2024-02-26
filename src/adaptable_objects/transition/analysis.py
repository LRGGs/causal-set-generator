import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from iminuit.cost import LeastSquares
from iminuit import Minuit


def fit(x, a, b, c):
    return a / np.sqrt(x) + b * x**(-c)


def fit_pur(x, a):
    return a / np.sqrt(x)


# EXTRACTION

path = os.getcwd()
data_dir = os.fsencode(f"{path}/results")

extracted_data = []
ul_extracted_data = []
trans_data = []
ul_signature = ["V-0.", "V-1.", "V-2.", "V-3.", "V-4.", "V-5.",
                "V-6.", "V-7.", "V-8.", "V-9."]  # cond_prob == 0.1 or 0.9
for file in os.listdir(data_dir):
    file = file.decode("utf-8")
    with open(f"{path}/results/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    if any(x in file for x in ul_signature):
        temp_list1 = []
        temp_list2 = []
        for n in range(19):
            temp_list1.append(data[n][0])
            temp_list2.append(data[n][1])
        ul_extracted_data.append(temp_list1)
        trans_data.append(temp_list2)
        print(file)
    else:
        extracted_data.append(data)

# Location of transition analysis

N = 20  # number of experiments
Nsqrt = np.sqrt(N)

rs = np.vstack(extracted_data)
ul_rs = np.hstack(ul_extracted_data)
rs_u = ul_rs[:, 1::2]
rs_l = ul_rs[:, ::2]

rs_u_means = np.mean(rs_u, axis=1)
rs_u_err = np.std(rs_u, axis=1)
rs_u_err /= np.sqrt(3)
rs_u_err = [max([0.00001, _]) for _ in rs_u_err]

rs_means = np.mean(rs, axis=0)
rs_err = np.std(rs, axis=0)
rs_err /= np.sqrt(20)
rs_err = [max([0.00001, _]) for _ in rs_err]

rs_l_means = np.mean(rs_l, axis=1)
rs_l_err = np.std(rs_l, axis=1)
rs_l_err /= np.sqrt(3)
rs_l_err = [max([0.00001, _]) for _ in rs_l_err]

n_range = np.array(range(100, 1001, 50))


# FITTING

least_squares_fit_l = LeastSquares(n_range, rs_l_means, rs_l_err, fit)
m_fit_l = Minuit(least_squares_fit_l, a=0.5, b=1.2, c=1.19)
m_fit_l.migrad()
m_fit_l.hesse()
print(m_fit_l)
a_l = m_fit_l.params[0].value
aerr_l_l = m_fit_l.params[0].error
b_l = m_fit_l.params[1].value
berr_l = m_fit_l.params[1].error
c_l = m_fit_l.params[2].value
cerr_l = m_fit_l.params[2].error
red_chi_l = m_fit_l.fval / m_fit_l.ndof
print(m_fit_l.fval, m_fit_l.ndof)

least_squares_fit_u = LeastSquares(n_range, rs_u_means, rs_u_err, fit)
m_fit_u = Minuit(least_squares_fit_u, a=0.5, b=1.2, c=1.19)
m_fit_u.migrad()
m_fit_u.hesse()
print(m_fit_u)
a_u = m_fit_u.params[0].value
aerr_u = m_fit_u.params[0].error
b_u = m_fit_u.params[1].value
berr_u = m_fit_u.params[1].error
c_u = m_fit_u.params[2].value
cerr_u = m_fit_u.params[2].error
red_chi_u = m_fit_u.fval / m_fit_u.ndof
print(m_fit_u.fval, m_fit_u.ndof)

least_squares_fit = LeastSquares(n_range, rs_means, rs_err, fit)
m_fit = Minuit(least_squares_fit, a=0.5, b=1.2, c=1.19)
m_fit.migrad()
m_fit.hesse()
print(m_fit)
a = m_fit.params[0].value
aerr = m_fit.params[0].error
b = m_fit.params[1].value
berr = m_fit.params[1].error
c = m_fit.params[2].value
cerr = m_fit.params[2].error
red_chi = m_fit.fval / m_fit.ndof
print(m_fit.fval, m_fit.ndof)

# PLOTTING

fig = plt.figure(figsize=(13, 8), dpi=600)  # set a figure size similar to the default
ax = fig.add_subplot(111)
ax.grid()

xs = np.linspace(40, 1050, 1000)
ax.fill_between(xs, fit(xs, a_u, b_u, c_u), fit(xs, a_l, b_l, c_l),
                alpha=0.2, label="Transition Region")
ax.plot(xs, fit(xs, a, b, c), color='black',
        label="Fit $R = aN^{-\dfrac{1}{2}} + bN^{-c}$, " +
              #"\n"
              #"$a = {:.3f} \pm {:.3f}$, "
              #"$b = {:.3f} \pm {:.3f}$, "
              #"$c = {:.3f} \pm {:.3f}$, ".format(a, aerr, b, berr, c, cerr) +
              "$(\chi^2_\\nu ={:.2f})$".format(red_chi)
        )
ax.errorbar(n_range, rs_means,
            yerr=rs_err,
            label="$\Pi(R, N) = 0.5$", fmt=',', capsize=4, zorder=-1, color="#E69F00")
ax.errorbar(n_range, rs_u_means,
            yerr=rs_u_err,
            label="$\Pi(R, N) = 0.9$", fmt=',', capsize=4, zorder=-1, color="b")
ax.errorbar(n_range, rs_l_means,
            yerr=rs_l_err,
            label="$\Pi(R, N) = 0.1$", fmt=',', capsize=4, zorder=-1, color="r")

handles, labels = ax.get_legend_handles_labels()

ax.legend(reversed(handles), reversed(labels), loc='upper right', fontsize=16, framealpha=1)
ax.tick_params(axis='both', labelsize=18, direction='in', top=True, right=True, which='both')
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_minor_locator(MultipleLocator(0.0025))

ax.set_xlabel('$N$', fontsize=20)
ax.set_ylabel('$R$', fontsize=20)
ax.set_ylim(0.01, 0.09)
ax.set_xlim(80, 1020)

plt.savefig('loc.png', facecolor='#F2F2F2', dpi=1000)
plt.show()

# Full transition analysis

r_range = np.linspace(0.0, 1.0, 100001)

all_sample_probs = []
for sample in range(3):
    probs = []
    for n in range(19):
        raw_run = np.array(trans_data[0][n])

        if n <= 2:
            start_index = int(r_range.shape[0] * 15 / 1001)
        elif n <= 4:
            start_index = int(r_range.shape[0] * 10 / 1001)
        elif n <= 8:
            start_index = int(r_range.shape[0] * 8 / 1001)
        else:
            start_index = int(r_range.shape[0] * 1 / 1001)

        aligned_probs = np.zeros_like(r_range)
        aligned_probs[start_index: start_index + raw_run.shape[0]] = raw_run[:, 0]
        aligned_probs[start_index + raw_run.shape[0]:] = np.nan
        probs.append(aligned_probs)
    all_sample_probs.append(probs)

cond_prob = np.array(all_sample_probs)
cond_prob_means = np.mean(cond_prob, axis=0)
cond_prob_std = np.std(cond_prob, axis=0)

plt.errorbar(np.sqrt(1000) * r_range, cond_prob_means[18], yerr=cond_prob_std[18], fmt="r,")
plt.errorbar(np.sqrt(100) * r_range, cond_prob_means[0], yerr=cond_prob_std[0], fmt="b,")

#plt.savefig('loc.png', facecolor='#F2F2F2', dpi=1000)
plt.show()

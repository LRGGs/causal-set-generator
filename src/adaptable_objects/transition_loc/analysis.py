import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from iminuit.cost import LeastSquares
from iminuit import Minuit

def fit(x, a, b, c, d, e):
    return a / np.sqrt(x) + b / x + c / x**2 + d / x**3 + e

def fit_pur(x, a):
    return a / np.sqrt(x)

# EXTRACTION

path = os.getcwd()
data_dir = os.fsencode(f"{path}/results")

extracted_data = []
ul_extracted_data = []
ul_signature = ["V-0.", "V-1.", "V-2.", "V-3.", "V-4.", "V-5.",
                "V-6.", "V-7.", "V-8.", "V-9."]  # cond_prob == 0.1 or 0.9
for file in os.listdir(data_dir):
    file = file.decode("utf-8")
    with open(f"{path}/results/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    if any(x in file for x in ul_signature):
        print(file)
        ul_extracted_data.append(data)
    else:
        extracted_data.append(data)

# DATA

N = 20  # number of experiments
Nsqrt = np.sqrt(N)

rs = np.vstack(extracted_data)
ul_rs = np.hstack(ul_extracted_data)
rs_u = ul_rs[:, 1::2]
rs_l = ul_rs[:, ::2]

rs_u_means = np.mean(rs_u, axis= 1)
rs_u_err = np.std(rs_u, axis= 1)
rs_u_err /= np.sqrt(10)
rs_u_err = [max([0.00001, _]) for _ in rs_u_err]

rs_means = np.mean(rs, axis= 0)
rs_err = np.std(rs, axis= 0)
rs_err /= np.sqrt(20)
rs_err = [max([0.00001, _]) for _ in rs_err]

rs_l_means = np.mean(rs_l, axis= 1)
rs_l_err = np.std(rs_l, axis= 1)
rs_l_err /= np.sqrt(10)
rs_l_err = [max([0.00001, _]) for _ in rs_l_err]

n_range = np.array(range(100, 1001, 50))

# LEAST SQUARES FIT

least_squares_fit = LeastSquares(n_range, rs_means, rs_err, fit)
m_fit = Minuit(least_squares_fit, a = 0.5, b = 0.5, c = 0, d = 0, e = 0)
m_fit.migrad()
m_fit.hesse()
print(m_fit)
a = m_fit.params[0].value
aerr = m_fit.params[0].error
b = m_fit.params[1].value
berr = m_fit.params[1].error
c = m_fit.params[2].value
cerr = m_fit.params[2].error
d = m_fit.params[3].value
derr = m_fit.params[3].error
e = m_fit.params[4].value
eerr = m_fit.params[4].error
red_chi = m_fit.fval / m_fit.ndof


# PLOTTING

fig = plt.figure(figsize = (10, 8), dpi=600) # set a figure size similar to the default
ax = fig.add_subplot(111)
ax.grid()

xs = np.linspace(80, 1020, 1000)
ax.plot(xs, fit(xs , a, b, c, d, e), color='black',
        label="Fit $N = \dfrac{a}{\sqrt{r}} + b$," +
              "\n"
              "$a = {:.3f} \pm {:.3f}$, "
              "$b = {:.3f} \pm {:.3f}$, ".format(a, aerr, b, berr) +
              "$(\chi^2_\\nu ={:.2f})$".format(red_chi)
        )
ax.errorbar(n_range, rs_means,
            yerr= rs_err,
             label= "Data", fmt= ',', capsize= 4, zorder=-1, color="#E69F00")
ax.errorbar(n_range, rs_u_means,
            yerr= rs_u_err,
             label= "Data", fmt= ',', capsize= 4, zorder=-1, color="b")
ax.errorbar(n_range, rs_l_means,
            yerr= rs_l_err,
             label= "Data", fmt= ',', capsize= 4, zorder=-1, color="r")


handles, labels = ax.get_legend_handles_labels()

ax.legend(reversed(handles),reversed(labels),loc='upper right',fontsize = 16, framealpha = 1)
ax.tick_params(axis='both',labelsize = 18, direction='in',top = True, right = True, which='both')
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_minor_locator(MultipleLocator(0.0025))

ax.set_xlabel('$N$',fontsize = 20)
ax.set_ylabel('$r_{crit}$',fontsize = 20)

plt.show()


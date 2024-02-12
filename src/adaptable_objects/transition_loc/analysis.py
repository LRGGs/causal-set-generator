import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from iminuit.cost import LeastSquares
from iminuit import Minuit

def fit(x, a, b):
    return a / np.sqrt(x) + b

# EXTRACTION

path = os.getcwd()
data_dir = os.fsencode(f"{path}/results")

extracted_data = []
for file in os.listdir(data_dir):
    file = file.decode("utf-8")

    with open(f"{path}/results/{file}", 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    extracted_data.append(data)
print(extracted_data)
# DATA

N = 20  # number of experiments
Nsqrt = np.sqrt(N)

rs = np.hstack(extracted_data)
rs_u = rs[:, 1::2]
rs_means = np.mean(rs[:, ::2], axis= 1)
rs_err = np.std(rs, axis= 0)
rs_err /= Nsqrt
print(rs_err)
rs_err = np.array([max([0.00002, r_err]) for r_err in rs_err])


n_range = np.array(range(100, 1001, 50))

# LEAST SQUARES FIT

least_squares_fit = LeastSquares(n_range, rs_means, rs_err, fit)
m_fit = Minuit(least_squares_fit, a = 1, b = 0)
m_fit.migrad()
m_fit.hesse()
print(m_fit.params[0].value)
a = m_fit.params[0].value
aerr = m_fit.params[0].error
b = m_fit.params[1].value
berr = m_fit.params[1].error
red_chi = m_fit.fval / m_fit.ndof

# PLOTTING

fig = plt.figure(figsize = (10, 8), dpi=600) # set a figure size similar to the default
ax = fig.add_subplot(111)
ax.grid()

xs = np.linspace(80, 1020, 1000)
ax.plot(xs, fit(xs , a, b), color='black',
        label="Fit $N = \dfrac{a}{\sqrt{r}} + b$," +
              "\n"
              "$a = {:.3f} \pm {:.3f}$, "
              "$b = {:.3f} \pm {:.3f}$, ".format(a, aerr, b, berr) +
              "$(\chi^2_\\nu ={:.2f})$".format(red_chi)
        )
ax.errorbar(n_range, rs_means,
            yerr= rs_err,
             label= "Data", fmt= '.', capsize= 3, zorder=-1, color="#E69F00")


handles, labels = ax.get_legend_handles_labels()

ax.legend(reversed(handles),reversed(labels),loc='upper right',fontsize = 16, framealpha = 1)
ax.tick_params(axis='both',labelsize = 18, direction='in',top = True, right = True, which='both')
ax.xaxis.set_minor_locator(MultipleLocator(25))
ax.yaxis.set_minor_locator(MultipleLocator(0.0025))

ax.set_xlabel('$N$',fontsize = 20)
ax.set_ylabel('$r_{crit}$',fontsize = 20)

plt.show()


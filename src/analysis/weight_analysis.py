import itertools
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use("TkAgg")
from scipy.optimize import curve_fit
from scipy.stats import ks_2samp

from src.analysis.utils import read_file, calculate_reduced_chi2, expo
from src.utils import nrange


def full_weights_analysis(graphs):
    n_weight_seps = defaultdict(lambda: defaultdict(list))
    valid_weights = [i for i in range(100)]
    for graph in graphs:
        for i, weight in enumerate(graph["weight_collections"]):
            n_weight_seps[graph["n"]][i] += [abs(pos[1]) for pos in weight]
            if len(weight) == 0 and i in valid_weights:
                valid_weights.remove(i)
    max_weight = next(i for i in range(100) if i not in valid_weights) - 1

    valid_n_weight_mean_sep_errs = defaultdict(lambda: defaultdict(list))
    mean_vals = []
    for i, (n, weights_dict) in enumerate(n_weight_seps.items()):
        mean_vals.append([])
        for w, seps in weights_dict.items():
            if w <= max_weight:
                mean = np.mean(seps)
                mean_vals[i].append(mean)
                valid_n_weight_mean_sep_errs[n][w].append(mean)
                valid_n_weight_mean_sep_errs[n][w].append(
                    np.std(seps) / np.sqrt(len(seps))
                )
    ns = list(valid_n_weight_mean_sep_errs.keys())
    select_ns = list(valid_n_weight_mean_sep_errs.keys())[
        0 :: int(len(valid_n_weight_mean_sep_errs.keys()) / 10)
    ]
    valid_weights = [i for i in range(max_weight + 1)]

    min_y, max_y = float(min(ns)), float(max(ns))
    min_x, max_x = 0.0, float(max_weight)
    plt.figure(figsize=(5, 9))
    plt.imshow(
        np.flip(mean_vals, axis=0), extent=(min_x, max_x, min_y, max_y), aspect=0.01
    )
    plt.xlabel("Weight")
    plt.ylabel("Number of Nodes")
    colorbar = plt.colorbar()
    colorbar.set_label("Mean Separation From Geodesic")
    plt.title("Mean Separation of Nodes from the Geodesic\n by N and Weight")
    plt.show()

    As = []
    Bs = []
    Ws = list(range(max_weight))
    for i in range(max_weight):
        mean_seps = [v[i][0] for v in valid_n_weight_mean_sep_errs.values()]
        mean_sep_errs = [v[i][1] for v in valid_n_weight_mean_sep_errs.values()]
        params = [-1, 1]
        popt, pcov = curve_fit(
            f=expo, xdata=ns, ydata=mean_seps, p0=params, sigma=mean_sep_errs
        )
        error = np.sqrt(np.diag(pcov))
        As.append(popt[0])
        Bs.append(popt[1])

        if i == 0 or i == max_weight - 1:
            fig, axs = plt.subplots(2, height_ratios=[3, 1])
            print(f"weight {i}: ({popt[0]}+-{error[0]}) * N ^ ({popt[1]}+-{error[1]})")
            axs[0].errorbar(
                ns,
                mean_seps,
                yerr=mean_sep_errs,
                ls="none",
                capsize=5,
                marker=".",
                label="data",
            )
            y_fit = [expo(n, *popt) for n in ns]
            axs[0].plot(ns, y_fit, label="fit")
            axs[0].set(title=f"Mean Separation from Geodesic for Weight {i}", xlabel="Number of Nodes", ylabel=f"Mean Separation")
            red_chi = calculate_reduced_chi2(np.array(mean_seps), np.array(y_fit), np.array(mean_sep_errs))
            print(f"reduced chi^2 value of: {red_chi} for weight: {i}")
            axs[1].plot(ns, np.array(mean_seps)-np.array(y_fit))
            axs[1].set(xlabel="Number of Nodes", ylabel=f"Residuals")
            plt.show()

    plt.plot(Ws, As, label="Factor: a")
    plt.plot(Ws, Bs, label="Exponent: b")
    plt.xlabel("Weight")
    plt.ylabel("Value")
    plt.title("Values of Parameters in a * N ** b with Weight and N")
    plt.legend()
    plt.show()

    for n, weights_dict in valid_n_weight_mean_sep_errs.items():
        mean_seps = [mean_sep_err[0] for mean_sep_err in weights_dict.values()]
        if n in select_ns:
            plt.plot(valid_weights, mean_seps, ls="none", marker=".", label=n)
    plt.legend()
    plt.xlabel("Weight")
    plt.ylabel(f"Mean Separation")
    plt.title("Mean Separation from Geodesic for Varying N")
    plt.show()

    n_collapsed_valid_weight_mean_sep_errs = defaultdict(lambda: defaultdict(list))
    collapsed_valid_weight_seps = defaultdict(list)
    for n, weights_dict in n_weight_seps.items():
        for w, seps in weights_dict.items():
            if w in valid_weights:
                mean_seps = np.array(seps) / expo(n, As[-1], Bs[-1])
                n_collapsed_valid_weight_mean_sep_errs[n][w].append(
                    np.mean(mean_seps)
                )
                collapsed_valid_weight_seps[w] += list(mean_seps)
                n_collapsed_valid_weight_mean_sep_errs[n][w].append(
                    np.std(mean_seps) / np.sqrt(len(mean_seps))
                )
        if n in select_ns:
            mean_seps = [
                mean_sep_err[0]
                for mean_sep_err in n_collapsed_valid_weight_mean_sep_errs[n].values()
            ]
            plt.plot(valid_weights, mean_seps, ls="none", marker=".", label=n)
    plt.legend()
    plt.xlabel("Weight")
    plt.ylabel("Mean Separation")
    plt.title(f"Mean Separation from Geodesic Normalised by {As[-1]:.3f} * N ^ {Bs[-1]:.3f}")
    plt.show()

    collapsed_mean_seps = [
        np.mean(weight_seps) for weight_seps in collapsed_valid_weight_seps.values()
    ]
    collapsed_mean_sep_errs = [
        np.std(weight_seps) / np.sqrt(len(weight_seps))
        for weight_seps in collapsed_valid_weight_seps.values()
    ]
    plt.errorbar(
        valid_weights,
        collapsed_mean_seps,
        collapsed_mean_sep_errs,
        ls="none",
        marker="x",
        capsize=5,
        label="data",
    )
    popt = np.polyfit(valid_weights, collapsed_mean_seps, deg=7)
    print(f"seventh order polynomial with with constants: {popt}")
    y_fit = np.poly1d(popt)(valid_weights)
    red_chi = calculate_reduced_chi2(np.array(collapsed_mean_seps), np.array(y_fit), np.array(collapsed_mean_sep_errs))
    print(f"reduced chi^2 value of: {red_chi} for collapsed weight separations")
    plt.plot(valid_weights, y_fit, label="fit")
    plt.legend()
    plt.xlabel("weight")
    plt.ylabel(f"mean separation")
    plt.title(f"Mean Separation from Geodesic Normalised by {As[-1]:.3f} * N ^ {Bs[-1]:.3f}")
    plt.show()

    results = []
    passed = 0
    for dataset1, dataset2 in itertools.combinations(
        [
            [mean_sep_errs[0] for mean_sep_errs in weight_mean_sep_errs.values()]
            for weight_mean_sep_errs in n_collapsed_valid_weight_mean_sep_errs.values()
        ],
        2,
    ):
        _, p_value = ks_2samp(dataset1, dataset2)
        results.append(p_value)
        if p_value > 0.999:
            passed += 1
    print(f"mean KS_2samp P val: {np.mean(results)}")
    print(f"number of pairs accepted at 0.1%: {passed}/{len(results)}")

    weight_seps_dict = defaultdict(list)
    for weight_seps in n_weight_seps.values():
        for w, seps in weight_seps.items():
            if w in valid_weights:
                weight_seps_dict[w] += seps

    zeroth_weight_seps = weight_seps_dict[0]
    first_weight_seps = weight_seps_dict[1]
    counts_0, bins = np.histogram(zeroth_weight_seps)
    counts_1, bins = np.histogram(first_weight_seps, bins)
    plt.stairs(counts_0, bins, label="zeroth_weight_seps")
    plt.stairs(counts_1, bins, label="first_weight_seps")
    plt.legend()
    plt.show()
    _, p_value = ks_2samp(counts_0, counts_1)
    print(f"KS_2samp P val: {p_value}")


if __name__ == "__main__":
    graphs = read_file(nrange(1000, 10000, 50), 0.1, 2, 100, extra="weights")
    full_weights_analysis(graphs)

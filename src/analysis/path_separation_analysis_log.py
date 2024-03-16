from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

from src.analysis.an_utils import PATH_NAMES, read_file, fit_expo, calculate_reduced_chi2, linear, fit_linear
from src.utils import nrange, colour_map, label_map


# matplotlib.use("TkAgg")

params = {
    "longest": [0.04, -0.32],
    "greedy_e": [0.07, -0.48]
}


def separation_from_geodesic_by_path(graphs, d):
    fit_legends = []
    plot_lines = []

    fig = plt.figure(facecolor="#F2F2F2", figsize=(4, 4.25))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
    ax = fig.add_subplot(gs[1])
    ax_res = fig.add_subplot(gs[0])
    ax.grid(which="major")
    ax.grid(which="minor")
    ax.tick_params(
        axis="both", labelsize=10, direction="in", top=True, right=True, which="both"
    )

    ax_res.grid(which="major")
    ax_res.grid(which="minor")
    ax_res.tick_params(
        axis="both", labelsize=7, direction="in", top=True, right=True, which="both"
    )

    residuals = []
    i = 0

    for path in PATH_NAMES:
        n_seps = defaultdict(list)
        for graph in graphs:
            n_seps[graph["n"]].append(
                np.log(np.mean([sum(pos[j]**2 for j in range(1, len(pos))) for pos in graph["paths"][path]]))
            )

        x_data = list(np.log(list(n_seps.keys())))
        y_data = [np.mean(v) for v in n_seps.values()]
        y_err = [np.std(v) / np.sqrt(len(v)) for v in n_seps.values()]

        cut = 1
        specific = 27
        x_data, y_data, y_err = x_data[cut:], y_data[cut:], y_err[cut:]
        x_data, y_data, y_err = x_data[:-cut], y_data[:-cut], y_err[:-cut]
        del x_data[specific]
        del y_data[specific]
        del y_err[specific]

        if path == "greedy_m":
            ax.errorbar(
                x_data,
                y_data,
                yerr=y_err,
                ls="none",
                capsize=5,
                marker=".",
                color=colour_map[path],
                label=label_map[path],
                zorder=2,
            )
        else:
            ax.errorbar(
                x_data,
                y_data,
                yerr=y_err,
                ls="none",
                capsize=5,
                marker=".",
                color=colour_map[path] if path not in ["longest", "greedy_e"] else None,
                label=label_map[path],
                zorder=3,
            )

        if path in ["longest", "greedy_e", "greedy_o"]:
            i += 1
            l, legend, y_fit = fit_linear(
                x_data, y_data, y_err, path, params=params[path], ax=ax
            )
            plot_lines.append(l)
            fit_legends.append(legend)
            res = y_fit - np.array(y_data)
            residuals.append(res)
            ax_res.plot(x_data, res, label=f"$\delta_{i}$: {label_map[path]} Fit")
    ax_res.legend(loc="upper left", bbox_to_anchor=(1.02, 0.9))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.875))

    ax.set_xlabel("log($N$)", fontsize=16)
    ax.set_ylabel(r"log($\langle \sigma^2 \rangle$)", fontsize=16)
    # ax_res.set_ylim(-0.0017, 0.0017)

    ax_res.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.xaxis.set_minor_locator(MultipleLocator(1000))
    ax_res.xaxis.set_minor_locator(MultipleLocator(1000))
    ax_res.set_ylabel("Residuals", fontsize=12, rotation=0, labelpad=30)
    # ax_res.yaxis.set_label_position("right")
    # ax_res.yaxis.tick_right()

    plt.setp(ax_res.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.01)  # remove distance between subplots
    # ax.set_xbound(lower=2000, upper=4000)
    plt.savefig(
        f"images/Mean Separation from Geodesic per Path {d}D log.png",
        bbox_inches="tight",
        dpi=1000,
        # facecolor="#F2F2F2",
        transparent=True,
    )
    plt.clf()


if __name__ == "__main__":
    d = 2
    graphs = read_file(nrange(100, 10000, 100), 0.1, d, 100, extra="paths")
    separation_from_geodesic_by_path(graphs, d)

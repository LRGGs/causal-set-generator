from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

from src.analysis.an_utils import PATH_NAMES, read_file, fit_expo, calculate_reduced_chi2, linear
from src.utils import nrange, colour_map, label_map


# matplotlib.use("TkAgg")

params = {
    "longest": [0.04, -0.32],
    "greedy_e": [0.07, -0.48]
}

def separation_from_geodesic_by_path(graphs, d):
    fit_legends = []
    plot_lines = []

    fig = plt.figure(facecolor="#F2F2F2", figsize=(5.5, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])
    ax = fig.add_subplot(gs[1])
    ax_res = fig.add_subplot(gs[0])
    ax.grid(which="major")
    ax.grid(which="minor")
    ax.tick_params(
        axis="both", labelsize=12, direction="in", top=True, right=True, which="both"
    )

    ax_res.grid(which="major")
    ax_res.grid(which="minor")
    ax_res.tick_params(
        axis="both", labelsize=7, direction="in", top=True, right=True, which="both"
    )


    residuals = []
    i = 0
    to_hist = {
        "longest": {"100": [], "8000": []},
        "greedy_e": {"100": [], "8000": []}
    }

    for path in PATH_NAMES:
        n_seps = defaultdict(list)
        for graph in graphs:
            n_seps[graph["n"]].append(
                np.mean([sum(pos[j]**2 for j in range(1, len(pos))) for pos in graph["paths"][path]])
            )

        if path in to_hist.keys():
            to_hist[path]["100"] = n_seps[100]
            to_hist[path]["8000"] = n_seps[8000]

        x_data = list(n_seps.keys())
        y_data = [np.mean(v) for v in n_seps.values()]
        y_err = [np.std(v) / np.sqrt(len(v)) for v in n_seps.values()]

        cut = 1
        specific = 27
        x_data, y_data, y_err = x_data[cut:], y_data[cut:], y_err[cut:]
        x_data, y_data, y_err = x_data[:-cut], y_data[:-cut], y_err[:-cut]
        del x_data[specific]
        del y_data[specific]
        del y_err[specific]

        if path == "greedy_e":
            ax.errorbar(
                x_data,
                y_data,
                yerr=y_err,
                ls="none",
                capsize=2,
                marker=",",
                # color=colour_map[path],
                label=label_map[path],
                zorder=2,
            )
        else:
            ax.errorbar(
                x_data,
                y_data,
                yerr=y_err,
                ls="none",
                capsize=2,
                marker=",",
                color=colour_map[path] if path not in ["longest", "greedy_e"] else None,
                label=label_map[path],
                zorder=3,
            )

        if path in ["longest", "greedy_e", "greedy_o"]:
            i += 1
            l, legend, y_fit = fit_expo(
                x_data, y_data, y_err, path, params=params[path], ax=ax
            )
            plot_lines.append(l)
            fit_legends.append(legend)
            res = y_fit - np.array(y_data)
            residuals.append(res)
            ax_res.plot(x_data, res, label=f"$\delta_{i}$: {label_map[path]} Fit")
    ax_res.legend(loc="upper left", bbox_to_anchor=(1.02, 0.9))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.875))

    ax.set_xlabel("$N$", fontsize=14)
    ax.set_ylabel(r"$\langle \sigma^2 \rangle$", fontsize=14)
    # ax_res.set_ylim(-0.0017, 0.0017)
    ax.set_yscale("log")
    # ax.set_xscale("log")

    ax_res.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    # ax.xaxis.set_minor_locator(MultipleLocator(5000))
    # ax_res.xaxis.set_minor_locator(MultipleLocator(10000))
    ax_res.set_ylabel("Residuals", fontsize=12, rotation=0, labelpad=30)
    # ax_res.yaxis.set_label_position("right")
    # ax_res.yaxis.tick_right()
    # ax.set_xlim(0, 15150)
    # ax_res.set_xlim(0, 15150)

    # ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.setp(ax_res.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.01)  # remove distance between subplots
    # ax.set_xbound(lower=2000, upper=4000)
    plt.savefig(
        f"images/Mean Separation from Geodesic per Path {d}D.png",
        bbox_inches="tight",
        dpi=1000,
        # facecolor="#F2F2F2",
        transparent=True,
    )
    plt.clf()


    fig = plt.figure(facecolor="#F2F2F2", figsize=(6, 4.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax = fig.add_subplot(gs[1])
    ax_top = fig.add_subplot(gs[0])
    # ax.grid(which="major")
    # ax.grid(which="minor")
    ax.tick_params(
        axis="both", labelsize=12, direction="in", top=True, right=True, which="both"
    )

    # ax_top.grid(which="major")
    # ax_top.grid(which="minor")
    ax_top.tick_params(
        axis="both", labelsize=12, direction="in", top=True, right=True, which="both"
    )

    ax_top.hist(to_hist["longest"]["100"], bins=15, label="N=100")
    ax_top.hist(to_hist["longest"]["8000"], bins=15, label="N=8000")
    ax_top.set_xlim(-0.003, 0.06)
    ax_top.set_ylim(0, 125)
    ax_top.set_xlabel("$\langle \sigma^2\\rangle$", fontsize=13)
    ax_top.set_ylabel("LRGGs", fontsize=13)
    ax_top.set_title("Longest Path", fontsize=13)

    # ax_top.set_yscale("log")

    ax.hist(to_hist["greedy_e"]["100"], bins=15, label="N=100")
    ax.hist(to_hist["greedy_e"]["8000"], bins=15, label="N=8000")
    ax.set_xlim(-0.002, 0.04)
    ax.set_ylim(0, 125)
    ax.set_xlabel("$\langle \sigma^2\\rangle$", fontsize=13)
    ax.set_ylabel("LRGGs", fontsize=13)
    ax.set_title("Greedy Euclidean Path", fontsize=13)
    # ax.set_yscale("log")

    ax.legend()
    ax_top.legend()
    plt.tight_layout()

    plt.savefig(
        f"images/distribution of mss.png",
        bbox_inches="tight",
        dpi=1000,
        # facecolor="#F2F2F2",
        transparent=True,
    )


if __name__ == "__main__":
    d = 4
    graphs = read_file(nrange(100, 15000, 150), 1, d, 320, extra="paths")
    separation_from_geodesic_by_path(graphs, d)

from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit
from scipy.stats import poisson
import math

from src.analysis.an_utils import PATH_NAMES, read_file, fit_expo, fit_expo_poly
from src.utils import nrange, colour_map, label_map


# matplotlib.use("TkAgg")


def separation_from_geodesic_by_path(graphs):
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
            if graph["n"] <= 6000:
            # n_seps[graph["n"]].append(
            #     np.mean([pos[1]**2 for pos in graph["paths"][path]])
            # )
                n_seps[graph["n"]] += [pos[1] ** 2 for pos in graph["paths"][path]]

        x_data = list(n_seps.keys())
        y_data = [np.mean(v) for v in n_seps.values()]
        y_err = [np.std(v) / np.sqrt(len(v)) for v in n_seps.values()]

        # for j, n in enumerate(x_data):
        #     y_weights = []
        #     y_err_weights = []
        #     for i, q in enumerate(y_data):
        #         x_i = x_data[i]
        #         y_weights.append(q*poisson.pmf(x_i, n))
        #         y_err_weights.append((y_err[i]*poisson.pmf(x_i, n))**2)
        #     y_data[j] = sum(y_weights)
        #     y_err[j] = np.sqrt(sum(y_err_weights))
        #
        # cut = 5
        # x_data, y_data, y_err = x_data[cut:], y_data[cut:], y_err[cut:]

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
            l, legend, y_fit = fit_expo(
                x_data, y_data, y_err, path, params=[0.4, -0.2], ax=ax
            )
            plot_lines.append(l)
            fit_legends.append(legend)
            res = y_fit - np.array(y_data)
            residuals.append(res)
            ax_res.plot(x_data, res, label=f"$\delta_{i}$: {label_map[path]} Fit")
    ax_res.legend(loc="upper right", bbox_to_anchor=(-0.02, 0.9))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.875))

    ax.set_xlabel("$N$", fontsize=16)
    ax.set_ylabel(r"$\langle \sigma^2 \rangle$", fontsize=16)
    ax_res.set_ylim(-0.002, 0.002)
    ax.set_yscale("log")
    # ax.set_xscale("log")

    ax_res.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.xaxis.set_minor_locator(MultipleLocator(1000))
    ax_res.xaxis.set_minor_locator(MultipleLocator(1000))
    ax_res.set_ylabel("Residuals", fontsize=12, rotation=0, labelpad=30)
    ax_res.yaxis.set_label_position("right")
    ax_res.yaxis.tick_right()

    plt.setp(ax_res.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=0.01)  # remove distance between subplots
    # ax.set_xbound(lower=2000, upper=4000)
    plt.savefig(
        "images/Mean Separation from Geodesic per Path.png",
        bbox_inches="tight",
        dpi=1000,
        facecolor="#F2F2F2",
    )
    # plt.show()
    plt.clf()

    # plt.hist(residuals[0], bins=15)
    # plt.hist(residuals[1], bins=15)
    # plt.show()
    # fit_legends = []
    # plot_lines = []
    #
    # for path in PATH_NAMES:
    #     n_seps = defaultdict(list)
    #     for graph in graphs:
    #         n_seps[graph["n"]].append(
    #             max([abs(pos[1]) for pos in graph["paths"][path]])
    #         )
    #
    #     x_data = list(n_seps.keys())
    #     y_data = [np.mean(v) for v in n_seps.values()]
    #     y_err = [np.std(v) / np.sqrt(len(v)) for v in n_seps.values()]
    #     plt.errorbar(
    #         x_data,
    #         y_data,
    #         yerr=y_err,
    #         ls="none",
    #         capsize=5,
    #         marker=".",
    #         label=path,
    #     )
    #
    #     if path in ["longest", "greedy_e", "greedy_o"]:
    #         l, legend, y_fit = fit_expo(x_data, y_data, y_err, path, params=[0.4, -0.2])
    #         plot_lines.append(l)
    #         fit_legends.append(legend)
    #
    # legend1 = plt.legend(plot_lines, fit_legends, loc='upper right', bbox_to_anchor=(1, 0.4),
    #                      fancybox=True, shadow=True)
    #
    # plt.legend(loc='lower left', bbox_to_anchor=(-0.08, -0.23, 1, 0.2), ncol=5, columnspacing=0.7,
    #            fancybox=True, shadow=True)
    # plt.gca().add_artist(legend1)
    # plt.title(f"Max Separation from Geodesic per Path")
    # plt.xlabel("Number of Nodes")
    # plt.ylabel("Max Separation from Geodesic")
    # plt.savefig("images/Max Separation from Geodesic per Path.png", bbox_inches="tight", facecolor='#F2F2F2', dpi=1000)
    #
    return residuals, x_data


if __name__ == "__main__":
    res, xs = [], []
    graphs = read_file(nrange(500, 6000, 100), 0.1, 2, 50, extra="paths")
    r, x = separation_from_geodesic_by_path(graphs)
    res += r
    xs.append(x)
    # graphs = read_file(nrange(2000, 4000, 100), 0.1, 2, 50, extra="paths2")
    # r, x = separation_from_geodesic_by_path(graphs)
    # res += r
    # xs.append(x)
    # i = 0
    plt.plot(xs[0], np.abs(res[0]), label="long")
    plt.plot(xs[0], np.abs(res[1]), "--", label="euc")
    plt.tight_layout()
    plt.legend()
    plt.show()

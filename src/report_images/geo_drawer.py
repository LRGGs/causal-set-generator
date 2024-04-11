import matplotlib.pyplot as plt


if __name__ == '__main__':

    fig, ax = plt.subplots(1, 2)
    euc = ax[0]
    mink = ax[1]

    euc.set_xlabel("X - Spatial Coordinate")
    euc.set_ylabel("Y - Spatial Coordinate")
    euc.set_xlim([0, 1])
    euc.set_ylim([0, 1.5])
    euc.set_aspect("equal", adjustable="box")

    mink.set_xlabel("X - Spatial Coordinate")
    mink.set_ylabel("T - Temporal Coordinate")
    mink.set_xlim([0, 1])
    mink.set_ylim([0, 1.5])
    mink.set_aspect("equal", adjustable="box")

    euc1 = plt.Circle((0.3, 0.25), 0.1, color='#1f78b4')
    euc2 = plt.Circle((0.7, 1.25), 0.1, color='#1f78b4')
    euc.plot([0.7, 0.7], [0.25, 1.25], c="g", linewidth=2, zorder=0)
    euc.plot([0.3, 0.7], [0.25, 0.25], c="g", linewidth=2, zorder=0, label="shorter")
    euc.plot([0.3, 0.7], [0.25, 1.25], c="r", linewidth=2, zorder=0, label="longer")
    euc.add_patch(euc1)
    euc.add_patch(euc2)
    euc.annotate("0", xy=(0.3, 0.25), fontsize=14, verticalalignment="center", horizontalalignment="center")
    euc.annotate("1", xy=(0.7, 1.25), fontsize=14, verticalalignment="center", horizontalalignment="center")
    euc.set_title("Euclidean")

    mink1 = plt.Circle((0.3, 0.25), 0.1, color='#1f78b4')
    mink2 = plt.Circle((0.3, 1.25), 0.1, color='#1f78b4')
    mink.plot([0.3, 0.3], [0.25, 1.25], c="g", linewidth=2, zorder=0)
    mink.plot([0.3, 0.8], [0.25, 0.75], c="r", linewidth=2, zorder=0)
    mink.plot([0.8, 0.3], [0.75, 1.25], c="r", linewidth=2, zorder=0)
    mink.add_patch(mink1)
    mink.add_patch(mink2)
    mink.annotate("0", xy=(0.3, 0.25), fontsize=15, verticalalignment="center", horizontalalignment="center")
    mink.annotate("1", xy=(0.3, 1.25), fontsize=15, verticalalignment="center", horizontalalignment="center")
    mink.set_title("Minkowski")

    for a in [euc, mink]:
        for item in ([a.xaxis.label, a.yaxis.label] +
                     a.get_xticklabels() + a.get_yticklabels()):
            item.set_fontsize(14)

        a.title.set_fontsize(17)
    plt.tight_layout()
    # plt.show()
    plt.savefig("../../images/geos.png", transparent=True, dpi=1000, bbox_inches="tight")

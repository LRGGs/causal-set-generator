import matplotlib.pyplot as plt
import numpy as np


def rotate_point(point, angle):
    """Rotate a point (x, y) by angle degrees anticlockwise."""
    x, y = point
    angle_rad = np.deg2rad(angle)
    new_x = (x * np.cos(angle_rad) - y * np.sin(angle_rad))/np.sqrt(2)
    new_y = (x * np.sin(angle_rad) + y * np.cos(angle_rad))/np.sqrt(2)

    return new_x, new_y


if __name__ == '__main__':

    # Number of points
    num_points = 500
    x_coords = np.random.rand(num_points)
    y_coords = np.random.rand(num_points)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0, 1)
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    points = list(zip(x_coords, y_coords))

    # Rotate the points by 90 degrees anticlockwise
    rotated_points = [rotate_point(point, 45) for point in points]

    # Extract x and y coordinates from the rotated points
    x_coords_rotated, y_coords_rotated = zip(*rotated_points)
    ccx, ccy = 0, 0.5
    con_x, con_y, uncon_x, uncon_y = [], [], [], []

    for x, y in zip(x_coords_rotated, y_coords_rotated):
        if (x-ccx)**2 - (y-ccy)**2 < 0 and y > ccy:
            con_x.append(x)
            con_y.append(y)
        else:
            uncon_x.append(x)
            uncon_y.append(y)
    plt.plot(ccx, ccy, "o", color="b", zorder=100)
    plt.plot(con_x, con_y, ".", markersize=4, color="g")
    plt.plot(uncon_x, uncon_y, ".", markersize=2, color="r")

    plt.plot([-0.35, 0.35], [0.85, 0.15], color="black", linewidth=1, zorder=-99)
    plt.plot([-0.35, 0.35], [0.15, 0.85], color="black", linewidth=1, zorder=-99)

    plt.fill_between([-0.25, 0], [0.75, 0.5], [0.75, 1], color="g", alpha=0.2, zorder=-100, linewidth=0.0)
    plt.fill_between([0, 0.25], [0.5, 0.75], [1, 0.75], color="g", alpha=0.2, zorder=-100, linewidth=0.0)

    plt.fill_between([-0.25, 0], [0.25, 0.5], [0.25, 0], color="b", alpha=0.2, zorder=-100, linewidth=0.0)
    plt.fill_between([0, 0.25], [0.5, 0.25], [0, 0.25], color="b", alpha=0.2, zorder=-100, linewidth=0.0)

    plt.fill_between([-0.5, -0.25], [0.5, 0.75], [0.5, 0.25], color="r", alpha=0.2, zorder=-100, linewidth=0.0)
    plt.fill_between([-0.25, 0], [0.75, 0.5], [0.25, 0.5], color="r", alpha=0.2, zorder=-100, linewidth=0.0)

    plt.fill_between([0, 0.25], [0.5, 0.75], [0.5, 0.25], color="r", alpha=0.2, zorder=-100, linewidth=0.0)
    plt.fill_between([0.25, 0.5], [0.75, 0.5], [0.25, 0.5], color="r", alpha=0.2, zorder=-100, linewidth=0.0)

    plt.setp(ax, xlabel='X - Spatial Coordinate')
    plt.setp(ax, ylabel='T - Temporal Coordinate')

    plt.savefig("../../images/conn.png", transparent=True, dpi=1000, bbox_inches="tight")

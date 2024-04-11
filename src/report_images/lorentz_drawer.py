import matplotlib.pyplot as plt
import numpy as np


beta = 0.25
gamma = 1/np.sqrt(1-beta**2)
l_mat = np.array([[gamma, -beta * gamma], [-beta * gamma, gamma]])


def lorentz(coords):
    coords = np.matmul(l_mat, coords)
    return coords


def square_lattice(num_points):
    points = []
    side_length = int(np.sqrt(num_points))
    x_values = np.linspace(0, 1, side_length)
    y_values = np.linspace(0, 1, side_length)
    for x in x_values:
        for y in y_values:
            points.append((x, y))
    return points

def rotate_point(point, angle):
    """Rotate a point (x, y) by angle degrees anticlockwise."""
    x, y = point
    angle_rad = np.deg2rad(angle)
    new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return new_x, new_y


def lorentz_boost(point, v):
    """Apply Lorentz boost to a point (x, y) in the x-direction."""
    c = 1  # Speed of light
    gamma = 1 / np.sqrt(1 - (v**2 / c**2))
    x, y = point
    x_boosted = gamma * (x - v)
    y_boosted = y
    return x_boosted, y_boosted



if __name__ == '__main__':

    fig, ax = plt.subplots(2, 2, figsize=(5.9, 5.5))
    fontsize = 16

    # Number of points
    num_points = 250
    points = square_lattice(num_points)

    # Rotate the points by 90 degrees anticlockwise
    rotated_points = [rotate_point(point, 45) for point in points]

    # Extract x and y coordinates from the rotated points
    x_coords_rotated, y_coords_rotated = zip(*rotated_points)

    v = 0.5  # Choose velocity parameter
    boosted_points = [lorentz(np.array(coords)) for coords in rotated_points]

    boosted_x, boosted_y = zip(*boosted_points)

    custom_xlim = (-1.05, 1.05)
    custom_ylim = (-0.1, 2)

    # Setting the values for all axes.
    plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)

    ax[0, 0].scatter(x_coords_rotated, y_coords_rotated, color='blue', s=1.75)
    ax[0, 1].scatter(boosted_x, boosted_y, color='blue', s=1.75)

    np.random.seed(42)

    # Generate random x and y coordinates
    x_coords = np.random.rand(num_points)
    y_coords = np.random.rand(num_points)

    points = list(zip(x_coords, y_coords))

    rotated_points = [rotate_point(point, 45) for point in points]

    # Extract x and y coordinates from the rotated points
    x_coords_rotated, y_coords_rotated = zip(*rotated_points)

    v = 0.5  # Choose velocity parameter
    boosted_points = [lorentz(np.array(coords)) for coords in rotated_points]

    boosted_x, boosted_y = zip(*boosted_points)

    ax[1, 0].scatter(x_coords_rotated, y_coords_rotated, color='blue', s=1.75)
    ax[1, 1].scatter(boosted_x, boosted_y, color='blue', s=1.75)

    plt.setp(ax[-1, :], xlabel='X - Spatial Coordinate')
    plt.setp(ax[:, 0], ylabel='T - Temporal Coordinate')

    # plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    fig.text(0.3, 0.9, '$\\beta = 0$', ha='center', va='center')
    fig.text(0.72, 0.9, '$\\beta = 0.5$', ha='center', va='center')

    fig.text(0.16, 0.84, '$A$', ha='center', va='center')
    fig.text(0.16, 0.42, '$C$', ha='center', va='center')
    fig.text(0.58, 0.84, '$B$', ha='center', va='center')
    fig.text(0.58, 0.42, '$D$', ha='center', va='center')

    fig.text(0, 0.28, 'Random Uniform', ha='center', va='center', rotation='vertical')
    fig.text(0, 0.705, 'Uniform Lattice', ha='center', va='center', rotation='vertical')

    # plt.show()
    plt.savefig("../../images/lore.png", transparent=True, dpi=1000, bbox_inches="tight")

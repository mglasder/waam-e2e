import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def interp_equidistant(x: np.ndarray, y: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """
    :param x: 1D array-like object containing the x-coordinates of the input curve points.
    :param y: 1D array-like object containing the y-coordinates of the input curve points.
    :param num_points: The number of equally spaced points to be generated along the interpolated curve.
    :return: A tuple of two 1D arrays representing the x-coordinates and y-coordinates of the interpolated curve points.

    This function performs equidistant interpolation of a curve defined by the input points (x, y). It generates a set
    of equally spaced points along the curve using interpolation. The interpolation is done by computing the cumulative
    distance along the curve, normalizing the distances to the range [0, 1], and then applying interpolation along this
    normalized distance. The resulting equally spaced points are returned as arrays of x-coordinates and y-coordinates.

    Example usage:
        x = np.array([0, 1, 3, 6, 10])
        y = np.array([0, 2, 4, 7, 9])
        num_points = 10

        interpolated_x, interpolated_y = interp_equidistant(x, y, num_points)
    """

    # Compute the cumulative distance along the curve
    distance = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0) ** 2 + np.ediff1d(y, to_begin=0) ** 2))
    distance = distance / distance[-1]  # Normalize distance to the range [0, 1]

    fx, fy = interp1d(distance, x), interp1d(distance, y)

    # Generate equally spaced points from 0 to 1
    alpha = np.linspace(0, 1, num_points)

    # Interpolated points
    x_points, y_points = fx(alpha), fy(alpha)

    return x_points, y_points


def use():
    x = np.linspace(-1, 1, 100)
    y = np.sqrt(1 - x**2)
    y2 = np.concatenate([y, -y[::-1]])

    x2 = np.concatenate([x, x[::-1]])

    num_points = 100

    x_points, y_points = interp_equidistant(x2, y2, num_points)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x2, y2, "-o", label="Original points", alpha=0.5)
    plt.plot(x_points, y_points, "r*", label="Interpolated points")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    use()

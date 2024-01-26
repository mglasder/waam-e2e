import numpy as np
from scipy import spatial


def find_nearest_point(pt: np.ndarray, arr_of_points: np.ndarray) -> tuple[np.ndarray, int, float]:
    """
    Find the nearest point in an array of points to a given point.

    :param pt: The point for which we want to find the nearest point.
    :type pt: np.ndarray

    :param arr_of_points: An array of points among which to find the nearest point.
    :type arr_of_points: np.ndarray

    :return: A tuple containing the nearest point, its index in the array, and the distance from the given point.
    :rtype: tuple[np.ndarray, int, float]
    """

    distance, index = spatial.KDTree(arr_of_points).query(pt)
    return arr_of_points[index], index, distance

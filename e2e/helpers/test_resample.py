import numpy as np
from numpy.testing import assert_almost_equal

from e2e.helpers.resample import interp_equidistant


def test_interpolation_num_points():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])

    x_points, y_points = interp_equidistant(x, y, num_points=10)

    assert len(x_points) == 10
    assert len(y_points) == 10


def test_output_values_unchanged():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 0, 0, 0, 0])

    x_points, y_points = interp_equidistant(x, y, num_points=5)

    assert_almost_equal(x_points, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), decimal=5)
    assert_almost_equal(y_points, np.array([0.0, 0.0, 0.0, 0.0, 0.0]), decimal=5)


def test_output_values_downsample():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])

    x_points, y_points = interp_equidistant(x, y, num_points=3)

    assert_almost_equal(x_points, np.array([0.0, 2.0, 4.0]), decimal=5)
    assert_almost_equal(y_points, np.array([0.0, 2.0, 4.0]), decimal=5)


def test_output_values_upsample():
    x = np.array([0, 2, 4])
    y = np.array([0, 2, 4])

    x_points, y_points = interp_equidistant(x, y, num_points=5)

    assert_almost_equal(x_points, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), decimal=5)
    assert_almost_equal(y_points, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), decimal=5)


def test_output_values_interp():
    x = np.array([0, 1.5, 2, 3.5, 4])
    y = np.array([0, 1.5, 2, 3.5, 4])

    x_points, y_points = interp_equidistant(x, y, num_points=5)

    assert_almost_equal(x_points, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), decimal=5)
    assert_almost_equal(y_points, np.array([0.0, 1.0, 2.0, 3.0, 4.0]), decimal=5)


def test_single_point():
    x = np.array([1])
    y = np.array([1])

    x_points, y_points = interp_equidistant(x, y, num_points=1)

    assert x_points[0] == 1
    assert y_points[0] == 1

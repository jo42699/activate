
import pytest

from problems import (
    sphere,
    rosenbrock,
    rastrigin,
    himmelblau,
)


def test_sphere_minimum():
    assert sphere([0, 0]) == pytest.approx(0.0)


def test_sphere_known_value():
    assert sphere([1, 2]) == pytest.approx(5.0)


def test_rosenbrock_minimum():
    assert rosenbrock([1, 1]) == pytest.approx(0.0)


def test_rosenbrock_origin():
    assert rosenbrock([0, 0]) == pytest.approx(1.0)


def test_rastrigin_minimum():
    assert rastrigin([0, 0]) == pytest.approx(0.0)


def test_rastrigin_known_value():
    assert rastrigin([1, 1]) == pytest.approx(2.0)


def test_himmelblau_minimum():
    assert himmelblau([3, 2]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "point",
    [
        [3.0, 2.0],
        [-2.805118, 3.131312],
        [-3.779310, -3.283186],
        [3.584428, -1.848126],
    ],
)
def test_himmelblau_all_known_minima(point):
    assert himmelblau(point) == pytest.approx(0.0, abs=1e-4)

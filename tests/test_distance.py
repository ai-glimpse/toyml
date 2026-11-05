"""Distance properties shared by classification and clustering."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from toyml._distance import euclidean_distance

VECTORS = st.lists(st.integers(-100, 100), min_size=1, max_size=8)


@given(values=VECTORS, exponent=st.integers(-250, 250))
def test_distance_preserves_small_and_large_scales(values: list[int], exponent: int) -> None:
    point = [value * 10.0**exponent for value in values]
    origin = [0.0] * len(point)
    actual = euclidean_distance(origin, point)
    expected = math.dist(origin, point)
    assert actual == pytest.approx(expected, rel=1e-14, abs=0.0)
    assert euclidean_distance(point, origin) == actual
    assert euclidean_distance(point, point) == 0.0


def test_distance_handles_overflowing_coordinate_difference() -> None:
    assert euclidean_distance([-1e308], [1e308]) == math.inf


@given(pairs=st.lists(st.tuples(st.floats(-1e150, 1e150), st.floats(-1e150, 1e150)), min_size=1, max_size=8))
def test_distance_matches_reference_for_arbitrary_points(pairs: list[tuple[float, float]]) -> None:
    first = [x for x, _ in pairs]
    second = [y for _, y in pairs]
    assert euclidean_distance(first, second) == pytest.approx(math.dist(first, second), rel=1e-14, abs=0.0)

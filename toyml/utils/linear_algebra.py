import math
import statistics


def euclidean_distance(v1: list[float], v2: list[float]) -> float:
    """
    Calculate the L2 distance between two vectors

    Example1:
    >>> euclidean_distance([0.0, 0.0], [3.0, 4.0])
    5.0
    """
    assert len(v1) == len(v2), f"{v1} and {v2} have different length!"
    return math.sqrt(sum(pow(v1[i] - v2[i], 2) for i in range(len(v1))))


def sum_square_error(c: list[list[float]]) -> float:
    """
    Calc the sum of squared errors.
    """
    mean_c = [statistics.mean([v[i] for v in c]) for i in range(len(c[0]))]
    return sum(euclidean_distance(mean_c, v) ** 2 for v in c)

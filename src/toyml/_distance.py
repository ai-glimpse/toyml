"""Numerically stable distance calculations shared by the models."""

import math


def euclidean_distance(x: list[float], y: list[float]) -> float:
    r"""Compute a scaled Euclidean norm for validated finite coordinates.

    With $d_i = |x_i - y_i|$ and finite $s = \max_i d_i > 0$:

    $$
    \sqrt{\sum_i d_i^2}
    = \sqrt{s^2 \sum_i (d_i / s)^2}
    = s \sqrt{\sum_i (d_i / s)^2}.
    $$

    Scaling keeps the largest squared term at one; fsum reduces summation error.
    Zero distance is handled before division. An overflowing coordinate difference
    returns infinity rather than evaluating inf / inf.
    """
    differences = [abs(xi - yi) for xi, yi in zip(x, y, strict=True)]
    scale = max(differences, default=0.0)
    if scale == 0.0:
        return 0.0
    if math.isinf(scale):
        return math.inf
    return scale * math.sqrt(math.fsum((difference / scale) ** 2 for difference in differences))

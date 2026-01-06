import math


# data points
points = [
    [1, 1],
    [1, 2],
    [5, 3],
    [6, 4],
]


# euclidean distance
def euclidean(
    x: list[float],
    y: list[float],
) -> float:
    return math.sqrt(sum(pow(x[i] - y[i], 2) for i in range(len(x))))

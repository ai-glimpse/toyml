import math

from .types import DataSet, Vector, Vectors


# Vector operations
def vectors_minus(v1: Vector, v2: Vector) -> Vector:
    """
    v1 - v2
    """
    assert len(v1) == len(
        v2
    ), """Can not minus \
    (vectors have different dimensions)"""
    d = len(v1)
    v = [0.0] * d
    for i in range(d):
        v[i] = v1[i] - v2[i]
    return v


def vectors_mean(vectors: Vectors) -> Vector:
    """
    Calculate vectors mean

    Example1:
    >>> vectors_means([[1.0, 2.0], [3.0, 4.0]])
    [2.0, 3.0]
    """
    d = len(vectors[0])
    n = len(vectors)
    v = [0.0] * d
    for i in range(d):
        v[i] = sum(vector[i] for vector in vectors) / n
    return v


def vectors_std(vectors: Vectors) -> Vector:
    """
    Calculate vectors(every column) standard variance
    """
    d = len(vectors[0])
    n = len(vectors)
    means = vectors_mean(vectors)
    v = [0.0] * d
    for i in range(d):
        v[i] = math.sqrt(sum((vector[i] - means[i]) ** 2 for vector in vectors) / n)
    return v


"""Distance metrics"""


# p = 1
def manhattan_distance(v1: Vector, v2: Vector) -> float:
    diffs = vectors_minus(v1, v2)
    return sum(abs(diff) for diff in diffs)


# p = 2
def euclidean_distance(v1: Vector, v2: Vector) -> float:
    """
    Calculate the L2 distance between two vectors

    Example1:
    >>> euclidean_distance([0.0, 0.0], [3.0, 4.0])
    5.0
    """
    assert len(v1) == len(v2), f"{v1} and {v2} have different length!"
    return math.sqrt(sum(pow(v1[i] - v2[i], 2) for i in range(len(v1))))


def distance_matrix(vectors: Vectors) -> Vectors:
    """
    Get the distance matrix by vectors.
    """
    n = len(vectors)
    dist_mat = [[0.0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(i, n):
            dist_mat[i][j] = euclidean_distance(vectors[i], vectors[j])
            dist_mat[j][i] = dist_mat[i][j]
    return dist_mat


def sse(c: Vectors) -> float:
    """
    Calc the sum of squared errors.
    Used in bisecting k-means.
    """
    mean_c = vectors_mean(c)
    return sum(euclidean_distance(mean_c, v) ** 2 for v in c)


# data transformation
def standarlization(dataset: DataSet, means: Vector, stds: Vector) -> None:
    """
    The standarlization of numerical dataset.
    """
    d = len(means)
    n = len(dataset)
    for j in range(d):
        for i in range(n):
            ele = dataset[i][j]
            dataset[i][j] = (ele - means[j]) / stds[j]

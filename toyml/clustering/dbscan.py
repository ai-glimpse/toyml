from __future__ import annotations

import logging
import random

from collections import deque
from dataclasses import dataclass, field

from toyml.utils.linear_algebra import euclidean_distance

logger = logging.getLogger(__name__)


@dataclass
class DBSCAN:
    """
    DBSCAN algorithm.

    Examples:
        >>> from toyml.clustering import DBSCAN
        >>> dataset = [[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
        >>> dbscan = DBSCAN(eps=3, min_samples=2).fit(dataset)
        >>> dbscan.clusters_
        [[0, 1, 2], [3, 4]]
        >>> dbscan.noises_
        [5]

    Tip: References
        1. Zhou Zhihua
        2. Han
        3. Kassambara
        4. Wikipedia
    """

    eps: float = 0.5
    """The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    This is not a maximum bound on the distances of points within a cluster.
    This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
    (same as sklearn)
    """
    min_samples: int = 5
    """The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    This includes the point itself. If min_samples is set to a higher value,
    DBSCAN will find denser clusters, whereas if it is set to a lower value, the found clusters will be more sparse.
    (same as sklearn)
    """
    clusters_: list[list[int]] = field(default_factory=list)
    core_objects_: list[int] = field(default_factory=list)
    noises_: list[int] = field(default_factory=list)

    distance_matrix_: list[list[float]] = field(default_factory=list)
    n_: int = 0

    def _get_neighbors(self, i: int) -> list[int]:
        neighbors = []
        for j in range(self.n_):
            if i != j and self.distance_matrix_[i][j] <= self.eps:
                neighbors.append(j)
        return neighbors

    def _get_core_objects(self) -> tuple[list[int], list[int]]:
        core_objects, noises = [], []
        for i in range(self.n_):
            if self._is_core_object(i):
                core_objects.append(i)
            else:
                noises.append(i)
        return core_objects, noises

    def fit(self, dataset: list[list[float]]) -> "DBSCAN":
        self.n_ = len(dataset)
        self.distance_matrix_ = distance_matrix(dataset)

        # initialize the unvisited set
        unvisited = set(range(self.n_))
        # get core objects
        self.core_objects_, self.noises_ = self._get_core_objects()
        # core objects used for training
        if len(self.core_objects_) == 0:
            logger.warning("No core objects found, all data points are noise. Try to adjust the hyperparameters.")
            return self
        random.shuffle(self.core_objects_)
        core_object_set = set(self.core_objects_)
        while core_object_set:
            unvisited_old = unvisited.copy()
            core_object = core_object_set.pop()
            queue: deque = deque()
            queue.append(core_object)
            unvisited.remove(core_object)
            while len(queue) > 0:
                q = queue.popleft()
                neighbors = self._get_neighbors(q)
                if self._is_core_object(q, neighbors):
                    delta = set(neighbors) & unvisited
                    # remove from unvisited
                    for point in delta:
                        queue.append(point)
                        unvisited.remove(point)
            cluster = unvisited_old.difference(unvisited)
            self.clusters_.append(list(cluster))
            core_object_set = core_object_set.difference(cluster)
        return self

    def _is_core_object(self, i: int, neighbors: None | list[int] = None) -> bool:
        if neighbors is None:
            neighbors = self._get_neighbors(i)
        # `+ 1` here to include the point itself
        if len(neighbors) + 1 >= self.min_samples:
            return True
        return False


def distance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """
    Get the distance matrix by vectors.
    """
    n = len(vectors)
    dist_mat = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            dist_mat[i][j] = euclidean_distance(vectors[i], vectors[j])
            dist_mat[j][i] = dist_mat[i][j]
    return dist_mat


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
    dbscan = DBSCAN(eps=3, min_samples=2).fit(dataset)
    for i, cluster in enumerate(dbscan.clusters_):
        print(f"cluster {i}: {[dataset[i] for i in cluster]}")
    print("noise: ", [dataset[i] for i in dbscan.noises_])

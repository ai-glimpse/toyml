from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class DBSCAN:
    """DBSCAN algorithm.

    Examples:
        >>> from toyml.clustering import DBSCAN
        >>> dataset = [[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
        >>> dbscan = DBSCAN(eps=3, min_samples=2).fit(dataset)
        >>> dbscan.clusters_
        [[0, 1, 2], [3, 4]]
        >>> dbscan.noises_
        [5]
        >>> dbscan.labels_
        [0, 0, 0, 1, 1, -1]

    Tip: References
        1. Zhou Zhihua
        2. Han
        3. Kassambara
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
    """The clusters found by the DBSCAN algorithm."""
    core_objects_: set[int] = field(default_factory=set)
    """The core objects found by the DBSCAN algorithm."""
    noises_: list[int] = field(default_factory=list)
    """The noises found by the DBSCAN algorithm."""
    labels_: list[int] = field(default_factory=list)
    """The cluster labels found by the DBSCAN algorithm."""

    def fit(self, data: list[list[float]]) -> DBSCAN:
        """Fit the DBSCAN model.

        Args:
            data: The dataset.

        Returns:
            self: The fitted DBSCAN model.
        """
        dataset = Dataset(data)

        # initialize the unvisited set
        unvisited = set(range(dataset.n))
        # get core objects
        self.core_objects_, self.noises_ = dataset.get_core_objects(self.eps, self.min_samples)

        # core objects used for training
        if len(self.core_objects_) == 0:
            logger.warning("No core objects found, all data points are noise. Try to adjust the hyperparameters.")
            return self

        # set of core objects: unordered
        core_object_set = self.core_objects_.copy()
        while core_object_set:
            unvisited_old = unvisited.copy()
            core_object = core_object_set.pop()
            queue: deque[int] = deque()
            queue.append(core_object)
            unvisited.remove(core_object)

            while queue:
                q = queue.popleft()
                neighbors = dataset.get_neighbors(q, self.eps)
                if len(neighbors) + 1 >= self.min_samples:
                    delta = set(neighbors) & unvisited
                    for point in delta:
                        queue.append(point)
                        unvisited.remove(point)

            cluster = unvisited_old - unvisited
            self.clusters_.append(list(cluster))
            core_object_set -= cluster

        self.labels_ = [-1] * dataset.n  # -1 means noise
        for i, cluster_index in enumerate(self.clusters_):
            for j in cluster_index:
                self.labels_[j] = i
        return self

    def fit_predict(self, data: list[list[float]]) -> list[int]:
        """Fit the DBSCAN model and return the cluster labels.

        Args:
            data: The dataset.

        Returns:
            The cluster labels.
        """
        return self.fit(data).labels_


@dataclass
class Dataset:
    """Dataset for DBSCAN.

    Args:
        data: The dataset.

    Attributes:
        data: The dataset.
        n: The number of data points.
        distance_matrix_: The distance matrix.
    """

    data: list[list[float]]
    n: int = field(init=False)
    distance_metric: Literal["euclidean"] = "euclidean"
    """The distance metric to use.(For now we only support euclidean)."""
    distance_matrix_: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.n = len(self.data)
        self.distance_matrix_ = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self) -> list[list[float]]:
        dist_mat = [[0.0 for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i, self.n):
                dist_mat[i][j] = self._get_distance(self.data[i], self.data[j])
                dist_mat[j][i] = dist_mat[i][j]
        return dist_mat

    def get_neighbors(self, i: int, eps: float) -> list[int]:
        """Get the neighbors of the i-th data point.

        Args:
            i: The index of the data point.
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.

        Returns:
            The indices of the neighbors (Don't include the point itself).
        """
        return [j for j in range(self.n) if i != j and self.distance_matrix_[i][j] <= eps]

    def get_core_objects(self, eps: float, min_samples: int) -> tuple[set[int], list[int]]:
        """Get the core objects and noises of the dataset.

        Args:
            eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            min_samples: The number of samples (or total weight) in a neighborhood
            for a point to be considered as a core point.

        Returns:
            core_objects: The indices of the core objects.
            noises: The indices of the noises.
        """
        core_objects = set()
        noises = []
        for i in range(self.n):
            neighbors = self.get_neighbors(i, eps)
            if len(neighbors) + 1 >= min_samples:  # +1 to include the point itself
                core_objects.add(i)
            else:
                noises.append(i)
        return core_objects, noises

    def _get_distance(self, x: list[float], y: list[float]) -> float:
        assert len(x) == len(y), f"{x} and {y} have different length!"
        if self.distance_metric == "euclidean":
            return math.sqrt(sum(pow(x[i] - y[i], 2) for i in range(len(x))))
        msg = f"Distance metric {self.distance_metric} not supported!"
        raise ValueError(msg)


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]]
    dbscan = DBSCAN(eps=3, min_samples=2).fit(dataset)
    for i, cluster in enumerate(dbscan.clusters_):
        print(f"cluster {i}: {[dataset[i] for i in cluster]}")
    print("noise: ", [dataset[i] for i in dbscan.noises_])

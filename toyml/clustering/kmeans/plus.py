from __future__ import annotations

import math
import random

from toyml.clustering.kmeans.simple import Kmeans
from toyml.utils.linear_algebra import euclidean_distance
from toyml.utils.types import DataSet, Vector


class KmeansPlus(Kmeans):
    """
    The implementation of k-means++ algorithm
    """

    def _get_min_sq_dist(self, point: Vector) -> float:
        min_sq_dist = math.inf
        assert self.centroids_ is not None
        for centroid in self.centroids_:
            if centroid != []:
                sq_dist = euclidean_distance(point, centroid) ** 2
                if sq_dist < min_sq_dist:
                    min_sq_dist = sq_dist
        return min_sq_dist

    def get_initial_centroids(self, dataset: list[list[float]]) -> DataSet:
        # get initial centroids by k-means++ algorithm
        # the first centroid
        self.centroids_ = [[] for _ in range(self.k)]
        self.centroids_[0] = random.choice(dataset)
        for i in range(1, self.k):
            min_distances = [self._get_min_sq_dist(point) for point in dataset]
            total_dist = sum(min_distances)
            weights = [dist / total_dist for dist in min_distances]
            self.centroids_[i] = random.choices(dataset, weights)[0]
        return self.centroids_


if __name__ == "__main__":
    dataset: DataSet = [[1.0, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
    k: int = 2
    # kmeans++
    print("Test K-means++...")
    kmeans_plus = KmeansPlus(k)
    kmeans_plus.fit(dataset)
    print(kmeans_plus.predict([0.0, 0.0]))

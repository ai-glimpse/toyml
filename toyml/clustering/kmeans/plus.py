from __future__ import annotations

import math
import random

from toyml.clustering.kmeans.simple import Kmeans
from toyml.utils.linear_algebra import euclidean_distance


class KmeansPlus(Kmeans):
    """
    The implementation of k-means++ algorithm
    """

    def get_initial_centroids(self, dataset: list[list[float]]) -> list[list[float]]:
        """
        Get initial centroids by k-means++ algorithm.

        Args:
            dataset: The dataset for clustering

        Returns:
            The initial centroids
        """
        # the first centroid
        self.centroids_ = [[] for _ in range(self.k)]
        self.centroids_[0] = random.choice(dataset)
        for i in range(1, self.k):
            min_distances = [self._get_min_square_distance(point) for point in dataset]
            total_dist = sum(min_distances)
            weights = [dist / total_dist for dist in min_distances]
            self.centroids_[i] = random.choices(dataset, weights)[0]
        return self.centroids_

    def _get_min_square_distance(self, point: list[float]) -> float:
        """
        Get the minimum square distance from the point to current centroids.

        Args:
            point: The point to calculate the distance.

        Returns:
            The minimum square distance
        """
        min_square_distance = math.inf
        assert self.centroids_ is not None
        for centroid in self.centroids_:
            if len(centroid) > 0:
                square_distance = euclidean_distance(point, centroid) ** 2
                if square_distance < min_square_distance:
                    min_square_distance = square_distance
        return min_square_distance


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
    k: int = 2
    # kmeans++
    print("Test K-means++...")
    kmeans_plus = KmeansPlus(k)
    kmeans_plus.fit(dataset)
    print(kmeans_plus.predict([0.0, 0.0]))

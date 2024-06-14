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
        for centroid in self._centroids:
            if centroid != []:
                sq_dist = euclidean_distance(point, centroid) ** 2
                if sq_dist < min_sq_dist:
                    min_sq_dist = sq_dist
        return min_sq_dist

    def _get_initial_centroids(self) -> DataSet:
        # get initial centroids by k-means++ algorithm
        # the first centroid
        self._centroids[0] = random.choice(self._dataset)
        for i in range(1, self._k):
            min_distances = [self._get_min_sq_dist(point) for point in self._dataset]
            total_dist = sum(min_distances)
            weights = [dist / total_dist for dist in min_distances]
            self._centroids[i] = random.choices(self._dataset, weights)[0]
        return self._centroids


if __name__ == "__main__":
    dataset: DataSet = [[1.0, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
    k: int = 2
    # kmeans++
    print("Test K-means++...")
    kmeans_plus = KmeansPlus(dataset, k)
    kmeans_plus.fit()
    kmeans_plus.print_cluster()
    kmeans_plus.predict([0.0, 0.0])

import math
import random

from typing import Any, Tuple

from toyml.utils.linear_algebra import euclidean_distance
from toyml.utils.types import Clusters, DataSet, Vector

"""
TODO:
1. Plot
2. Test
"""


class Kmeans:
    """
    K-means algorithm.

    Refrences:
    1. Zhou Zhihua
    2. Murphy

    Note: Here we just code the naive K-means and K-means++ algorithms.
    We implement the Bisecting K-means algorithm in toyml.hierarchical.Diana.
    """

    def __init__(self, dataset: DataSet, k: int, max_iter: int = 500) -> None:
        """
        dataset: the set of data points for clustering
        k: the number of clusters, specified by user
        max_iter: The number of iterations the algorithm will run
        for if it does not converge before that.
        """
        self._dataset: DataSet = dataset
        self._k: int = k
        self._max_iter: int = max_iter
        # results
        self._centroids: Any = [[] for i in range(self._k)]
        self._clusters: Clusters = [[]]

    def _get_initial_centroids(self) -> DataSet:
        """
        get initial centroids by a simple random selection
        """
        return random.sample(self._dataset, self._k)

    def _get_centroid_label(self, point: Vector, centroids: DataSet) -> int:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        return distances.index(min(distances))

    def _get_clusters(self, centroids: DataSet) -> Clusters:
        clusters = [[] for i in range(self._k)]
        for i, point in enumerate(self._dataset):
            centroid_label = self._get_centroid_label(point, centroids)
            clusters[centroid_label].append(i)
        return clusters

    def _get_centroids(self, clusters: Clusters) -> DataSet:
        # clusters: indexes -> data points
        points_clusters = [[self._dataset[i] for i in cluster] for cluster in clusters]
        centroids = [[] for i in range(self._k)]
        for i, cluster in enumerate(points_clusters):
            centroid = [sum(t) / self._k for t in zip(*cluster)]
            centroids[i] = centroid
        return centroids

    def fit(self) -> Tuple[DataSet, Clusters]:
        centroids = self._get_initial_centroids()
        clusters = self._get_clusters(centroids)  # just for unbound error
        for _ in range(self._max_iter):
            clusters = self._get_clusters(centroids)
            prev_centroids = centroids
            centroids = self._get_centroids(clusters)
            # If no centroids change, the algorithm is convergent
            if prev_centroids == centroids:
                print("Training Converged")
                break
        self._centroids = centroids
        self._clusters = clusters
        return (centroids, clusters)

    def predict(self, point: Vector) -> int:
        return self._get_centroid_label(point, self._centroids)

    def print_cluster(self) -> None:
        for i in range(self._k):
            print(f"label({i}) -> {self._centroids[i]}: {self._clusters[i]}")

    def print_labels(self) -> None:
        y_pred = [0] * len(self._dataset)
        for cluster_index in range(self._k):
            for sample_index in self._clusters[cluster_index]:
                y_pred[sample_index] = cluster_index
        print("Sample labels: ", y_pred)


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
    dataset = [[1.0, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
    k = 2
    # kmeans
    print("Test K-means...")
    kmeans = Kmeans(dataset, k)
    kmeans.fit()
    kmeans.print_cluster()
    kmeans.predict([0.0, 0.0])
    # kmeans++
    print("Test K-means++...")
    kmeansplus = KmeansPlus(dataset, k)
    kmeansplus.fit()
    kmeansplus.print_cluster()
    kmeansplus.predict([0.0, 0.0])

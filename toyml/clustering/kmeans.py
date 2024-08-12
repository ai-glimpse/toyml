from __future__ import annotations

import math
import random

from dataclasses import dataclass
from typing import Literal, Optional

from toyml.utils.linear_algebra import euclidean_distance


@dataclass
class Kmeans:
    """
    K-means algorithm (with Kmeans++ initialization as option).

    Examples:
        >>> from toyml.clustering import Kmeans
        >>> dataset = [[1, 0], [1, 1], [1, 2], [10, 0], [10, 1], [10, 2]]
        >>> kmeans = Kmeans(k=2).fit(dataset)
        >>> kmeans.clusters_
        {0: [0, 1, 2], 1: [3, 4, 5]}
        >>> kmeans.centroids_
        {0: [1.0, 1.0], 1: [10.0, 1.0]}
        >>> kmeans.predict([0, 1])
        0
        >>> kmeans.iter_
        2

    Tip: References
        1. Zhou Zhihua
        2. Murphy

    Note:
        Here we just implement the naive K-means algorithm.

    See Also:
      - Bisecting K-means algorithm: [toyml.clustering.kmeans.bisecting][kmeans-bisect]
    """

    k: int
    """The number of clusters, specified by user."""
    max_iter: int = 500
    """The number of iterations the algorithm will run for if it does not converge before that."""
    tol: float = 1e-5
    """The tolerance for convergence."""
    centroids_init_method: Literal["random", "kmeans++"] = "random"
    """The method to initialize the centroids."""
    iter_: int = 0
    clusters_: Optional[dict[int, list[int]]] = None
    """The clusters of the dataset."""
    centroids_: Optional[dict[int, list[float]]] = None
    """The centroids of the clusters."""

    def fit(self, dataset: list[list[float]]) -> "Kmeans":
        """
        Args:
            dataset: the set of data points for clustering

        Returns:
            self.
        """
        centroids = self._get_initial_centroids(dataset)
        clusters = dict()
        for _ in range(self.max_iter):
            self.iter_ += 1
            clusters = self._get_clusters(dataset, centroids)
            prev_centroids = centroids
            centroids = self._get_centroids(dataset, clusters)
            if self._is_converged(prev_centroids, centroids):
                break

        self.clusters_ = clusters
        self.centroids_ = centroids
        return self

    def predict(self, point: list[float]) -> int:
        """
        Predict the label of the point.

        Args:
            point: The data point to predict.

        Returns:
            The label of the point.

        """
        if self.centroids_ is None:
            raise ValueError("The model is not fitted yet")
        return self._get_centroid_label(point, self.centroids_)

    def _get_initial_centroids(self, dataset: list[list[float]]) -> dict[int, list[float]]:
        """
        get initial centroids by a simple random selection
        """
        if self.centroids_init_method == "random":
            return self._get_initial_centroids_random(dataset)
        elif self.centroids_init_method == "kmeans++":
            return self._get_initial_centroids_kmeans_plus(dataset)
        else:
            raise ValueError(f"Invalid centroids initialization method: {self.centroids_init_method}")

    def _is_converged(self, prev_centroids: dict[int, list[float]], centroids: dict[int, list[float]]) -> bool:
        """
        Check if the centroids converged.

        Args:
            prev_centroids: previous centroids
            centroids: current centroids

        Returns:
            Whether the centroids converged.
        """
        # check every centroid
        for i, centroid in centroids.items():
            prev_centroid = prev_centroids[i]
            # check every dimension
            for j in range(len(prev_centroid)):
                if abs(prev_centroid[j] - centroid[j]) > self.tol:
                    return False
        return True

    def _get_initial_centroids_random(self, dataset: list[list[float]]) -> dict[int, list[float]]:
        """
        Get initial centroids by a simple random selection.

        Args:
            dataset: The dataset for clustering

        Returns:
            The initial centroids
        """
        data_points = random.sample(dataset, self.k)
        centroids = {i: data_point for i, data_point in enumerate(data_points)}
        return centroids

    def _get_initial_centroids_kmeans_plus(self, dataset: list[list[float]]) -> dict[int, list[float]]:
        """
        Get initial centroids by k-means++ algorithm.

        Args:
            dataset: The dataset for clustering

        Returns:
            The initial centroids
        """
        self.centroids_ = dict()
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
        assert self.centroids_ is not None
        return min(
            (euclidean_distance(point, centroid) ** 2 for centroid in self.centroids_.values() if centroid),
            default=math.inf,
        )

    @staticmethod
    def _get_centroid_label(point: list[float], centroids: dict[int, list[float]]) -> int:
        """
        Get the label of the centroid, which is closest to the point
        """
        distances = [(i, euclidean_distance(point, centroid)) for i, centroid in centroids.items()]
        return min(distances, key=lambda x: x[1])[0]

    def _get_clusters(self, dataset: list[list[float]], centroids: dict[int, list[float]]) -> dict[int, list[int]]:
        clusters: dict[int, list[int]] = {i: [] for i in range(self.k)}
        for i, point in enumerate(dataset):
            centroid_label = self._get_centroid_label(point, centroids)
            clusters[centroid_label].append(i)
        return clusters

    def _get_centroids(self, dataset: list[list[float]], clusters: dict[int, list[int]]) -> dict[int, list[float]]:
        centroids: dict[int, list[float]] = {i: [] for i in range(self.k)}
        for cluster_i, cluster in clusters.items():
            cluster_points = [dataset[i] for i in cluster]
            centroid = [sum(t) / len(cluster) for t in zip(*cluster_points)]
            centroids[cluster_i] = centroid
        return centroids

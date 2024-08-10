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
        [[3, 4, 5], [0, 1, 2]]
        >>> kmeans.centroids_
        [[10.0, 1.0], [1.0, 1.0]]
        >>> kmeans.predict([0, 1])
        1

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
    clusters_: Optional[list[list[int]]] = None
    """The clusters of the dataset."""
    centroids_: Optional[list[list[float]]] = None
    """The centroids of the clusters."""

    def get_initial_centroids(self, dataset: list[list[float]]) -> list[list[float]]:
        """
        get initial centroids by a simple random selection
        """
        if self.centroids_init_method == "random":
            return self._get_initial_centroids_random(dataset)
        elif self.centroids_init_method == "kmeans++":
            return self._get_initial_centroids_kmeans_plus(dataset)
        else:
            raise ValueError(f"Invalid centroids initialization method: {self.centroids_init_method}")

    def fit(self, dataset: list[list[float]]) -> "Kmeans":
        """
        Args:
            dataset: the set of data points for clustering

        Returns:
            self.
        """
        centroids = self.get_initial_centroids(dataset)
        clusters = []
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

    def _is_converged(self, prev_centroids: list[list[float]], centroids: list[list[float]]) -> bool:
        """
        Check if the centroids converged.

        Args:
            prev_centroids: previous centroids
            centroids: current centroids

        Returns:
            Whether the centroids converged.
        """
        # check every centroid
        for i, centroid in enumerate(centroids):
            prev_centroid = prev_centroids[i]
            # check every dimension
            for j in range(len(prev_centroid)):
                if abs(prev_centroid[j] - centroid[j]) > self.tol:
                    return False
        return True

    def _get_initial_centroids_random(self, dataset: list[list[float]]) -> list[list[float]]:
        """
        Get initial centroids by a simple random selection.

        Args:
            dataset: The dataset for clustering

        Returns:
            The initial centroids
        """
        return random.sample(dataset, self.k)

    def _get_initial_centroids_kmeans_plus(self, dataset: list[list[float]]) -> list[list[float]]:
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
        assert self.centroids_ is not None
        return min(
            (euclidean_distance(point, centroid) ** 2 for centroid in self.centroids_ if centroid), default=math.inf
        )

    @staticmethod
    def _get_centroid_label(point: list[float], centroids: list[list[float]]) -> int:
        """
        Get the label of the centroid, which is closest to the point
        """
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        return distances.index(min(distances))

    def _get_clusters(self, dataset: list[list[float]], centroids: list[list[float]]) -> list[list[int]]:
        clusters: list[list[int]] = [[] for _ in range(self.k)]
        for i, point in enumerate(dataset):
            centroid_label = self._get_centroid_label(point, centroids)
            clusters[centroid_label].append(i)
        return clusters

    def _get_centroids(self, dataset: list[list[float]], clusters: list[list[int]]) -> list[list[float]]:
        # clusters: indexes -> data points
        points_clusters = [[dataset[i] for i in cluster] for cluster in clusters]
        centroids: list[list[float]] = [[] for _ in range(self.k)]
        for i, cluster in enumerate(points_clusters):
            centroid = [sum(t) / len(cluster) for t in zip(*cluster)]
            centroids[i] = centroid
        return centroids

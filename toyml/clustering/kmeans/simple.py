from __future__ import annotations

import random

from dataclasses import dataclass
from typing import Optional

from toyml.utils.linear_algebra import euclidean_distance


@dataclass
class Kmeans:
    """
    Naive K-means algorithm.

    References:
    1. Zhou Zhihua
    2. Murphy

    Note:
        Here we just implement the naive K-means algorithm.

    See Also:
      - K-means++ algorithm: [toyml.clustering.kmeans.plus.KmeansPlus][]
      - Bisecting K-means algorithm: [toyml.clustering.kmeans.bisecting][kmeans-bisect]
    """

    k: int
    """The number of clusters, specified by user."""
    max_iter: int = 500
    """The number of iterations the algorithm will run for if it does not converge before that."""
    clusters_: Optional[list[list[int]]] = None
    """The clusters of the dataset."""
    centroids_: Optional[list[list[float]]] = None
    """The centroids of the clusters."""

    def get_initial_centroids(self, dataset: list[list[float]]) -> list[list[float]]:
        """
        get initial centroids by a simple random selection
        """
        return random.sample(dataset, self.k)

    @staticmethod
    def get_centroid_label(point: list[float], centroids: list[list[float]]) -> int:
        """
        Get the label of the centroid, which is closest to the point
        """
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        return distances.index(min(distances))

    def _get_clusters(self, dataset: list[list[float]], centroids: list[list[float]]) -> list[list[int]]:
        clusters: list[list[int]] = [[] for _ in range(self.k)]
        for i, point in enumerate(dataset):
            centroid_label = self.get_centroid_label(point, centroids)
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

    def fit(self, dataset: list[list[float]]) -> "Kmeans":
        """
        Args:
            dataset: the set of data points for clustering

        Returns:

        """
        centroids = self.get_initial_centroids(dataset)
        clusters = []
        for _ in range(self.max_iter):
            clusters = self._get_clusters(dataset, centroids)
            prev_centroids = centroids
            centroids = self._get_centroids(dataset, clusters)
            # If no centroids change, the algorithm is convergent
            # TODO: better convergence criteria
            if prev_centroids == centroids:
                print("Training Converged")
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
        return self.get_centroid_label(point, self.centroids_)


if __name__ == "__main__":
    # Create a more diverse dataset with clear clusters
    dataset: list[list[float]] = [
        [1, 2],
        [2, 1],
        [2, 3],
        [1, 4],
        [3, 2],  # Cluster 1
        [8, 7],
        [9, 8],
        [7, 9],
        [8, 10],
        [10, 8],  # Cluster 2
        [15, 15],
        [16, 14],
        [14, 16],
        [15, 17],
        [17, 15],  # Cluster 3
    ]
    k: int = 3

    print("Test K-means...")
    kmeans = Kmeans(k)
    kmeans.fit(dataset)

    print("Clusters:")
    assert kmeans.clusters_ is not None
    for i, cluster in enumerate(kmeans.clusters_):
        print(f"Cluster {i + 1}: {cluster}")

    print("\nCentroids:")
    assert kmeans.centroids_ is not None
    for i, centroid in enumerate(kmeans.centroids_):
        print(f"Centroid {i + 1}: {centroid}")

    # Test prediction
    test_point = [12.0, 13.0]
    predicted_cluster = kmeans.predict(test_point)
    print(f"\nPredicted cluster for point {test_point}: {predicted_cluster + 1}")

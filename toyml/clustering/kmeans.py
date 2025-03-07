from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class Kmeans:
    """K-means algorithm (with Kmeans++ initialization as option).

    Examples:
        >>> from toyml.clustering import Kmeans
        >>> dataset = [[1.0, 2.0], [1.0, 4.0], [1.0, 0.0], [10.0, 2.0], [10.0, 4.0], [11.0, 0.0]]
        >>> kmeans = Kmeans(k=2, random_seed=42).fit(dataset)
        >>> kmeans.clusters_
        {0: [3, 4, 5], 1: [0, 1, 2]}
        >>> kmeans.centroids_
        {0: [10.333333333333334, 2.0], 1: [1.0, 2.0]}
        >>> kmeans.labels_
        [1, 1, 1, 0, 0, 0]
        >>> kmeans.predict([0, 1])
        1
        >>> kmeans.iter_
        2

    There is a `fit_predict` method that can be used to fit and predict.

    Examples:
        >>> from toyml.clustering import Kmeans
        >>> dataset = [[1, 0], [1, 1], [1, 2], [10, 0], [10, 1], [10, 2]]
        >>> Kmeans(k=2, random_seed=42).fit_predict(dataset)
        [1, 1, 1, 0, 0, 0]

    Tip: References
        1. Zhou Zhihua
        2. Murphy

    Note:
        Here we just implement the naive K-means algorithm.

    See Also:
      - Bisecting K-means algorithm: [toyml.clustering.bisect_kmeans][kmeans-bisect]
    """

    k: int
    """The number of clusters, specified by user."""
    max_iter: int = 500
    """The number of iterations the algorithm will run for if it does not converge before that."""
    tol: float = 1e-5
    """The tolerance for convergence."""
    centroids_init_method: Literal["random", "kmeans++"] = "random"
    """The method to initialize the centroids."""
    random_seed: int | None = None
    """The random seed used to initialize the centroids."""
    distance_metric: Literal["euclidean"] = "euclidean"
    """The distance metric to use.(For now we only support euclidean)."""
    iter_: int = 0
    clusters_: dict[int, list[int]] = field(default_factory=dict)
    """The clusters of the dataset."""
    centroids_: dict[int, list[float]] = field(default_factory=dict)
    """The centroids of the clusters."""
    labels_: list[int] = field(default_factory=list)
    """The cluster labels of the dataset."""

    def __post_init__(self) -> None:
        random.seed(self.random_seed)

    def fit(self, dataset: list[list[float]]) -> Kmeans:
        """Fit the dataset with K-means algorithm.

        Args:
            dataset: the set of data points for clustering

        Returns:
            self.
        """
        self.centroids_ = self._get_initial_centroids(dataset)
        for _ in range(self.max_iter):
            self.iter_ += 1
            prev_centroids = self.centroids_
            self._iter_step(dataset)
            if self._is_converged(prev_centroids):
                break
        return self

    def _iter_step(self, dataset: list[list[float]]) -> None:
        """Can be used to control the fitting process step by step."""
        self.clusters_ = self._get_clusters(dataset)
        self.centroids_ = self._get_centroids(dataset)
        self.labels_ = self._predict_dataset_labels(dataset)

    def fit_predict(self, dataset: list[list[float]]) -> list[int]:
        """Fit and predict the cluster label of the dataset.

        Args:
            dataset: the set of data points for clustering

        Returns:
            Cluster labels of the dataset samples.
        """
        return self.fit(dataset).labels_

    def predict(self, point: list[float]) -> int:
        """Predict the label of the point.

        Args:
            point: The data point to predict.

        Returns:
            The label of the point.

        """
        if len(self.centroids_) == 0:
            msg = "The model is not fitted yet"
            raise ValueError(msg)
        return self._get_point_centroid_label(point, self.centroids_)

    def _calc_distance(self, x: list[float], y: list[float]) -> float:
        assert len(x) == len(y), f"{x} and {y} have different length!"
        if self.distance_metric == "euclidean":
            return math.sqrt(sum(pow(x[i] - y[i], 2) for i in range(len(x))))
        msg = f"Distance metric {self.distance_metric} not supported!"
        raise ValueError(msg)

    def _predict_dataset_labels(self, dataset: list[list[float]]) -> list[int]:
        labels = [-1] * len(dataset)
        for cluster_label, cluster in self.clusters_.items():
            for data_point_index in cluster:
                labels[data_point_index] = cluster_label
        return labels

    def _get_initial_centroids(self, dataset: list[list[float]]) -> dict[int, list[float]]:
        """Get initial centroids by a simple random selection."""
        if self.centroids_init_method == "random":
            return self._get_initial_centroids_random(dataset)
        if self.centroids_init_method == "kmeans++":
            return self._get_initial_centroids_kmeans_plus(dataset)
        msg = f"Invalid centroids initialization method: {self.centroids_init_method}"
        raise ValueError(msg)

    def _is_converged(self, prev_centroids: dict[int, list[float]]) -> bool:
        """Check if the centroids converged.

        Args:
            prev_centroids: previous centroids

        Returns:
            Whether the centroids converged.
        """
        # check every centroid
        for i, centroid in self.centroids_.items():
            prev_centroid = prev_centroids[i]
            # check every dimension
            for j in range(len(prev_centroid)):
                if abs(prev_centroid[j] - centroid[j]) > self.tol:
                    return False
        return True

    def _get_initial_centroids_random(self, dataset: list[list[float]]) -> dict[int, list[float]]:
        """Get initial centroids by a simple random selection.

        Args:
            dataset: The dataset for clustering

        Returns:
            The initial centroids
        """
        data_points = random.sample(dataset, self.k)
        centroids = dict(enumerate(data_points))
        return centroids

    def _get_initial_centroids_kmeans_plus(self, dataset: list[list[float]]) -> dict[int, list[float]]:
        """Get initial centroids by k-means++ algorithm.

        Args:
            dataset: The dataset for clustering

        Returns:
            The initial centroids
        """
        self.centroids_ = {}
        self.centroids_[0] = random.choice(dataset)
        for i in range(1, self.k):
            min_distances = [self._get_min_square_distance(point) for point in dataset]
            total_dist = sum(min_distances)
            weights = [dist / total_dist for dist in min_distances]
            self.centroids_[i] = random.choices(dataset, weights)[0]
        return self.centroids_

    def _get_min_square_distance(self, point: list[float]) -> float:
        """Get the minimum square distance from the point to current centroids.

        Args:
            point: The point to calculate the distance.

        Returns:
            The minimum square distance
        """
        return min(
            (self._calc_distance(point, centroid) ** 2 for centroid in self.centroids_.values() if centroid),
            default=math.inf,
        )

    def _get_point_centroid_label(self, point: list[float], centroids: dict[int, list[float]]) -> int:
        """Get the label of the centroid, which is closest to the point."""
        distances = [(i, self._calc_distance(point, centroid)) for i, centroid in centroids.items()]
        return min(distances, key=lambda x: x[1])[0]

    def _get_clusters(self, dataset: list[list[float]]) -> dict[int, list[int]]:
        clusters: dict[int, list[int]] = {i: [] for i in range(self.k)}
        for i, point in enumerate(dataset):
            centroid_label = self._get_point_centroid_label(point, self.centroids_)
            clusters[centroid_label].append(i)
        return clusters

    def _get_centroids(self, dataset: list[list[float]]) -> dict[int, list[float]]:
        centroids: dict[int, list[float]] = {i: [] for i in range(self.k)}
        for cluster_i, cluster in self.clusters_.items():
            cluster_points = [dataset[i] for i in cluster]
            centroid = [sum(t) / len(cluster) for t in zip(*cluster_points, strict=False)]
            centroids[cluster_i] = centroid
        return centroids

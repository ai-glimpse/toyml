import logging
import math
import random
from dataclasses import dataclass, field
from typing import Literal

from toyml._distance import euclidean_distance

logger = logging.getLogger(__name__)


@dataclass
class KMeans:
    """K-means algorithm (with KMeans++ initialization as option).

    Examples:
        >>> from toyml.clustering import KMeans
        >>> dataset = [[1.0, 2.0], [1.0, 4.0], [1.0, 0.0], [10.0, 2.0], [10.0, 4.0], [11.0, 0.0]]
        >>> kmeans = KMeans(k=2, random_seed=42).fit(dataset)
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
        >>> from toyml.clustering import KMeans
        >>> dataset = [[1, 0], [1, 1], [1, 2], [10, 0], [10, 1], [10, 2]]
        >>> KMeans(k=2, random_seed=42).fit_predict(dataset)
        [1, 1, 1, 0, 0, 0]

    Tip: References
        1. Zhou Zhihua
        2. Murphy

    Note:
        Here we just implement the naive K-means algorithm.

    See Also:
      - Bisecting K-means algorithm: [toyml.clustering.bisecting_kmeans][kmeans-bisect]
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
    """Integer seeds reproduce each fit; None keeps an independent random stream."""
    distance_metric: Literal["euclidean"] = "euclidean"
    """The distance metric to use.(For now we only support euclidean)."""
    iter_: int = field(default=0, init=False)
    clusters_: dict[int, list[int]] = field(default_factory=dict, init=False)
    """The clusters of the dataset."""
    centroids_: dict[int, list[float]] = field(default_factory=dict, init=False)
    """The centroids of the clusters."""
    labels_: list[int] = field(default_factory=list, init=False)
    """The cluster labels of the dataset."""

    def __post_init__(self) -> None:
        self.random_state = random.Random(self.random_seed)

    def fit(self, dataset: list[list[float]]) -> KMeans:
        """Fit the dataset with K-means algorithm.

        Args:
            dataset: the set of data points for clustering

        Returns:
            self.
        """
        self._validate_input(dataset)
        # Replay integer-seeded fits; unseeded instances keep advancing their private stream.
        if self.random_seed is not None:
            self.random_state.seed(self.random_seed)
        self.iter_ = 0
        self.centroids_ = self._get_initial_centroids(dataset)
        for _ in range(self.max_iter):
            self.iter_ += 1
            prev_centroids = self.centroids_
            self._iter_step(dataset)
            if self._is_converged(prev_centroids):
                break
        # Expose assignments to the final centers, including after an iteration limit.
        self.clusters_ = self._get_clusters(dataset)
        self.labels_ = self._predict_dataset_labels(dataset)
        nonempty = sum(bool(cluster) for cluster in self.clusters_.values())
        if nonempty < self.k:
            logger.warning("Found %d non-empty clusters, fewer than requested k=%d", nonempty, self.k)
        return self

    def _validate_input(self, dataset: list[list[float]]) -> None:
        if not dataset or not dataset[0]:
            msg = "The dataset must contain at least one sample and one feature"
            raise ValueError(msg)
        if isinstance(self.k, bool) or not isinstance(self.k, int) or not 1 <= self.k <= len(dataset):
            msg = "k must be an integer between 1 and the sample count"
            raise ValueError(msg)
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int) or self.max_iter < 1:
            msg = "max_iter must be a positive integer"
            raise ValueError(msg)
        if not isinstance(self.tol, (int, float)) or not math.isfinite(self.tol) or self.tol < 0:
            msg = "tol must be finite and non-negative"
            raise ValueError(msg)
        if any(len(row) != len(dataset[0]) for row in dataset):
            msg = "All samples must have the same dimension"
            raise ValueError(msg)
        try:
            finite = all(isinstance(x, (int, float)) and math.isfinite(x) for row in dataset for x in row)
        except OverflowError:
            finite = False
        if not finite:
            msg = "Features must be finite numeric values"
            raise ValueError(msg)

    def _iter_step(self, dataset: list[list[float]]) -> None:
        """Can be used to control the fitting process step by step."""
        self.clusters_ = self._get_clusters(dataset)
        self._relocate_empty_clusters(dataset)
        self.centroids_ = self._get_centroids(dataset)
        self.labels_ = self._predict_dataset_labels(dataset)

    def fit_predict(self, dataset: list[list[float]]) -> list[int]:
        """Fit and predict the cluster label of the dataset.

        Args:
            dataset: the set of data points for clustering

        Returns:
            Cluster labels of the dataset samples.
        """
        return self.fit(dataset).labels_.copy()

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
        if self.distance_metric == "euclidean":
            distance = euclidean_distance(x, y)
            if not math.isfinite(distance):
                msg = "Distance exceeds the floating-point range"
                raise ValueError(msg)
            return distance
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
        data_points = [point.copy() for point in self.random_state.sample(dataset, self.k)]
        centroids = dict(enumerate(data_points))
        return centroids

    def _get_initial_centroids_kmeans_plus(self, dataset: list[list[float]]) -> dict[int, list[float]]:
        """Get initial centroids by k-means++ algorithm.

        Args:
            dataset: The dataset for clustering

        Returns:
            The initial centroids
        """
        centroids = {0: self.random_state.choice(dataset).copy()}
        for i in range(1, self.k):
            distances = [min(self._calc_distance(point, center) for center in centroids.values()) for point in dataset]
            centroids[i] = self._sample_next_centroid(dataset, distances).copy()
        return centroids

    def _sample_next_centroid(self, dataset: list[list[float]], distances: list[float]) -> list[float]:
        """Sample proportional to squared distance, with a fallback for fully covered data."""
        scale = max(distances)
        if scale == 0.0:
            # All locations are covered; remaining centers may coincide with existing ones.
            return self.random_state.choice(dataset)
        # A common scale cancels in probabilities and avoids squaring raw extreme distances.
        weights = [(distance / scale) ** 2 for distance in distances]
        return self.random_state.choices(dataset, weights)[0]

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
        centroids = {
            label: [self._mean(column) for column in zip(*(dataset[i] for i in members), strict=True)]
            for label, members in self.clusters_.items()
            if members
        }
        return self._fill_empty_centroids(centroids)

    @staticmethod
    def _mean(values: tuple[float, ...]) -> float:
        """Average finite coordinates with accurate summation and a scaled overflow fallback."""
        try:
            return math.fsum(values) / len(values)
        except OverflowError:
            # Normalize only on overflow; the common path preserves subnormal coordinates.
            scale = max(abs(value) for value in values)
            return (math.fsum(value / scale for value in values) / len(values)) * scale

    def _relocate_empty_clusters(self, dataset: list[list[float]]) -> None:
        """Seed empty clusters with farthest samples, leaving every donor nonempty."""
        empty = [label for label, members in self.clusters_.items() if not members]
        if not empty:
            return
        candidates = iter(
            sorted(
                (
                    (self._calc_distance(dataset[i], self.centroids_[label]), i, label)
                    for label, members in self.clusters_.items()
                    for i in members
                ),
                key=lambda candidate: candidate[0],
                reverse=True,
            )
        )
        for label in empty:
            for distance, index, donor in candidates:
                if distance == 0.0:
                    return  # Moving an already represented location cannot separate centers.
                if len(self.clusters_[donor]) > 1:
                    self.clusters_[donor].remove(index)
                    self.clusters_[label].append(index)
                    break

    def _fill_empty_centroids(self, centroids: dict[int, list[float]]) -> dict[int, list[float]]:
        """Keep valid coordinates for empty clusters remaining after relocation."""
        largest = max(centroids, key=lambda label: len(self.clusters_[label]))
        return {
            label: centroids[label] if label in centroids else centroids[largest].copy() for label in self.clusters_
        }

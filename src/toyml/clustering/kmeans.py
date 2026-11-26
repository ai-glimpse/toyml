import logging
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Literal

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

    type _Centroids = dict[int, list[float]]
    type _Clusters = dict[int, list[int]]

    k: int
    """The number of clusters, specified by user."""
    max_iter: int = 500
    """The number of iterations the algorithm will run for if it does not converge before that."""
    tol: float = 1e-5
    """Stop when every centroid coordinate moves by at most this amount."""
    centroids_init_method: Literal["random", "kmeans++"] = "random"
    """The method to initialize the centroids."""
    random_seed: int | None = None
    """Each fit uses a fresh generator. An integer seed reproduces the initialization."""
    distance_metric: Literal["euclidean"] = "euclidean"
    """The distance metric to use.(For now we only support euclidean)."""
    iter_: int = field(default=0, init=False)
    clusters_: dict[int, list[int]] = field(default_factory=dict, init=False)
    """The clusters of the dataset."""
    centroids_: dict[int, list[float]] = field(default_factory=dict, init=False)
    """The centroids of the clusters."""
    labels_: list[int] = field(default_factory=list, init=False)
    """The cluster labels of the dataset."""

    def fit(self, dataset: list[list[float]]) -> KMeans:
        """Fit the model. A failed fit preserves the previous fitted result."""
        self._validate_fit(dataset)
        initial = self._initial_centroids(dataset, random.Random(self.random_seed))
        centroids, iterations = self._lloyd(dataset, initial)

        # Even at the iteration limit, report assignments to the final centroids.
        labels = self._assign(dataset, centroids)
        clusters = self._group_by_label(labels)
        self.centroids_, self.iter_ = centroids, iterations
        self.labels_, self.clusters_ = labels, clusters
        nonempty = sum(bool(members) for members in clusters.values())
        if nonempty < self.k:
            logger.warning("Found %d non-empty clusters, fewer than requested k=%d", nonempty, self.k)
        return self

    def fit_predict(self, dataset: list[list[float]]) -> list[int]:
        """Fit the model and return a copy of its training labels."""
        return self.fit(dataset).labels_.copy()

    def predict(self, point: list[float]) -> int:
        """Choose the lowest label among centroids with the smallest computed float distance."""
        if not self.centroids_:
            msg = "The model is not fitted yet"
            raise ValueError(msg)
        self._validate_point(point, len(self.centroids_[0]))
        return self._nearest(point, self.centroids_)

    def _lloyd(self, dataset: list[list[float]], centroids: _Centroids) -> tuple[_Centroids, int]:
        """Run Lloyd iterations from the given centroids."""
        for iteration in range(1, self.max_iter + 1):
            labels = self._assign(dataset, centroids)
            clusters = self._group_by_label(labels)
            self._relocate_empty_clusters(dataset, centroids, clusters)
            next_centroids = self._centroids(dataset, clusters)
            if self._converged(centroids, next_centroids):
                return next_centroids, iteration
            centroids = next_centroids
        return centroids, self.max_iter

    def _initial_centroids(self, dataset: list[list[float]], rng: random.Random) -> _Centroids:
        if self.centroids_init_method == "random":
            return dict(enumerate(rng.sample(dataset, self.k)))

        # K-means++ samples proportional to the squared distance to the nearest centroid.
        centroids = [rng.choice(dataset)]
        while len(centroids) < self.k:
            distances = [min(self._distance(point, centroid) for centroid in centroids) for point in dataset]
            centroids.append(self._sample_next_centroid(dataset, distances, rng))
        return dict(enumerate(centroids))

    @staticmethod
    def _sample_next_centroid(dataset: list[list[float]], distances: list[float], rng: random.Random) -> list[float]:
        scale = max(distances)
        if scale == 0.0:  # Every distinct location is already covered.
            return rng.choice(dataset)
        weights = [(distance / scale) ** 2 for distance in distances]
        return rng.choices(dataset, weights)[0]

    def _assign(self, dataset: list[list[float]], centroids: _Centroids) -> list[int]:
        return [self._nearest(point, centroids) for point in dataset]

    def _nearest(self, point: list[float], centroids: _Centroids) -> int:
        return min(range(len(centroids)), key=lambda label: self._distance(point, centroids[label]))

    def _group_by_label(self, labels: list[int]) -> _Clusters:
        clusters: KMeans._Clusters = {label: [] for label in range(self.k)}
        for index, label in enumerate(labels):
            clusters[label].append(index)
        return clusters

    def _relocate_empty_clusters(self, dataset: list[list[float]], centroids: _Centroids, clusters: _Clusters) -> None:
        """Move the farthest positive-distance samples into the initially empty clusters."""
        empty = [label for label, members in clusters.items() if not members]
        if not empty:
            return
        candidates = [
            (self._distance(dataset[i], centroids[label]), i, label)
            for label, members in clusters.items()
            for i in members
        ]
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        for label, (distance, index, donor) in zip(empty, candidates, strict=False):
            if distance == 0.0:
                break
            clusters[donor].remove(index)
            clusters[label].append(index)

    def _centroids(self, dataset: list[list[float]], clusters: _Clusters) -> _Centroids:
        centroids = {
            label: [self._mean(column) for column in zip(*(dataset[i] for i in members), strict=True)]
            for label, members in clusters.items()
            if members
        }
        # Give any empty cluster a copy of the largest cluster's center.
        largest = max(clusters, key=lambda label: len(clusters[label]))
        return {
            label: centroids[label] if members else centroids[largest].copy() for label, members in clusters.items()
        }

    def _converged(self, previous: _Centroids, updated: _Centroids) -> bool:
        return all(
            abs(a - b) <= self.tol for label in previous for a, b in zip(previous[label], updated[label], strict=True)
        )

    def _distance(self, x: list[float], y: list[float]) -> float:
        match self.distance_metric:
            case "euclidean":
                distance = self._euclidean_distance(x, y)
            case _:
                msg = f"Distance metric {self.distance_metric} not supported!"
                raise ValueError(msg)

        if not math.isfinite(distance):
            msg = "Distance exceeds the floating-point range"
            raise ValueError(msg)
        return distance

    @staticmethod
    def _mean(values: tuple[float, ...]) -> float:
        return float(statistics.mean(values))

    def _validate_fit(self, dataset: list[list[float]]) -> None:
        if not dataset or not dataset[0]:
            msg = "The dataset must contain at least one sample and one feature"
            raise ValueError(msg)
        if isinstance(self.k, bool) or not isinstance(self.k, int) or not 1 <= self.k <= len(dataset):
            msg = "k must be an integer between 1 and the sample count"
            raise ValueError(msg)
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, int) or self.max_iter < 1:
            msg = "max_iter must be a positive integer"
            raise ValueError(msg)
        if not self._finite_number(self.tol) or self.tol < 0:
            msg = "tol must be finite and non-negative"
            raise ValueError(msg)
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            msg = "random_seed must be an integer or None"
            raise ValueError(msg)
        if self.centroids_init_method not in ("random", "kmeans++"):
            msg = f"Invalid centroids initialization method: {self.centroids_init_method}"
            raise ValueError(msg)
        if self.distance_metric != "euclidean":
            msg = f"Distance metric {self.distance_metric} not supported!"
            raise ValueError(msg)
        dimension = len(dataset[0])
        for point in dataset:
            self._validate_point(point, dimension)

    def _validate_point(self, point: list[float], dimension: int) -> None:
        if len(point) != dimension:
            msg = "All samples must have the same dimension"
            raise ValueError(msg)
        if not all(self._finite_number(value) for value in point):
            msg = "Features must be finite numeric values"
            raise ValueError(msg)

    @staticmethod
    def _finite_number(value: object) -> bool:
        try:
            return isinstance(value, (float, int)) and math.isfinite(value)
        except OverflowError:  # An integer too large for a float.
            return False

    @staticmethod
    def _euclidean_distance(x: list[float], y: list[float]) -> float:
        """Compute the floating-point Euclidean norm of the coordinate differences."""
        try:
            return math.hypot(*(a - b for a, b in zip(x, y, strict=True)))
        except OverflowError:
            # A difference between two valid integers can be too large to convert to float.
            return math.inf

import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class KNN:
    """K-Nearest Neighbors classification algorithm implementation.

    This class implements the K-Nearest Neighbors algorithm for classification tasks.
    Each `fit` validates the current hyperparameters against the training data.
    After changing a hyperparameter, call `fit` again before using `predict`.

    Examples:
        >>> from toyml.classification import KNN
        >>> dataset = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        >>> labels = ["A", "A", "B", "B"]
        >>> knn = KNN(k=3, std_transform=True).fit(dataset, labels)
        >>> knn.predict([2.5, 3.5])
        'A'

    References:
        1. Li Hang
        2. Tan
        3. Zhou Zhihua
        4. Murphy
        5. Harrington
    """

    k: int
    """Number of neighbors in the vote, from one to the training sample count."""
    std_transform: bool = True
    """Whether to standardize features using the training means and sample standard deviations."""
    distance_metric: Literal["euclidean"] = field(default="euclidean", kw_only=True)
    """Distance used to rank neighbors. Currently supports `"euclidean"`."""
    dataset_: list[list[float]] | None = field(default=None, init=False)
    """Training samples in the feature space used for neighbor searches."""
    labels_: list[Any] | None = field(default=None, init=False)
    """Training labels in the same order as the samples."""
    standardizer_: Standardizer | None = field(default=None, init=False)
    """Standardizer fitted to the training data when standardization is enabled."""

    def fit(self, dataset: list[list[float]], labels: list[Any]) -> KNN:
        """Store the labeled samples. A failed refit preserves the previous fitted attributes.

        Args:
            dataset: At least `k` rows of finite numeric features, all of the
                same positive width.
            labels: One hashable class label per row, in training order.

        Returns:
            The fitted instance (`self`).

        Raises:
            ValueError: If the neighbor count, distance metric, samples, or labels are
                invalid, or standardization exceeds the floating-point range.
            TypeError: If `std_transform` is not a boolean.
        """
        n_samples = len(dataset)
        self._validate_hyperparameters(n_samples)
        self._validate_labels(labels, n_samples)

        if self.std_transform:
            standardizer = Standardizer()
            # fit_transform validates the features before computing their training statistics.
            samples = standardizer.fit_transform(dataset)
        else:
            standardizer = None
            self._validate_samples(dataset)
            samples = [point.copy() for point in dataset]
        # Publish the rows, labels and scaler together so a failed refit preserves the previous model.
        self.dataset_, self.labels_, self.standardizer_ = samples, labels.copy(), standardizer
        return self

    def predict(self, x: list[float]) -> Any:  # noqa: ANN401
        """Classify a single sample by majority vote among its nearest neighbors.

        Args:
            x: Finite numeric features in training order, with the same width as a training row.

        Returns:
            The most frequent label among the nearest `k` samples. Samples whose
            distances round to the same float retain training order. Tied votes
            favor the label encountered first among the selected neighbors, from
            nearest to farthest.

        Raises:
            ValueError: If the model is unfitted or the query has invalid features
                or cannot be standardized.
        """
        if self.dataset_ is None or self.labels_ is None:
            msg = "The model is not fitted yet!"
            raise ValueError(msg)
        if self.standardizer_ is not None:
            x = self.standardizer_.transform([x])[0]
        else:
            self._validate_point(x, len(self.dataset_[0]))
        nearest = self._nearest_labels(x, self.dataset_, self.labels_)
        # Counter resolves tied counts by first occurrence in nearest-neighbor order.
        return Counter(nearest).most_common(1)[0][0]

    def _nearest_labels(self, x: list[float], samples: list[list[float]], labels: list[Any]) -> list[Any]:
        # A stable sort gives the earlier training row the last slot when distances tie at k.
        indices = sorted(range(len(samples)), key=lambda i: self._distance(x, samples[i]))
        return [labels[i] for i in indices[: self.k]]

    def _distance(self, x: list[float], y: list[float]) -> float:
        match self.distance_metric:
            case "euclidean":
                return self._euclidean_distance(x, y)

    @staticmethod
    def _euclidean_distance(x: list[float], y: list[float]) -> float:
        try:
            return math.hypot(*(a - b for a, b in zip(x, y, strict=True)))
        except OverflowError:
            # Two finite integer coordinates can have a difference larger than a float.
            return math.inf

    def _validate_hyperparameters(self, n_samples: int) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or not 1 <= self.k <= n_samples:
            msg = "k must be an integer between 1 and the training sample count"
            raise ValueError(msg)
        if not isinstance(self.std_transform, bool):
            msg = "std_transform must be a boolean"
            raise TypeError(msg)
        if self.distance_metric != "euclidean":
            msg = f"Unsupported distance metric: {self.distance_metric}"
            raise ValueError(msg)

    @staticmethod
    def _validate_labels(labels: list[Any], n_samples: int) -> None:
        if len(labels) != n_samples:
            msg = "The number of labels must match the number of samples"
            raise ValueError(msg)
        try:
            for label in labels:
                hash(label)
        except TypeError as exc:
            msg = "Each label must be hashable"
            raise ValueError(msg) from exc

    def _validate_samples(self, dataset: list[list[float]]) -> None:
        dimension = len(dataset[0])
        if dimension == 0:
            msg = "Each sample must have at least one feature"
            raise ValueError(msg)
        for point in dataset:
            self._validate_point(point, dimension)

    @staticmethod
    def _validate_point(point: list[float], dimension: int) -> None:
        if len(point) != dimension:
            msg = f"Expected {dimension} features per sample, got {len(point)}"
            raise ValueError(msg)
        try:
            finite = all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
        except OverflowError:
            finite = False
        if not finite:
            msg = "Features must be finite numeric values"
            raise ValueError(msg)


@dataclass
class Standardizer:
    """A class for standardizing numerical datasets.

    Provides methods to fit a standardization model to a dataset,
    transform datasets using the fitted model, and perform both operations
    in a single step.
    """

    _column_stats: list[tuple[float, float, float]] = field(default_factory=list, init=False)
    """Fitted (mean, scale, unit) per column. Mean and scale are expressed in that unit."""

    def fit(self, dataset: list[list[float]]) -> Standardizer:
        """Fit the standardization model to the given dataset.

        Args:
            dataset: The input dataset to fit the model to.

        Returns:
            The fitted Standardizer instance.

        Raises:
            ValueError: If features are empty, inconsistent, non-finite, or cannot be scaled.
        """
        self._validate_dataset(dataset)
        try:
            self._column_stats = [self._fit_column(column) for column in zip(*dataset, strict=True)]
        except OverflowError as exc:
            msg = "Standardization statistics exceed the floating-point range"
            raise ValueError(msg) from exc
        return self

    def transform(self, dataset: list[list[float]]) -> list[list[float]]:
        """Transform the given dataset using the fitted standardization model.

        Args:
            dataset: The input dataset to transform.

        Returns:
            New standardized rows; the input dataset is unchanged.

        Raises:
            ValueError: If unfitted or given invalid or unrepresentable feature values.
        """
        if not self._column_stats:
            msg = "The model is not fitted yet!"
            raise ValueError(msg)
        self._validate_dataset(dataset, len(self._column_stats))
        return self._transform(dataset)

    def fit_transform(self, dataset: list[list[float]]) -> list[list[float]]:
        """Fit the column statistics, then use them to standardize the training rows.

        Args:
            dataset: The input dataset to fit and transform.

        Returns:
            New standardized rows; the input dataset is unchanged.

        Raises:
            ValueError: If features are invalid or cannot be standardized.
        """
        self.fit(dataset)
        return self._transform(dataset)

    @staticmethod
    def _fit_column(column: tuple[float, ...]) -> tuple[float, float, float]:
        unit = 1.0
        std = statistics.stdev(column) if len(column) > 1 else 0.0
        if std < sys.float_info.min and any(value != column[0] for value in column):
            # Measuring in units of 2**-1022 makes even the smallest float step a normal number.
            unit = sys.float_info.min
            column = tuple(value / unit for value in column)
            std = statistics.stdev(column)
        return statistics.mean(column), std or 1.0, unit

    def _transform(self, dataset: list[list[float]]) -> list[list[float]]:
        """Apply training statistics to rows already validated by fit or transform."""
        return [
            [self._zscore(value, *stats) for value, stats in zip(row, self._column_stats, strict=True)]
            for row in dataset
        ]

    @staticmethod
    def _zscore(value: float, mean: float, scale: float, unit: float) -> float:
        if unit != 1.0:
            value /= unit
        difference = value - mean
        # Opposite extremes can overflow during centering even when their score fits.
        score = difference / scale if abs(difference) <= sys.float_info.max else value / scale - mean / scale
        if not math.isfinite(score):
            msg = "Standardized values exceed the floating-point range"
            raise ValueError(msg)
        return score

    @staticmethod
    def _validate_dataset(dataset: list[list[float]], dimension: int | None = None) -> None:
        """Validate finite, non-empty numeric rows, optionally against a fitted dimension."""
        if not dataset or not dataset[0]:
            msg = "The dataset must contain at least one sample and one feature"
            raise ValueError(msg)
        width = len(dataset[0])
        if any(len(row) != width for row in dataset):
            msg = "All rows must have the same number of columns"
            raise ValueError(msg)
        if dimension is not None and width != dimension:
            msg = "Feature dimension does not match the training data"
            raise ValueError(msg)
        try:
            finite = all(isinstance(value, (int, float)) and math.isfinite(value) for row in dataset for value in row)
        except OverflowError:
            finite = False
        if not finite:
            msg = "Features must be finite numeric values"
            raise ValueError(msg)

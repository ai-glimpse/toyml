from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def _validate_dataset(dataset: list[list[float]], dimension: int | None = None) -> int:
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
    return width


@dataclass
class KNN:
    """K-Nearest Neighbors classification algorithm implementation.

    This class implements the K-Nearest Neighbors algorithm for classification tasks.
    It supports optional standardization of the input data.

    Examples:
        >>> dataset = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        >>> labels = ["A", "A", "B", "B"]
        >>> knn = KNN(k=3, std_transform=True).fit(dataset, labels)
        >>> knn.predict([2.5, 3.5])
        'A'

    Attributes:
        k: The number of nearest neighbors to consider for classification.
        std_transform: Whether to standardize the input data (default: True).
        dataset_: The fitted dataset (standardized if std_transform is True).
        labels_: The labels corresponding to the fitted dataset.
        standardizer_: The Standardizer instance if std_transform is True.


    References:
        1. Li Hang
        2. Tan
        3. Zhou Zhihua
        4. Murphy
        5. Harrington
    """

    k: int
    std_transform: bool = True
    dataset_: list[list[float]] | None = field(default=None, init=False)
    labels_: list[Any] | None = field(default=None, init=False)
    standardizer_: Standardizer | None = field(default=None, init=False)

    def fit(self, dataset: list[list[float]], labels: list[Any]) -> KNN:
        """Fit the KNN model to the given dataset and labels.

        Args:
            dataset: The input dataset to fit the model to.
            labels: The labels corresponding to the input dataset.

        Returns:
            The fitted KNN instance.

        Raises:
            ValueError: If features, labels, or the neighbor count are invalid.
            TypeError: If std_transform is not a boolean.
        """
        _validate_dataset(dataset)
        self._validate_k(len(dataset))
        if not isinstance(self.std_transform, bool):
            msg = "std_transform must be a boolean"
            raise TypeError(msg)
        if len(labels) != len(dataset):
            msg = "The number of labels must match the number of samples"
            raise ValueError(msg)
        try:
            for label in labels:
                hash(label)
        except TypeError as exc:
            msg = "Each label must be hashable"
            raise ValueError(msg) from exc

        standardizer = None
        if self.std_transform:
            standardizer = Standardizer()
            fitted_data = standardizer.fit_transform(dataset)
        else:
            fitted_data = [row.copy() for row in dataset]
        # Commit a complete fit; validation or preprocessing failures leave the old model intact.
        self.dataset_ = fitted_data
        self.labels_ = labels.copy()
        self.standardizer_ = standardizer
        return self

    def _validate_k(self, sample_count: int) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or not 1 <= self.k <= sample_count:
            msg = "k must be an integer between 1 and the training sample count"
            raise ValueError(msg)

    def predict(self, x: list[float]) -> Any:  # noqa: ANN401
        """Predict the label of the input data.

        Args:
            x: The input data to predict.

        Returns:
            The predicted label.

        Raises:
            ValueError: If the model is unfitted or the query or neighbor count is invalid.
        """
        if self.dataset_ is None or self.labels_ is None:
            msg = "The model is not fitted yet!"
            raise ValueError(msg)

        self._validate_k(len(self.dataset_))
        _validate_dataset([x], len(self.dataset_[0]))
        if self.std_transform:
            if self.standardizer_ is None:
                msg = "Cannot find the standardization!"
                raise ValueError(msg)
            x = self.standardizer_.transform([x])[0]
        distances = [self._calculate_distance(x, point) for point in self.dataset_]
        # get k-nearest neighbors' label
        k_nearest_labels = [
            label for _, label in sorted(zip(distances, self.labels_, strict=True), key=lambda x: x[0])
        ][: self.k]
        label = Counter(k_nearest_labels).most_common(1)[0][0]
        return label

    @staticmethod
    def _calculate_distance(x: list[float], y: list[float]) -> float:
        """Compute a scaled Euclidean norm for validated finite coordinates."""
        differences = [abs(xi - yi) for xi, yi in zip(x, y, strict=True)]
        scale = max(differences, default=0.0)
        if scale == 0.0:
            return 0.0
        # Finite coordinates can have an unrepresentable difference; avoid inf / inf.
        if math.isinf(scale):
            return math.inf
        # Scaling avoids squaring huge/tiny values; fsum preserves smaller contributions.
        return scale * math.sqrt(math.fsum((difference / scale) ** 2 for difference in differences))


@dataclass
class Standardizer:
    """A class for standardizing numerical datasets.

    Provides methods to fit a standardization model to a dataset,
    transform datasets using the fitted model, and perform both operations
    in a single step.
    """

    _means: list[float] = field(default_factory=list, init=False)
    _stds: list[float] = field(default_factory=list, init=False)
    _dimension: int | None = field(default=None, init=False)

    def fit(self, dataset: list[list[float]]) -> Standardizer:
        """Fit the standardization model to the given dataset.

        Args:
            dataset: The input dataset to fit the model to.

        Returns:
            The fitted Standardizer instance.

        Raises:
            ValueError: If features are empty, inconsistent, non-finite, or cannot be scaled.
        """
        dimension = _validate_dataset(dataset)
        try:
            means = self._dataset_column_means(dataset)
            stds = self._dataset_column_stds(dataset)
        except OverflowError as exc:
            msg = "Standardization statistics exceed the floating-point range"
            raise ValueError(msg) from exc
        self._dimension = dimension
        self._means = means
        self._stds = stds
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
        if self._dimension is None:
            msg = "The model is not fitted yet!"
            raise ValueError(msg)
        return self.standardization(dataset)

    def fit_transform(self, dataset: list[list[float]]) -> list[list[float]]:
        """Fit the standardization model to the dataset and transform it in one step.

        Args:
            dataset: The input dataset to fit and transform.

        Returns:
            New standardized rows; the input dataset is unchanged.
        """
        self.fit(dataset)
        return self.transform(dataset)

    def standardization(self, dataset: list[list[float]]) -> list[list[float]]:
        """Standardize the given numerical dataset.

        The standardization is performed by subtracting the mean and dividing
        by the standard deviation for each feature.
        When the standard deviation is 0, all the values in the column are the same,
        here we set std to 1 to make every value in the column become 0 and avoid division by zero.

        Args:
            dataset: The input dataset to standardize.

        Returns:
            New standardized rows; the input dataset is unchanged.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self._dimension is None:
            msg = "The model is not fitted yet!"
            raise ValueError(msg)
        _validate_dataset(dataset, self._dimension)
        # Small nonzero standard deviations carry real variation regardless of feature units.
        scales = [1.0 if std == 0.0 else std for std in self._stds]
        result = [
            [(value - mean) / scale for value, mean, scale in zip(row, self._means, scales, strict=True)]
            for row in dataset
        ]
        if not all(math.isfinite(value) for row in result for value in row):
            msg = "Standardized values exceed the floating-point range"
            raise ValueError(msg)
        return result

    @staticmethod
    def _dataset_column_means(dataset: list[list[float]]) -> list[float]:
        """Calculate vectors mean."""
        return [statistics.mean(column) for column in zip(*dataset, strict=False)]

    @staticmethod
    def _dataset_column_stds(dataset: list[list[float]]) -> list[float]:
        """Compute sample standard deviations, using zero for a single observation."""
        return [statistics.stdev(column) if len(column) > 1 else 0.0 for column in zip(*dataset, strict=True)]


if __name__ == "__main__":
    dataset: list[list[float]] = [[0.0], [1], [2], [3]]
    labels: list[Any] = [0, 0, 1, 1]
    knn = KNN(3).fit(dataset, labels)
    print(knn.labels_)
    print(knn.predict([1.1]))

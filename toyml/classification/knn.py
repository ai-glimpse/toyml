from __future__ import annotations

import math
import statistics

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class KNN:
    """
    K-Nearest Neighbors classification algorithm implementation.

    This class implements the K-Nearest Neighbors algorithm for classification tasks.
    It supports optional standardization of the input data.

    Attributes:
        k: The number of nearest neighbors to consider for classification.
        std_transform: Whether to standardize the input data (default: True).
        dataset_: The fitted dataset (standardized if std_transform is True).
        labels_: The labels corresponding to the fitted dataset.
        standardizationer_: The Standardizationer instance if std_transform is True.

    Examples:
        >>> dataset = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        >>> labels = ['A', 'A', 'B', 'B']
        >>> knn = KNN(k=3, std_transform=True).fit(dataset, labels)
        >>> knn.predict([2.5, 3.5])
        'A'

    References:
        1. Li Hang
        2. Tan
        3. Zhou
        4. Murphy
        5. Harrington
    """

    k: int
    std_transform: bool = True
    dataset_: Optional[list[list[float]]] = None
    labels_: Optional[list[Any]] = None
    standardizationer_: Optional[Standardizationer] = None

    def fit(self, dataset: list[list[float]], labels: list[Any]) -> KNN:
        """
        Fit the KNN model to the given dataset and labels.

        Args:
            dataset: The input dataset to fit the model to.
            labels: The labels corresponding to the input dataset.

        Returns:
            The fitted KNN instance.
        """
        self.dataset_ = dataset
        self.labels_ = labels
        if self.std_transform:
            self.standardizationer_ = Standardizationer()
            self.dataset_ = self.standardizationer_.fit_transform(self.dataset_)
        return self

    def predict(self, x: list[float]) -> Any:
        """
        Predict the label of the input data.

        Args:
            x: The input data to predict.

        Returns:
            The predicted label.

        Raises:
            ValueError: If the model is not fitted yet.
        """
        if self.dataset_ is None or self.labels_ is None:
            raise ValueError("The model is not fitted yet!")

        if self.std_transform:
            if self.standardizationer_ is None:
                raise ValueError("Cannot find the standardization!")
            x = self.standardizationer_.transform([x])[0]
        distances = [self._calculate_distance(x, point) for point in self.dataset_]
        # get k-nearest neighbors' label
        k_nearest_labels = [label for _, label in sorted(zip(distances, self.labels_), key=lambda x: x[0])][:: self.k]
        label = Counter(k_nearest_labels).most_common(1)[0][0]
        return label

    @staticmethod
    def _calculate_distance(x: list[float], y: list[float]) -> float:
        """
        Calculate the Euclidean distance between two points using a numerically stable method.

        This implementation avoids overflow by using the two-pass algorithm.
        """
        assert len(x) == len(y), f"{x} and {y} have different length!"

        # First pass: find the maximum absolute difference
        max_diff = max(abs(xi - yi) for xi, yi in zip(x, y))

        if math.isclose(max_diff, 0, abs_tol=1e-9):
            return 0.0  # All elements are identical

        # Second pass: calculate the normalized sum of squares
        sum_squares = sum(((xi - yi) / max_diff) ** 2 for xi, yi in zip(x, y))

        return max_diff * math.sqrt(sum_squares)


@dataclass
class Standardizationer:
    """
    A class for standardizing numerical datasets.

    Provides methods to fit a standardization model to a dataset,
    transform datasets using the fitted model, and perform both operations
    in a single step.
    """

    _means: list[float] = field(default_factory=list)
    _stds: list[float] = field(default_factory=list)
    _dimension: Optional[int] = None

    def fit(self, dataset: list[list[float]]) -> Standardizationer:
        """
        Fit the standardization model to the given dataset.

        Args:
            dataset: The input dataset to fit the model to.

        Returns:
            The fitted Standardizationer instance.

        Raises:
            ValueError: If the dataset has inconsistent dimensions.
        """
        self._dimension = self._get_dataset_dimension(dataset)
        self._means = self._dataset_column_means(dataset)
        self._stds = self._dataset_column_stds(dataset)
        return self

    def transform(self, dataset: list[list[float]]) -> list[list[float]]:
        """
        Transform the given dataset using the fitted standardization model.

        Args:
            dataset: The input dataset to transform.

        Returns:
            The standardized dataset.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self._dimension is None:
            raise ValueError("The model is not fitted yet!")
        return self.standardization(dataset)

    def fit_transform(self, dataset: list[list[float]]) -> list[list[float]]:
        """
        Fit the standardization model to the dataset and transform it in one step.

        Args:
            dataset: The input dataset to fit and transform.

        Returns:
            The standardized dataset.
        """
        self.fit(dataset)
        return self.transform(dataset)

    def standardization(self, dataset: list[list[float]]) -> list[list[float]]:
        """
        Standardize the given numerical dataset.
        The standardization is performed by subtracting the mean and dividing by the standard deviation for each feature.
        When the standard deviation is 0, all the values in the column are the same,
        here we set std to 1 to make every value in the column become 0 and avoid division by zero.

        Args:
            dataset: The input dataset to standardize.

        Returns:
            The standardized dataset.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self._dimension is None:
            raise ValueError("The model is not fitted yet!")
        for j, column in enumerate(zip(*dataset)):
            mean, std = self._means[j], self._stds[j]
            # ref: https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/preprocessing/data.py#L70
            if math.isclose(std, 0, abs_tol=1e-9):
                std = 1
            for i, value in enumerate(column):
                dataset[i][j] = (value - mean) / std
        return dataset

    @staticmethod
    def _get_dataset_dimension(dataset: list[list[float]]) -> int:
        dimension = len(dataset[0])
        if not all([len(row) == dimension for row in dataset]):
            raise ValueError("All rows must have the same number of columns")
        return dimension

    @staticmethod
    def _dataset_column_means(dataset: list[list[float]]) -> list[float]:
        """
        Calculate vectors mean
        """
        return [statistics.mean(column) for column in zip(*dataset)]

    @staticmethod
    def _dataset_column_stds(dataset: list[list[float]]) -> list[float]:
        """
        Calculate vectors(every column) standard variance
        """
        return [statistics.stdev(column) for column in zip(*dataset)]


if __name__ == "__main__":
    dataset: list[list[float]] = [[0.0], [1], [2], [3]]
    labels: list[Any] = [0, 0, 1, 1]
    knn = KNN(3).fit(dataset, labels)
    print(knn.labels_)
    print(knn.predict([1.1]))

from __future__ import annotations

import math
import statistics

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Standardizationer:
    _means: list[float] = field(default_factory=list)
    _stds: list[float] = field(default_factory=list)
    _dimension: Optional[int] = None

    def fit(self, dataset: list[list[float]]) -> Standardizationer:
        self._dimension = self._get_dataset_dimension(dataset)
        self._means = self._dataset_column_means(dataset)
        self._stds = self._dataset_column_stds(dataset)
        return self

    def transform(self, dataset: list[list[float]]) -> list[list[float]]:
        if self._dimension is None:
            raise ValueError("The model is not fitted yet!")
        return self.standardization(dataset)

    def fit_transform(self, dataset: list[list[float]]) -> list[list[float]]:
        self.fit(dataset)
        return self.transform(dataset)

    def standardization(self, dataset: list[list[float]]) -> list[list[float]]:
        """
        The standardization of numerical dataset.
        """
        if self._dimension is None:
            raise ValueError("The model is not fitted yet!")
        for j, column in enumerate(zip(*dataset)):
            for i, value in enumerate(column):
                dataset[i][j] = (value - self._means[j]) / self._stds[j]
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


@dataclass
class KNN:
    """
    The implementation of K-Nearest Neighbors classification algorithm.

    Tip: References
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
    standarizationer_: Optional[Standardizationer] = None

    def fit(self, dataset: list[list[float]], labels: list[Any]) -> KNN:
        self.dataset_ = dataset
        self.labels_ = labels
        if self.std_transform:
            self.standarizationer_ = Standardizationer()
            self.dataset_ = self.standarizationer_.fit_transform(self.dataset_)
        return self

    def predict(self, x: list[float]) -> Any:
        if self.dataset_ is None or self.labels_ is None:
            raise ValueError("The model is not fitted yet!")

        if self.std_transform:
            if self.standarizationer_ is None:
                raise ValueError("Cannot find the standardization!")
            x = self.standarizationer_.transform([x])[0]
        distances = [self._calculate_distance(x, point) for point in self.dataset_]
        # get k-nearest neighbors' label
        k_nearest_labels = [label for _, label in sorted(zip(distances, self.labels_), key=lambda x: x[0])][:: self.k]
        label = Counter(k_nearest_labels).most_common(1)[0][0]
        return label

    @staticmethod
    def _calculate_distance(x: list[float], y: list[float]) -> float:
        """
        TODO: Implement other distance metrics
        """
        assert len(x) == len(y), f"{x} and {y} have different length!"
        return math.sqrt(sum(pow(x[i] - y[i], 2) for i in range(len(x))))


if __name__ == "__main__":
    dataset: list[list[float]] = [[0.0], [1], [2], [3]]
    labels: list[Any] = [0, 0, 1, 1]
    knn = KNN(3).fit(dataset, labels)
    print(knn.labels_)
    print(knn.predict([1.1]))

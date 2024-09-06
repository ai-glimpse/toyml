from __future__ import annotations

import math

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from toyml.utils.linear_algebra import euclidean_distance


@dataclass
class Standarizationer:
    _means: list[float] = field(default_factory=list)
    _stds: list[float] = field(default_factory=list)

    def fit(self, dataset: list[list[float]]) -> Standarizationer:
        self._means = self._vectors_mean(dataset)
        self._stds = self._vectors_std(dataset)
        return self

    def transform(self, dataset: list[list[float]]) -> list[list[float]]:
        return self.standardization(dataset, self._means, self._stds)

    def fit_transform(self, dataset: list[list[float]]) -> list[list[float]]:
        self.fit(dataset)
        return self.transform(dataset)

    def _vectors_mean(self, vectors: list[list[float]]) -> list[float]:
        """
        Calculate vectors mean

        Example1:
        >>> vectors_mean([[1.0, 2.0], [3.0, 4.0]])
        [2.0, 3.0]
        """
        d = len(vectors[0])
        n = len(vectors)
        v = [0.0] * d
        for i in range(d):
            v[i] = sum(vector[i] for vector in vectors) / n
        return v

    def _vectors_std(self, vectors: list[list[float]]) -> list[float]:
        """
        Calculate vectors(every column) standard variance
        """
        d = len(vectors[0])
        n = len(vectors)
        v = [0.0] * d
        for i in range(d):
            v[i] = math.sqrt(sum((vector[i] - self._means[i]) ** 2 for vector in vectors) / n)
        return v

    # data transformation
    def standardization(self, dataset: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
        """
        The standardization of numerical dataset.
        """
        d = len(means)
        n = len(dataset)
        for j in range(d):
            for i in range(n):
                value = dataset[i][j]
                dataset[i][j] = (value - means[j]) / stds[j]
        return dataset


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
    _dataset: Optional[list[list[float]]] = None
    _labels: Optional[list[Any]] = None
    _standarizationer: Optional[Standarizationer] = None

    def fit(self, dataset: list[list[float]], labels: list[Any]) -> KNN:
        self._dataset = dataset
        self._labels = labels
        if self.std_transform:
            self._standarizationer = Standarizationer()
            self._dataset = self._standarizationer.fit_transform(self._dataset)
        return self

    def predict(self, x: list[float]) -> Any:
        if self._dataset is None or self._labels is None:
            raise ValueError("The model is not fitted yet!")

        if self.std_transform:
            if self._standarizationer is None:
                raise ValueError("Cannot find the standarization!")
            x = self._standarizationer.transform([x])[0]
        distances = [euclidean_distance(x, point) for point in self._dataset]
        # get k-nearest neighbors' label
        k_nearest_labels = [label for _, label in sorted(zip(distances, self._labels), key=lambda x: x[0])][:: self.k]
        label = Counter(k_nearest_labels).most_common(1)[0][0]
        return label


if __name__ == "__main__":
    dataset: list[list[float]] = [[0.0], [1], [2], [3]]
    labels: list[Any] = [0, 0, 1, 1]
    knn = KNN(3).fit(dataset, labels)
    print(knn.predict([1.1]))

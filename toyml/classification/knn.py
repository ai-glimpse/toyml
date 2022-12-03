from collections import Counter

from toyml.utils.linear_algebra import (
    euclidean_distance,
    standarlization,
    vectors_mean,
    vectors_std,
)
from toyml.utils.types import DataSet, Label, Labels, Vector


class KNeighborsClassifier:
    """
    The implementation of K-Nearest Neighbors.

    Ref:
    1. Li Hang
    2. Tan
    3. Zhou
    4. Murphy
    5. Harrington
    """

    def __init__(
        self,
        dataset: DataSet,
        labels: Labels,
        k: int,
        dist=euclidean_distance,
        std=True,
    ) -> None:
        if not isinstance(dataset, list):
            raise TypeError(
                f"invalid type in {type(dataset)} for the 'dataset' argument"
            )
        self._dataset = dataset
        self._labels = labels
        self._k = k
        self._dist = dist
        # for standarlizarion
        self._is_std = std
        self._means = vectors_mean(dataset)
        self._stds = vectors_std(dataset)

    def fit(self) -> None:
        if self._is_std:
            standarlization(self._dataset, self._means, self._stds)
        else:
            pass

    def predict(self, x: Vector) -> Label:
        # x -> standarlization -> v
        d = len(x)
        v = [0.0] * d
        for i in range(d):
            v[i] = (x[i] - self._means[i]) / self._stds[i]
        dists = [self._dist(v, point) for point in self._dataset]
        # get k-nearest neighbors' label
        labels = [x for _, x in sorted(zip(dists, self._labels))][:: self._k]
        return Counter(labels).most_common(1)[0][0]


if __name__ == "__main__":
    dataset = [[0.0], [1], [2], [3]]
    labels = [0, 0, 1, 1]
    knn = KNeighborsClassifier(dataset, labels, 3)
    knn.fit()
    print(knn.predict([1.1]))

import math

from dataclasses import dataclass, field
from typing import Literal, Tuple

from toyml.utils.linear_algebra import euclidean_distance


@dataclass
class AGNES:
    """
    Agglomerative clustering algorithm.(Bottom-up Hierarchical Clustering)

    Examples:
        >>> from toyml.clustering import AGNES
        >>> dataset = [[1, 0], [1, 1], [1, 2], [10, 0], [10, 1], [10, 2]]
        >>> AGNES(k=2).fit_predict(dataset)
        [0, 0, 0, 1, 1, 1]

    Tip: References
        1. Zhou Zhihua
        2. Tan
    """

    n_cluster: int
    """The number of clusters, specified by user."""
    linkage: Literal["single", "complete", "average"] = "single"
    """The linkage method to use."""
    distance_matrix_: list[list[float]] = field(default_factory=list)
    """The distance matrix."""
    clusters_: list[list[int]] = field(default_factory=list)
    """The clusters."""
    labels_: list[int] = field(default_factory=list)
    """The labels of each sample."""

    def _get_clusters_distance(
        self,
        dataset: list[list[float]],
        cluster1: list[int],
        cluster2: list[int],
        linkage: Literal["single", "complete", "average"] = "single",
    ) -> float:
        """
        Get the distance between clusters c1 and c2 using the specified linkage method.
        """
        distances = [euclidean_distance(dataset[i], dataset[j]) for i in cluster1 for j in cluster2]

        if linkage == "single":
            return min(distances)
        elif linkage == "complete":
            return max(distances)
        elif linkage == "average":
            return sum(distances) / len(distances)
        else:
            raise ValueError("Invalid linkage method")

    def _gen_init_dist_matrix(self, dataset: list[list[float]]) -> list[list[float]]:
        """
        Gte initial distance matrix from sample points
        """
        n = len(dataset)
        distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        for i, cluster_i in enumerate(self.clusters_):
            for j, cluster_j in enumerate(self.clusters_):
                if j >= i:
                    distance_matrix[i][j] = self._get_clusters_distance(dataset, cluster_i, cluster_j)
                    distance_matrix[j][i] = distance_matrix[i][j]
        return distance_matrix

    def _get_closest_clusters(self) -> Tuple[int, int]:
        """
        Search the distance matrix to get the closest clusters.
        """
        min_dist = math.inf
        closest_clusters = (0, 0)
        for i in range(len(self.distance_matrix) - 1):
            for j in range(i + 1, len(self.distance_matrix)):
                if self.distance_matrix[i][j] < min_dist:
                    min_dist = self.distance_matrix[i][j]
                    closest_clusters = (i, j)
        return closest_clusters

    def fit(self, dataset: list[list[float]]) -> "AGNES":
        """
        Fit the model.
        """
        self.clusters_ = [[i] for i in range(len(dataset))]
        self.distance_matrix = self._gen_init_dist_matrix(dataset)
        while len(self.clusters_) > self.n_cluster:
            i, j = self._get_closest_clusters()
            # combine cluster_i and cluster_j to new cluster_i
            self.clusters_[i] += self.clusters_[j]
            # remove jth cluster
            self.clusters_.pop(j)
            # update distance matrix
            # remove jth raw in dist matrix
            self.distance_matrix.pop(j)
            # remove jth column
            for raw in self.distance_matrix:
                raw.pop(j)
            # calc new dist
            for j in range(len(self.clusters_)):
                self.distance_matrix[i][j] = self._get_clusters_distance(dataset, self.clusters_[i], self.clusters_[j])
                self.distance_matrix[j][i] = self.distance_matrix[i][j]

        # labels
        self.labels_ = [0] * len(dataset)
        for cluster_label, cluster in enumerate(self.clusters_):
            for sample_index in cluster:
                self.labels_[sample_index] = cluster_label
        return self

    def fit_predict(self, dataset: list[list[float]]) -> list[int]:
        """
        Fit the model and return the labels of each sample.
        """
        self.fit(dataset)
        return self.labels_


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [1, 5], [1, 0], [10, 2], [10, 5], [10, 0]]
    n_cluster: int = 2
    agnes = AGNES(n_cluster)
    agnes.fit(dataset)
    for i in range(n_cluster):
        print(f"Cluster[{i}]: {agnes.clusters_[i]}")

    y_pred = agnes.fit_predict(dataset)
    print("Sample labels: ", y_pred)

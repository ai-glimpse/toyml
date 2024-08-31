import math

from typing import Tuple

from toyml.utils.linear_algebra import euclidean_distance
from toyml.utils.types import Clusters, DistMat, Vector


class AGNES:
    """
    Agglomerate clustering(nesting) algorithm.(Bottom-up)

    REF:
    1. Zhou Zhihua
    2. Tan
    """

    def __init__(self, dataset: list[list[float]], k: int) -> None:
        """Initialize the Agglomerate clustering algorithm.

        Args:
            dataset: the set of data points for clustering
            k: the number of clusters, specified by user
        """
        self._dataset = dataset
        self._k = k
        self._n = len(dataset)
        self.clusters_ = [[i] for i in range(self._n)]
        self.distance_matrix = [[0.0 for _ in range(self._n)] for _ in range(self._n)]

    def _get_clusters_distance(self, c1: list[int], c2: list[int], measure="single") -> float:
        """
        Get the distance between clusters c1 and c2 using the specified linkage method.
        """
        distances = [euclidean_distance(self._dataset[i], self._dataset[j]) for i in c1 for j in c2]

        if measure == "single":
            return min(distances)
        elif measure == "complete":
            return max(distances)
        elif measure == "average":
            return sum(distances) / len(distances)
        else:
            raise ValueError("Invalid linkage method")

    def _gen_init_dist_matrix(self) -> DistMat:
        """
        Gte initial distance matrix from sample points
        """
        for i, cluster_i in enumerate(self.clusters_):
            for j, cluster_j in enumerate(self.clusters_):
                if j >= i:
                    self.distance_matrix[i][j] = self._get_clusters_distance(cluster_i, cluster_j)
                    self.distance_matrix[j][i] = self.distance_matrix[i][j]
        return self.distance_matrix

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
        # print(closest_clusters)
        return closest_clusters

    def fit(self) -> Clusters:
        """
        Get every operation togather, train our model and get k clusters
        """
        self._gen_init_dist_matrix()
        while len(self.clusters_) > self._k:
            # print(self._clusters)
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
                self.distance_matrix[i][j] = self._get_clusters_distance(self.clusters_[i], self.clusters_[j])
                self.distance_matrix[j][i] = self.distance_matrix[i][j]

        return self.clusters_

    def predict(self, point: Vector) -> int:
        """
        Predict the label of the new sample point
        """
        min_dist = math.inf
        label = -1
        for cluster_label, cluster in enumerate(self.clusters_):
            dist = min([euclidean_distance(point, self._dataset[i]) for i in cluster])
            if dist < min_dist:
                min_dist = dist
                label = cluster_label
        return label

    def print_cluster(self) -> None:
        """
        Show our k clusters.
        """
        for i in range(self._k):
            print(f"Cluster[{i}]: {self.clusters_[i]}")

    def print_labels(self) -> None:
        """
        Show our samples' labels.
        """
        y_pred = [0] * self._n
        for cluster_label, cluster in enumerate(self.clusters_):
            for sample_index in cluster:
                y_pred[sample_index] = cluster_label
        print("Sample labels: ", y_pred)


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [1, 5], [1, 0], [10, 2], [10, 5], [10, 0]]
    k: int = 2
    agnes = AGNES(dataset, k)
    agnes.fit()
    agnes.print_cluster()
    agnes.print_labels()
    print("Prediction of [0, 0]: ", agnes.predict([0.0, 0.0]))

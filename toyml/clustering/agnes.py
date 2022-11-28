import math

from typing import Tuple

from toyml.utils.linear_algebra import euclidean_distance
from toyml.utils.types import Cluster, Clusters, DataSet, DistMat, Vector

"""
TODO:
1. Plot
2. Test
"""


class Agnes:
    """
    Agglomerative clustering(nesting) algorithm.(Bottom-up)

    REF:
    1. Zhou Zhihua
    2. Tan
    """

    def __init__(self, dataset: DataSet, k: int) -> None:
        """
        dataset: the set of data points for clustering
        k: the number of clusters, specified by user
        """
        self._dataset = dataset
        self._k = k
        self._n = len(dataset)
        self._clusters = [[i] for i in range(self._n)]
        self._dist_mat = [[0.0 for i in range(self._n)] for j in range(self._n)]

    def _get_dist(self, c1: Cluster, c2: Cluster, measure=min) -> float:
        """
        Get the distance between to cluster c1 and c2 with euclidean distance.
        Using the min function as default measure(single link).
        """
        distances = []
        for i in c1:
            for j in c2:
                dist = euclidean_distance(self._dataset[i], self._dataset[j])
                distances.append(dist)
        return measure(distances)

    def _gen_init_dist_matrix(self) -> DistMat:
        """
        Gte initial distance matrix from sample points
        """
        for i, cluster_i in enumerate(self._clusters):
            for j, cluster_j in enumerate(self._clusters):
                if j >= i:
                    self._dist_mat[i][j] = self._get_dist(cluster_i, cluster_j)
                    self._dist_mat[j][i] = self._dist_mat[i][j]
        return self._dist_mat

    def _get_closest_clusters(self) -> Tuple[int, int]:
        """
        Search the distance matrix to get the closest clusters.
        """
        min_dist = math.inf
        closest_clusters = (0, 0)
        for i in range(len(self._dist_mat) - 1):
            for j in range(i + 1, len(self._dist_mat)):
                if self._dist_mat[i][j] < min_dist:
                    min_dist = self._dist_mat[i][j]
                    closest_clusters = (i, j)
        # print(closest_clusters)
        return closest_clusters

    def fit(self) -> Clusters:
        """
        Get every operation togather, train our model and get k clusters
        """
        self._gen_init_dist_matrix()
        while len(self._clusters) > self._k:
            # print(self._clusters)
            i, j = self._get_closest_clusters()
            # combine cluster_i and cluster_j to new cluster_i
            self._clusters[i] += self._clusters[j]
            # remove jth cluster
            self._clusters.pop(j)
            # update distance matrix
            # remove jth raw in dist matrix
            self._dist_mat.pop(j)
            # remove jth column
            for raw in self._dist_mat:
                raw.pop(j)
            # calc new dist
            for j in range(len(self._clusters)):
                self._dist_mat[i][j] = self._get_dist(
                    self._clusters[i], self._clusters[j]
                )
                self._dist_mat[j][i] = self._dist_mat[i][j]

        return self._clusters

    def predict(self, point: Vector) -> int:
        """
        Predict the label of the new sample point
        """
        min_dist = math.inf
        label = -1
        for cluster_label, cluster in enumerate(self._clusters):
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
            print(f"Cluster[{i}]: {self._clusters[i]}")

    def print_labels(self) -> None:
        """
        Show our samples' labels.
        """
        y_pred = [0] * self._n
        for cluster_label, cluster in enumerate(self._clusters):
            for sample_index in cluster:
                y_pred[sample_index] = cluster_label
        print("Sample labels: ", y_pred)


if __name__ == "__main__":
    dataset = [[1.0, 2], [1, 5], [1, 0], [10, 2], [10, 5], [10, 0]]
    k = 2
    agnes = Agnes(dataset, k)
    agnes.fit()
    agnes.print_cluster()
    agnes.print_labels()
    print("Prediction of [0, 0]: ", agnes.predict([0.0, 0.0]))

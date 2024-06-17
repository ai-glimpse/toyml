import random

from typing import Any, Tuple

from toyml.utils.linear_algebra import euclidean_distance
from toyml.utils.types import Clusters, DataSet, Vector


class Kmeans:
    """
    Naive K-means algorithm.

    References:
    1. Zhou Zhihua
    2. Murphy

    Note:
        Here we just code the naive K-means.

    See Also:
      - K-means++ algorithm: [toyml.clustering.kmeans.plus.KmeansPlus][]
      - Bisecting K-means algorithm: [`toyml.clustering.kmeans.bisect`][kmeans-bisect]
    """

    def __init__(self, dataset: DataSet, k: int, max_iter: int = 500) -> None:
        """Initialize the K-means algorithm

        Args:
            dataset: the set of data points for clustering
            k: the number of clusters, specified by user
            max_iter: The number of iterations the algorithm will run for if it does not converge before that.

        """
        self._dataset: DataSet = dataset
        self._k: int = k
        self._max_iter: int = max_iter
        # results
        self._centroids: Any = [[] for _ in range(self._k)]
        self._clusters: Clusters = [[]]

    def _get_initial_centroids(self) -> DataSet:
        """
        get initial centroids by a simple random selection
        """
        return random.sample(self._dataset, self._k)

    @staticmethod
    def _get_centroid_label(point: Vector, centroids: DataSet) -> int:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        return distances.index(min(distances))

    def _get_clusters(self, centroids: DataSet) -> Clusters:
        clusters: Clusters = [[] for _ in range(self._k)]
        for i, point in enumerate(self._dataset):
            centroid_label = self._get_centroid_label(point, centroids)
            clusters[centroid_label].append(i)
        return clusters

    def _get_centroids(self, clusters: Clusters) -> DataSet:
        # clusters: indexes -> data points
        points_clusters = [[self._dataset[i] for i in cluster] for cluster in clusters]
        centroids: DataSet = [[] for _ in range(self._k)]  # TODO: DataSet type or newer type name with save type comb
        for i, cluster in enumerate(points_clusters):
            centroid = [sum(t) / self._k for t in zip(*cluster)]
            centroids[i] = centroid
        return centroids

    def fit(self) -> Tuple[DataSet, Clusters]:
        centroids = self._get_initial_centroids()
        clusters = self._get_clusters(centroids)  # just for unbound error
        for _ in range(self._max_iter):
            clusters = self._get_clusters(centroids)
            prev_centroids = centroids
            centroids = self._get_centroids(clusters)
            # If no centroids change, the algorithm is convergent
            if prev_centroids == centroids:
                print("Training Converged")
                break
        self._centroids = centroids
        self._clusters = clusters
        return centroids, clusters

    def predict(self, point: Vector) -> int:
        """
        Predict the label of the point
        Args:
            point: the data point to predict

        Returns: the label of the point

        """
        return self._get_centroid_label(point, self._centroids)

    def print_cluster(self) -> None:
        for i in range(self._k):
            print(f"label({i}) -> {self._centroids[i]}: {self._clusters[i]}")

    def print_labels(self) -> None:
        y_pred = [0] * len(self._dataset)
        for cluster_index in range(self._k):
            for sample_index in self._clusters[cluster_index]:
                y_pred[sample_index] = cluster_index
        print("Sample labels: ", y_pred)


if __name__ == "__main__":
    dataset: DataSet = [[1.0, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]]
    k: int = 2
    # kmeans
    print("Test K-means...")
    kmeans = Kmeans(dataset, k)
    kmeans.fit()
    kmeans.print_cluster()
    kmeans.predict([0.0, 0.0])

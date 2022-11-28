from toyml.clustering.kmeans_clustering import Kmeans
from toyml.utils.linear_algebra import sse
from toyml.utils.types import Cluster, Clusters, DataSet


class BisectingKmeans:
    """
    Bisecting K-means algorithm.
    Belong to Divisive hierarchical clustering (DIANA) algorithm.(top-down)

    REF:
    1. Harrington
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
        # top level: only one cluster
        # remember that our cluster only contains the sample indexes
        self._clusters = [list(range(self._n))]

    def _get_sse_error_from_cluster(self, cluster: Cluster) -> float:
        cluster_data = [self._dataset[i] for i in cluster]
        return sse(cluster_data)

    def fit(self) -> Clusters:
        while len(self._clusters) < self._k:
            total_error = sum(
                self._get_sse_error_from_cluster(cluster) for cluster in self._clusters
            )
            min_error = total_error
            split_cluster_index = -1
            split_cluster_into = [[] for i in range(2)]
            for cluster_index, cluster in enumerate(self._clusters):
                # perform K-means with k=2
                cluster_data = [self._dataset[i] for i in cluster]
                kmeans = Kmeans(cluster_data, 2)
                cluster1, cluster2 = kmeans.fit()[1]
                # error calc
                cluster_unsplit_error = self._get_sse_error_from_cluster(cluster)
                cluster_split_error = self._get_sse_error_from_cluster(
                    cluster1
                ) + self._get_sse_error_from_cluster(cluster2)
                new_total_error = (
                    total_error - cluster_unsplit_error + cluster_split_error
                )
                if new_total_error < min_error:
                    min_error = new_total_error
                    split_cluster_index = cluster_index
                    split_cluster_into = [cluster1, cluster2]
            # commit this split
            self._clusters.pop(split_cluster_index)
            self._clusters.insert(split_cluster_index, split_cluster_into[0])
            self._clusters.insert(split_cluster_index, split_cluster_into[1])
        return self._clusters

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
    # Bisecting K-means testing
    diana = BisectingKmeans(dataset, k)
    diana.fit()
    diana.print_cluster()
    diana.print_labels()

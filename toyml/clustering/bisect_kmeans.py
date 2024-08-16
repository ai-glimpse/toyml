from __future__ import annotations

from dataclasses import dataclass, field

from toyml.clustering.kmeans import Kmeans
from toyml.utils.linear_algebra import sum_square_error


@dataclass
class BisectingKmeans:
    """
    Bisecting K-means algorithm.
    Belong to Divisive hierarchical clustering (DIANA) algorithm.(top-down)

    Tip: References
        1. Harrington
        2. Tan

    See Also:
      - K-means algorithm: [toyml.clustering.kmeans][]
    """

    k: int
    """The number of clusters, specified by user."""
    clusters: list[list[int]] = field(default_factory=list)
    """The clusters of the dataset."""
    labels: list[int] = field(default_factory=list)
    """The cluster labels of the dataset."""

    def fit(self, dataset: list[list[float]]) -> "BisectingKmeans":
        n = len(dataset)
        # check dataset
        if self.k > n:
            raise ValueError(
                f"Number of clusters(k) cannot be greater than the number of samples(n), not get {self.k=} > {n=}"
            )
        # start with only one cluster which contains all the data points in dataset
        self.clusters = [list(range(n))]
        self.labels = self._get_dataset_labels(dataset)
        total_error = sum(sum_square_error([dataset[i] for i in cluster]) for cluster in self.clusters)
        # iterate until got k clusters
        while len(self.clusters) < self.k:
            # init values for later iteration
            split_cluster_index = -1
            split_cluster_into: tuple[list[int], list[int]] = ([], [])
            for cluster_index, cluster in enumerate(self.clusters):
                # perform K-means with k=2
                cluster_data = [dataset[i] for i in cluster]
                # If the cluster cannot be split further, skip it
                if len(cluster_data) < 2:
                    continue
                kmeans = Kmeans(k=2).fit(cluster_data)
                assert kmeans.clusters is not None
                cluster1, cluster2 = kmeans.clusters[0], kmeans.clusters[1]
                # Note: map the cluster's inner index to the truth index in dataset
                cluster1 = [cluster[i] for i in cluster1]
                cluster2 = [cluster[i] for i in cluster2]
                # split error calculation
                cluster_unsplit_error = sum_square_error([dataset[i] for i in cluster])
                cluster_split_error = sum_square_error([dataset[i] for i in cluster1]) + sum_square_error(
                    [dataset[i] for i in cluster2]
                )
                new_total_error = total_error - cluster_unsplit_error + cluster_split_error
                if new_total_error < total_error:
                    total_error = new_total_error
                    split_cluster_index = cluster_index
                    split_cluster_into = (cluster1, cluster2)

            if split_cluster_index == -1:  # won't happen normally
                raise ValueError("Can not split the cluster further")
            else:
                self._commit_split(split_cluster_index, split_cluster_into)
                self.labels = self._get_dataset_labels(dataset)
        return self

    def fit_predict(self, dataset: list[list[float]]) -> list[int]:
        """
        Fit and predict the cluster label of the dataset.

        Args:
            dataset: the set of data points for clustering

        Returns:
            Cluster labels of the dataset samples.
        """
        return self.fit(dataset).labels

    def _commit_split(
        self,
        split_cluster_index: int,
        split_cluster_into: tuple[list[int], list[int]],
    ):
        # print(self.clusters[split_cluster_index], '-->', split_cluster_into)
        self.clusters.pop(split_cluster_index)
        self.clusters.insert(split_cluster_index, split_cluster_into[0])
        self.clusters.insert(split_cluster_index, split_cluster_into[1])

    def _get_dataset_labels(self, dataset: list[list[float]]) -> list[int]:
        labels = [-1] * len(dataset)
        for cluster_label, cluster in enumerate(self.clusters):  # type: ignore
            for data_point_index in cluster:
                labels[data_point_index] = cluster_label
        return labels


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [1, 5], [1, 0], [10, 2], [10, 5], [10, 0]]
    k = 6
    # Bisecting K-means testing
    diana = BisectingKmeans(k).fit(dataset)
    print(diana.clusters)
    print(diana.labels)

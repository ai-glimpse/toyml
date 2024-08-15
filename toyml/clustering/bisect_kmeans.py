from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
    clusters: Optional[list[list[int]]] = None
    """The clusters of the dataset."""

    def fit(self, dataset: list[list[float]]) -> "BisectingKmeans":
        n = len(dataset)
        # check dataset
        if self.k > n:
            raise ValueError(
                f"Number of clusters(k) cannot be greater than the number of samples(n)," f"not get {k=} > {n=}"
            )
        # start with only one cluster which contains all the data points in dataset
        self.clusters = [list(range(n))]
        total_error = sum(sum_square_error([dataset[i] for i in cluster]) for cluster in self.clusters)
        # iterate until got k clusters
        while len(self.clusters) < self.k:
            # init values for later iteration
            split_cluster_index = -1
            split_cluster_into: list[list[int]] = [[] for _ in range(2)]
            for cluster_index, cluster in enumerate(self.clusters):
                # perform K-means with k=2
                cluster_data = [dataset[i] for i in cluster]
                if len(cluster_data) < 2:
                    break
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
                    split_cluster_into = [cluster1, cluster2]
            # TODO: better condition/logic here
            if split_cluster_index == -1:
                raise ValueError("Can not split the cluster further")
            else:
                # commit this split
                # print(self.clusters[split_cluster_index], '-->', split_cluster_into)
                self.clusters.pop(split_cluster_index)
                self.clusters.insert(split_cluster_index, split_cluster_into[0])
                self.clusters.insert(split_cluster_index, split_cluster_into[1])
        return self

    def print_cluster(self) -> None:
        """
        Show our k clusters.
        """
        assert self.clusters is not None
        for i in range(self.k):
            print(f"Cluster[{i}]: {self.clusters[i]}")


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [1, 5], [1, 0], [10, 2], [10, 5], [10, 0]]
    k = 4
    # Bisecting K-means testing
    diana = BisectingKmeans(k).fit(dataset)
    diana.print_cluster()

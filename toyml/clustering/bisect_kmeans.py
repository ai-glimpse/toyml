from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from toyml.clustering.kmeans import Kmeans
from toyml.utils.linear_algebra import sse


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

    @staticmethod
    def _get_cluster_sse(cluster: list[int], dataset: list[list[float]]) -> float:
        cluster_data = [dataset[i] for i in cluster]
        return sse(cluster_data)

    def fit(self, dataset: list[list[float]]) -> "BisectingKmeans":
        n = len(dataset)
        self.clusters = [list(range(n))]
        while len(self.clusters) < self.k:
            total_error = sum(self._get_cluster_sse(cluster, dataset) for cluster in self.clusters)
            min_error = total_error
            split_cluster_index = -1
            split_cluster_into: list[list[int]] = [[] for _ in range(2)]
            for cluster_index, cluster in enumerate(self.clusters):
                # perform K-means with k=2
                cluster_data = [dataset[i] for i in cluster]
                kmeans = Kmeans(k=2).fit(cluster_data)
                assert kmeans.clusters is not None
                cluster1, cluster2 = kmeans.clusters[0], kmeans.clusters[1]
                # error calc
                cluster_unsplit_error = self._get_cluster_sse(cluster, dataset)
                cluster_split_error = self._get_cluster_sse(cluster1, dataset) + self._get_cluster_sse(
                    cluster2, dataset
                )
                new_total_error = total_error - cluster_unsplit_error + cluster_split_error
                if new_total_error < min_error:
                    min_error = new_total_error
                    split_cluster_index = cluster_index
                    split_cluster_into = [cluster1, cluster2]
            # commit this split
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
    k = 2
    # Bisecting K-means testing
    diana = BisectingKmeans(k).fit(dataset)
    diana.print_cluster()

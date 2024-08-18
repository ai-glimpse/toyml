from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from toyml.clustering.kmeans import Kmeans
from toyml.utils.linear_algebra import sum_square_error


@dataclass
class ClusterTree:
    parent: Optional[ClusterTree] = None
    """Parent node."""
    left: Optional[ClusterTree] = None
    """Left child node."""
    right: Optional[ClusterTree] = None
    """Right child node."""
    cluster: list[int] = field(default_factory=list)
    """The cluster: dataset sample indices."""

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def leaf_cluster_nodes(self) -> list[ClusterTree]:
        """Get all the leaves in the cluster tree, which are the clusters of dataset"""
        clusters = []

        def dfs(node: ClusterTree):
            # only collect the leaf nodes
            if node.is_leaf():
                clusters.append(node)
            if node.left:
                dfs(node.left)
            if node.right:
                dfs(node.right)

        dfs(self)
        return clusters

    def get_clusters(self) -> list[list[int]]:
        """
        Get all clusters(cluster in leaf nodes).
        """
        clusters = [cluster_node.cluster for cluster_node in self.leaf_cluster_nodes()]
        return clusters

    def add_left_child(self, node: ClusterTree) -> ClusterTree:
        self.left = node
        node.parent = self
        return self

    def add_right_child(self, node: ClusterTree) -> ClusterTree:
        self.right = node
        node.parent = self
        return self


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
    cluster_tree: ClusterTree = field(default_factory=ClusterTree)
    """The cluster tree"""
    labels: list[int] = field(default_factory=list)
    """The cluster labels of the dataset."""

    def fit(self, dataset: list[list[float]]) -> "BisectingKmeans":
        """
        Fit the dataset with Bisecting K-means algorithm.

        Args:
            dataset: The set of data points for clustering.

        Returns:
            self.

        """
        n = len(dataset)
        # check dataset
        if self.k > n:
            raise ValueError(
                f"Number of clusters(k) cannot be greater than the number of samples(n), not get {self.k=} > {n=}"
            )
        # start with only one cluster which contains all the data points in dataset
        cluster = list(range(n))
        self.cluster_tree.cluster = cluster
        self.labels = self._get_dataset_labels(dataset)
        total_error = sum_square_error([dataset[i] for i in cluster])
        # iterate until got k clusters
        while len(self.cluster_tree.get_clusters()) < self.k:
            # init values for later iteration
            to_splot_cluster_node = None
            split_cluster_into: tuple[list[int], list[int]] = ()  # type: ignore
            for cluster_index, cluster_node in enumerate(self.cluster_tree.leaf_cluster_nodes()):
                # perform K-means with k=2
                cluster_data = [dataset[i] for i in cluster_node.cluster]
                # If the cluster cannot be split further, skip it
                if len(cluster_data) < 2:
                    continue
                # Bisect by kmeans with k=2
                cluster_unsplit_error, cluster_split_error, (cluster1, cluster2) = self._bisect_by_kmeans(
                    cluster_data, cluster_node
                )
                new_total_error = total_error - cluster_unsplit_error + cluster_split_error
                if new_total_error < total_error:
                    total_error = new_total_error
                    split_cluster_into = (cluster1, cluster2)
                    to_splot_cluster_node = cluster_node

            if to_splot_cluster_node is not None:
                self._commit_split(to_splot_cluster_node, split_cluster_into)
                self.labels = self._get_dataset_labels(dataset)

        return self

    def fit_predict(self, dataset: list[list[float]]) -> list[int]:
        """
        Fit and predict the cluster label of the dataset.

        Args:
            dataset: The set of data points for clustering.

        Returns:
            Cluster labels of the dataset samples.
        """
        return self.fit(dataset).labels

    @staticmethod
    def _bisect_by_kmeans(
        cluster_data: list[list[float]],
        cluster_node: ClusterTree,
    ) -> tuple[float, float, tuple[list[int], list[int]]]:
        kmeans = Kmeans(k=2).fit(cluster_data)
        assert kmeans.clusters is not None
        cluster1, cluster2 = kmeans.clusters[0], kmeans.clusters[1]
        # Note: map the cluster's inner index to the truth index in dataset
        cluster1 = [cluster_node.cluster[i] for i in cluster1]
        cluster2 = [cluster_node.cluster[i] for i in cluster2]
        # split error calculation
        cluster_unsplit_error = sum_square_error([dataset[i] for i in cluster_node.cluster])
        cluster_split_error = sum_square_error([dataset[i] for i in cluster1]) + sum_square_error(
            [dataset[i] for i in cluster2]
        )
        return cluster_unsplit_error, cluster_split_error, (cluster1, cluster2)

    @staticmethod
    def _commit_split(
        cluster_node: ClusterTree,
        split_cluster_into: tuple[list[int], list[int]],
    ):
        # cluster tree
        cluster_node.add_left_child(ClusterTree(cluster=split_cluster_into[0]))
        cluster_node.add_right_child(ClusterTree(cluster=split_cluster_into[1]))

    def _get_dataset_labels(self, dataset: list[list[float]]) -> list[int]:
        labels = [-1] * len(dataset)
        for cluster_label, cluster in enumerate(self.cluster_tree.get_clusters()):
            for data_point_index in cluster:
                labels[data_point_index] = cluster_label
        return labels


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [10.0, 1.0], [10.0, 2.0], [11.0, 1.0]]
    k = 2
    # Bisecting K-means testing
    diana = BisectingKmeans(k).fit(dataset)
    print(diana.cluster_tree.get_clusters())
    print(diana.labels)

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

from toyml.utils.linear_algebra import euclidean_distance


@dataclass
class ClusterTreeNode:
    """
    Cluster tree node.
    The cluster tree is a binary tree, each node is a cluster.
    The root node is the whole dataset, and each node is split into two clusters until the number of clusters is equal to the specified number of clusters.
    The cluster is represented by the indices of the dataset.
    The cluster tree is used to store the cluster information and the relationship between clusters.
    """

    parent: Optional[ClusterTreeNode] = None
    """Parent node."""
    children: list[ClusterTreeNode] = field(default_factory=list)
    """Children nodes."""
    sample_indices: list[int] = field(default_factory=list)
    """The cluster: dataset sample indices."""

    def add_child(self, child: ClusterTreeNode):
        child.parent = self
        self.children.append(child)


@dataclass
class AGNES:
    """
    Agglomerative clustering algorithm.(Bottom-up Hierarchical Clustering)

    Examples:
        >>> from toyml.clustering import AGNES
        >>> dataset = [[1, 0], [1, 1], [1, 2], [10, 0], [10, 1], [10, 2]]
        >>> AGNES(n_cluster=2).fit_predict(dataset)
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
    clusters_: list[ClusterTreeNode] = field(default_factory=list)
    """The clusters."""
    labels_: list[int] = field(default_factory=list)
    """The labels of each sample."""
    cluster_tree_: ClusterTreeNode = field(default_factory=ClusterTreeNode)

    def _get_clusters_distance(
        self,
        dataset: list[list[float]],
        cluster1: ClusterTreeNode,
        cluster2: ClusterTreeNode,
    ) -> float:
        """
        Get the distance between clusters c1 and c2 using the specified linkage method.
        """
        distances = [
            euclidean_distance(dataset[i], dataset[j]) for i in cluster1.sample_indices for j in cluster2.sample_indices
        ]

        if self.linkage == "single":
            return min(distances)
        elif self.linkage == "complete":
            return max(distances)
        elif self.linkage == "average":
            return sum(distances) / len(distances)
        else:
            raise ValueError("Invalid linkage method")

    def _get_init_distance_matrix(self, dataset: list[list[float]]) -> list[list[float]]:
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
        for i in range(len(self.distance_matrix_) - 1):
            for j in range(i + 1, len(self.distance_matrix_)):
                if self.distance_matrix_[i][j] < min_dist:
                    min_dist = self.distance_matrix_[i][j]
                    closest_clusters = (i, j)
        return closest_clusters

    def fit(self, dataset: list[list[float]]) -> "AGNES":
        """
        Fit the model.
        """
        self._init_clusters(len(dataset))
        self.distance_matrix_ = self._get_init_distance_matrix(dataset)
        while len(self.clusters_) > self.n_cluster:
            i, j = self._get_closest_clusters()
            # merge cluster_i and cluster_j
            self._merge_clusters(i, j)
            # update distance matrix
            self._update_distance_matrix(dataset, i, j)
        # assign dataset labels
        self._get_labels(len(dataset))
        return self

    def fit_predict(self, dataset: list[list[float]]) -> list[int]:
        """
        Fit the model and return the labels of each sample.
        """
        self.fit(dataset)
        return self.labels_

    def _init_clusters(self, n: int):
        """
        Initialize clusters.

        Args:
            n: the number of samples
        """
        self.clusters_ = [ClusterTreeNode(sample_indices=[i]) for i in range(n)]
        self.cluster_tree_ = ClusterTreeNode(
            # Note: here we set cluster tree's children to self.clusters_
            # So we don't need to update the cluster tree's children when self.clusters_ update
            children=self.clusters_,
            sample_indices=list(range(n)),
        )
        # all leaf tree nodes' parent is the cluster tree root node
        for cluster in self.clusters_:
            cluster.parent = self.cluster_tree_

    def _merge_clusters(self, i: int, j: int):
        """
        Merge two clusters to a new cluster.

        Args:
            i: the first indices of the clusters to merge
            j: the second indices of the clusters to merge
        """
        cluster_i, cluster_j = self.clusters_[i], self.clusters_[j]
        # build new parent cluster
        parent_cluster = ClusterTreeNode()
        # sort the sample indices for convenience
        parent_cluster.sample_indices = sorted(cluster_i.sample_indices + cluster_j.sample_indices)
        parent_cluster.add_child(cluster_i)
        parent_cluster.add_child(cluster_j)
        parent_cluster.parent = self.cluster_tree_
        # replace cluster1 with parent_cluster in clusters_
        self.clusters_[i] = parent_cluster
        self.clusters_.pop(j)

    def _update_distance_matrix(self, dataset: list[list[float]], i: int, j: int):
        """
        Update the distance matrix after merging two clusters.
        """
        self.distance_matrix_.pop(j)
        # remove jth column
        for raw in self.distance_matrix_:
            raw.pop(j)
        # calc new dist
        for j in range(len(self.clusters_)):
            self.distance_matrix_[i][j] = self._get_clusters_distance(dataset, self.clusters_[i], self.clusters_[j])
            self.distance_matrix_[j][i] = self.distance_matrix_[i][j]

    def _get_labels(self, n: int):
        self.labels_ = [-1] * n
        for cluster_label, cluster in enumerate(self.clusters_):
            for sample_index in cluster.sample_indices:
                self.labels_[sample_index] = cluster_label


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [1, 5], [1, 0], [10, 2], [10, 5], [10, 0]]
    n_cluster: int = 2
    agnes = AGNES(n_cluster).fit(dataset)
    for i in range(n_cluster):
        print(f"Cluster[{i}]: {agnes.clusters_[i].sample_indices}")
    y_pred = agnes.fit_predict(dataset)
    print("Sample labels: ", y_pred)

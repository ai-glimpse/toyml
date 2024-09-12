from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


@dataclass
class AGNES:
    """
    Agglomerative clustering algorithm (Bottom-up Hierarchical Clustering)

    Examples:
        >>> from toyml.clustering import AGNES
        >>> dataset = [[1, 0], [1, 1], [1, 2], [10, 0], [10, 1], [10, 2]]
        >>> agnes = AGNES(n_cluster=2).fit(dataset)
        >>> print(agnes.labels_)
        [0, 0, 0, 1, 1, 1]

        >>> # Using fit_predict method
        >>> labels = agnes.fit_predict(dataset)
        >>> print(labels)
        [0, 0, 0, 1, 1, 1]

        >>> # Using different linkage methods
        >>> agnes = AGNES(n_cluster=2, linkage="complete").fit(dataset)
        >>> print(agnes.labels_)
        [0, 0, 0, 1, 1, 1]

        >>> # Plotting dendrogram
        >>> agnes = AGNES(n_cluster=1).fit(dataset)  # doctest: +SKIP
        >>> agnes.plot_dendrogram(show=True)  # doctest: +SKIP

    Tip: The AGNES Dendrogram Plot
        ![AGNES Dendrogram](../../images/agnes_dendrogram.png)

    Tip: References
        1. Zhou Zhihua
        2. Tan
    """

    n_cluster: int
    """The number of clusters, specified by user."""
    linkage: Literal["single", "complete", "average"] = "single"
    """The linkage method to use."""
    distance_metric: Literal["euclidean"] = "euclidean"
    """The distance metric to use.(For now we only support euclidean)."""
    distance_matrix_: list[list[float]] = field(default_factory=list)
    """The distance matrix."""
    clusters_: list[ClusterTree] = field(default_factory=list)
    """The clusters."""
    labels_: list[int] = field(default_factory=list)
    """The labels of each sample."""
    cluster_tree_: Optional[ClusterTree] = None
    linkage_matrix: list[list[float]] = field(default_factory=list)
    _cluster_index: int = 0

    def __post_init__(self) -> None:
        # check the linkage method
        if self.linkage not in ["single", "complete", "average"]:
            raise ValueError(
                f"Invalid linkage method: {self.linkage}, should be one of ['single', 'complete', 'average']"
            )
        # check the number of clusters
        if self.n_cluster < 1:
            raise ValueError("The number of clusters should be at least 1")

    def fit(self, dataset: list[list[float]]) -> AGNES:
        """
        Fit the model.
        """
        self._validate(dataset)
        n = len(dataset)
        self.clusters_ = [ClusterTree(cluster_index=i, sample_indices=[i]) for i in range(n)]
        self._cluster_index = n
        self.distance_matrix_ = self._get_init_distance_matrix(dataset)
        while len(self.clusters_) > self.n_cluster:
            (i, j), cluster_ij_distance = self._get_closest_clusters()
            # merge cluster_i and cluster_j
            self._merge_clusters(i, j, cluster_ij_distance)
            # update distance matrix
            self._update_distance_matrix(dataset, i, j)
        # build cluster_tree_
        self.cluster_tree_ = self._build_cluster_tree(n)
        # assign dataset labels
        self._get_labels(len(dataset))
        return self

    def fit_predict(self, dataset: list[list[float]]) -> list[int]:
        """
        Fit the model and return the labels of each sample.
        """
        self.fit(dataset)
        return self.labels_

    def _validate(self, dataset: list[list[float]]) -> None:
        """
        Validate the dataset.
        """
        n = len(dataset)
        # check the number of clusters
        if self.n_cluster > n:
            raise ValueError(
                "The number of clusters should be less than the number of data points,"
                "but got n_cluster={self.n_cluster} and n={n}"
            )
        # check the dataset rows
        if any(len(row) != len(dataset[0]) for row in dataset):
            raise ValueError("All samples should have the same dimension")

    def _get_clusters_distance(
        self,
        dataset: list[list[float]],
        cluster1: ClusterTree,
        cluster2: ClusterTree,
    ) -> float:
        """
        Get the distance between clusters c1 and c2 using the specified linkage method.
        """
        distances = [
            self._get_distance(dataset[i], dataset[j]) for i in cluster1.sample_indices for j in cluster2.sample_indices
        ]

        if self.linkage == "single":
            return min(distances)
        elif self.linkage == "complete":
            return max(distances)
        elif self.linkage == "average":
            return sum(distances) / len(distances)

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

    def _get_closest_clusters(self) -> Tuple[Tuple[int, int], float]:
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
        return closest_clusters, min_dist

    def _build_cluster_tree(self, n: int) -> ClusterTree:
        if len(self.clusters_) != 1:
            cluster_tree = ClusterTree(cluster_index=-1, sample_indices=list(range(n)))
            for cluster in self.clusters_:
                cluster_tree.add_child(cluster)
        else:
            cluster_tree = self.clusters_[0]
        return cluster_tree

    def _merge_clusters(self, i: int, j: int, cluster_ij_distance: float) -> None:
        """
        Merge two clusters to a new cluster.

        Args:
            i: the first indices of the clusters to merge
            j: the second indices of the clusters to merge
            cluster_ij_distance: the distance between the two clusters
        """
        cluster_i, cluster_j = self.clusters_[i], self.clusters_[j]
        # build new parent cluster
        parent_cluster = ClusterTree(cluster_index=self._cluster_index)
        self._cluster_index += 1
        # sort the sample indices for convenience
        parent_cluster.sample_indices = sorted(cluster_i.sample_indices + cluster_j.sample_indices)
        parent_cluster.add_child(cluster_i)
        parent_cluster.add_child(cluster_j)
        # parent_cluster.parent = self.cluster_tree_
        parent_cluster.children_cluster_distance = cluster_ij_distance
        # replace cluster1 with parent_cluster in clusters_
        self.clusters_[i] = parent_cluster
        self.clusters_.pop(j)

        # linkage matrix
        self.linkage_matrix.append(
            [cluster_i.cluster_index, cluster_j.cluster_index, cluster_ij_distance, len(parent_cluster.sample_indices)]
        )

    def _update_distance_matrix(self, dataset: list[list[float]], i: int, j: int) -> None:
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

    def _get_labels(self, n: int) -> None:
        self.labels_ = [-1] * n
        for cluster_label, cluster in enumerate(self.clusters_):
            for sample_index in cluster.sample_indices:
                self.labels_[sample_index] = cluster_label

    def _get_distance(self, x: list[float], y: list[float]) -> float:
        assert len(x) == len(y), f"{x} and {y} have different length!"
        if self.distance_metric == "euclidean":
            return math.sqrt(sum(pow(x[i] - y[i], 2) for i in range(len(x))))
        else:
            raise ValueError(f"Distance metric {self.distance_metric} not supported!")

    def plot_dendrogram(
        self,
        figure_name: str = "agnes_dendrogram.png",
        show: bool = False,
    ) -> None:
        """
        Plot the dendrogram of the clustering result.

        This method visualizes the hierarchical structure of the clustering
        using a dendrogram. It requires the number of clusters to be set to 1
        during initialization.

        Args:
            figure_name: The filename for saving the plot.
                               Defaults to "agnes_dendrogram.png".
            show: If True, displays the plot. Defaults to False.

        Raises:
            ValueError: If the number of clusters is not 1.

        Note:
            This method requires matplotlib and scipy to be installed.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        from scipy.cluster.hierarchy import dendrogram

        if self.n_cluster != 1:
            raise ValueError("The number of clusters should be 1 to plot dendrogram")
        # Plot the dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(np.array(self.linkage_matrix))
        plt.title("AGNES Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.savefig(f"{figure_name}", dpi=300, bbox_inches="tight")
        if show:
            plt.show()


@dataclass
class ClusterTree:
    """
    Represents a node in the hierarchical clustering tree.

    Each node is a cluster containing sample indices.
    Leaf nodes represent individual samples, while internal nodes
    represent merged clusters. The root node contains all samples.
    """

    cluster_index: int
    parent: Optional[ClusterTree] = None
    """Parent node."""
    children: list[ClusterTree] = field(default_factory=list)
    """Children nodes."""
    sample_indices: list[int] = field(default_factory=list)
    """The cluster: dataset sample indices."""
    children_cluster_distance: Optional[float] = None

    def add_child(self, child: ClusterTree) -> None:
        child.parent = self
        self.children.append(child)

    def __repr__(self) -> str:
        return f"CT({self.cluster_index}): {self.sample_indices}"


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 2], [1, 5], [1, 0], [10, 2], [10, 5], [10, 0]]
    n_cluster: int = 2
    # fit
    agnes = AGNES(n_cluster).fit(dataset)
    for i in range(n_cluster):
        print(f"Cluster[{i}]: {agnes.clusters_[i].sample_indices}")
    # fit_predict
    y_pred = AGNES(n_cluster).fit_predict(dataset)
    print("Sample labels: ", y_pred)
    # Plot the dendrogram
    agnes = AGNES(n_cluster=1).fit(dataset)
    agnes.plot_dendrogram()

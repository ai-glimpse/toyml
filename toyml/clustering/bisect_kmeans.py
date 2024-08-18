from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx

from toyml.clustering.kmeans import Kmeans
from toyml.utils.linear_algebra import sum_square_error


@dataclass
class ClusterTree:
    """
    Cluster tree node.
    The cluster tree is a binary tree, each node is a cluster.
    The root node is the whole dataset, and each node is split into two clusters until the number of clusters is equal to the specified number of clusters.
    The cluster is represented by the indices of the dataset.
    The cluster tree is used to store the cluster information and the relationship between clusters.
    """

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

    def plot(self):
        """
        Plot the cluster tree with adaptive node sizes.
        """

        def _build_graph(node, G=None, pos=None, x=0, y=0, layer=1):
            if G is None:
                G = nx.Graph()
                pos = {}

            node_id = id(node)
            G.add_node(node_id)
            pos[node_id] = (x, y)
            G.nodes[node_id]["label"] = f"{node.cluster}"
            G.nodes[node_id]["size"] = len(node.cluster) * 1000  # Adjust node size based on cluster size

            if node.left:
                left_id = id(node.left)
                G.add_edge(node_id, left_id)
                l_x, l_y = x - 1 / 2 ** (layer + 1), y - 0.5
                _build_graph(node.left, G, pos, l_x, l_y, layer + 1)

            if node.right:
                right_id = id(node.right)
                G.add_edge(node_id, right_id)
                r_x, r_y = x + 1 / 2 ** (layer + 1), y - 0.5
                _build_graph(node.right, G, pos, r_x, r_y, layer + 1)

            return G, pos

        G, pos = _build_graph(self)

        plt.figure(figsize=(12, 8))

        # Get node sizes
        node_sizes = [G.nodes[node]["size"] for node in G.nodes()]

        nx.draw(G, pos, with_labels=False, node_color="lightblue", node_size=node_sizes, arrows=False)

        labels = nx.get_node_attributes(G, "label")
        nx.draw_networkx_labels(G, pos, labels, font_size=8)  # Reduced font size for better fit

        plt.axis("off")
        plt.title("Cluster Tree")
        # plt.tight_layout()
        plt.savefig("cluster_tree.png", dpi=300, bbox_inches="tight")
        plt.show()


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
            split_cluster_into: Optional[tuple[list[int], list[int]]] = None
            for cluster_index, cluster_node in enumerate(self.cluster_tree.leaf_cluster_nodes()):
                # perform K-means with k=2
                cluster_data = [dataset[i] for i in cluster_node.cluster]
                # If the cluster cannot be split further, skip it
                if len(cluster_data) < 2:
                    continue
                # Bisect by kmeans with k=2
                cluster_unsplit_error, cluster_split_error, (cluster1, cluster2) = self._bisect_by_kmeans(
                    cluster_data, cluster_node, dataset
                )
                new_total_error = total_error - cluster_unsplit_error + cluster_split_error
                if new_total_error < total_error:
                    total_error = new_total_error
                    split_cluster_into = (cluster1, cluster2)
                    to_splot_cluster_node = cluster_node

            if to_splot_cluster_node is not None and split_cluster_into is not None:
                self._commit_split(to_splot_cluster_node, split_cluster_into)
                self.labels = self._get_dataset_labels(dataset)
            else:
                break

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

    def _bisect_by_kmeans(
        self,
        cluster_data: list[list[float]],
        cluster_node: ClusterTree,
        dataset: list[list[float]],
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
    k = 6
    # Bisecting K-means testing
    diana = BisectingKmeans(k).fit(dataset)
    print(diana.cluster_tree.get_clusters())
    print(diana.labels)
    diana.cluster_tree.plot()

from __future__ import annotations

import math
import statistics

from dataclasses import dataclass, field
from typing import Optional

from toyml.clustering.kmeans import Kmeans


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
    centroid: Optional[list[float]] = None
    """The centroid of the cluster."""

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def leaf_cluster_nodes(self) -> list[ClusterTree]:
        """Get all the leaves in the cluster tree, which are the clusters of dataset"""
        clusters = []

        def dfs(node: ClusterTree) -> None:
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

    def plot(self) -> None:  # pragma: no cover
        """
        Plot the cluster tree with adaptive node sizes.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        def _build_graph(
            node: ClusterTree,
            graph: Optional[nx.Graph] = None,
            pos: Optional[dict[int, tuple[float, float]]] = None,
            x: float = 0,
            y: float = 0,
            layer: int = 1,
        ) -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
            if graph is None:
                graph = nx.Graph()
                pos = {}

            node_id = id(node)
            graph.add_node(node_id)
            pos[node_id] = (x, y)  # type: ignore
            graph.nodes[node_id]["label"] = f"{node.cluster}"
            graph.nodes[node_id]["size"] = len(node.cluster) * 1000  # Adjust node size based on cluster size

            if node.left:
                left_id = id(node.left)
                graph.add_edge(node_id, left_id)
                l_x, l_y = x - 1 / 2 ** (layer + 1), y - 0.5
                _build_graph(node.left, graph, pos, l_x, l_y, layer + 1)

            if node.right:
                right_id = id(node.right)
                graph.add_edge(node_id, right_id)
                r_x, r_y = x + 1 / 2 ** (layer + 1), y - 0.5
                _build_graph(node.right, graph, pos, r_x, r_y, layer + 1)

            return graph, pos  # type: ignore

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

    Examples:
        >>> from toyml.clustering import BisectingKmeans
        >>> dataset = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [10.0, 1.0], [10.0, 2.0], [11.0, 1.0]]
        >>> bisect_kmeans = BisectingKmeans(k=3)
        >>> labels = bisect_kmeans.fit_predict(dataset)
        >>> clusters = bisect_kmeans.cluster_tree_.get_clusters()

    References:
        1. Harrington
        2. Tan

    See Also:
      - K-means algorithm: [toyml.clustering.kmeans][]
    """

    k: int
    """The number of clusters, specified by user."""
    cluster_tree_: ClusterTree = field(default_factory=ClusterTree)
    """The cluster tree"""
    labels_: list[int] = field(default_factory=list)
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
        self.cluster_tree_.cluster = cluster
        self.cluster_tree_.centroid = self._get_cluster_centroids(dataset, cluster)
        self.labels_ = self._predict_dataset_labels(dataset)
        total_error = sum_square_error([dataset[i] for i in cluster])
        # iterate until got k clusters
        while len(self.cluster_tree_.get_clusters()) < self.k:
            # init values for later iteration
            to_splot_cluster_node = None
            split_cluster_into: Optional[tuple[list[int], list[int]]] = None
            for cluster_index, cluster_node in enumerate(self.cluster_tree_.leaf_cluster_nodes()):
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
                self._commit_split(to_splot_cluster_node, split_cluster_into, dataset)
                self.labels_ = self._predict_dataset_labels(dataset)
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
        return self.fit(dataset).labels_

    def predict(self, points: list[list[float]]) -> list[int]:
        """
        Predict the cluster label of the given points.

        Args:
            points: A list of data points to predict.

        Returns:
            A list of predicted cluster labels for the input points.

        Raises:
            ValueError: If the model has not been fitted yet.
        """
        if self.cluster_tree_.centroid is None:
            raise ValueError("The model has not been fitted yet.")

        clusters = self.cluster_tree_.get_clusters()
        predictions = []
        for point in points:
            node = self.cluster_tree_
            while not node.is_leaf():
                if node.left is None or node.right is None:
                    raise ValueError("Invalid cluster tree structure.")

                dist_left = euclidean_distance(point, node.left.centroid)  # type: ignore[arg-type]
                dist_right = euclidean_distance(point, node.right.centroid)  # type: ignore[arg-type]

                node = node.left if dist_left < dist_right else node.right
            cluster_index = clusters.index(node.cluster)
            predictions.append(cluster_index)

        return predictions

    @staticmethod
    def _bisect_by_kmeans(
        cluster_data: list[list[float]],
        cluster_node: ClusterTree,
        dataset: list[list[float]],
    ) -> tuple[float, float, tuple[list[int], list[int]]]:
        kmeans = Kmeans(k=2).fit(cluster_data)
        assert kmeans.clusters_ is not None
        cluster1, cluster2 = kmeans.clusters_[0], kmeans.clusters_[1]
        # Note: map the cluster's inner index to the truth index in dataset
        cluster1 = [cluster_node.cluster[i] for i in cluster1]
        cluster2 = [cluster_node.cluster[i] for i in cluster2]
        # split error calculation
        cluster_unsplit_error = sum_square_error([dataset[i] for i in cluster_node.cluster])
        cluster_split_error = sum_square_error([dataset[i] for i in cluster1]) + sum_square_error(
            [dataset[i] for i in cluster2]
        )
        return cluster_unsplit_error, cluster_split_error, (cluster1, cluster2)

    def _commit_split(
        self,
        cluster_node: ClusterTree,
        split_cluster_into: tuple[list[int], list[int]],
        dataset: list[list[float]],
    ) -> None:
        # cluster tree
        cluster_node.add_left_child(
            ClusterTree(
                cluster=split_cluster_into[0], centroid=self._get_cluster_centroids(dataset, split_cluster_into[0])
            )
        )
        cluster_node.add_right_child(
            ClusterTree(
                cluster=split_cluster_into[1], centroid=self._get_cluster_centroids(dataset, split_cluster_into[1])
            )
        )

    def _predict_dataset_labels(self, dataset: list[list[float]]) -> list[int]:
        labels = [-1] * len(dataset)
        for cluster_label, cluster in enumerate(self.cluster_tree_.get_clusters()):
            for data_point_index in cluster:
                labels[data_point_index] = cluster_label
        return labels

    @staticmethod
    def _get_cluster_centroids(dataset: list[list[float]], cluster: list[int]) -> list[float]:
        cluster_points = [dataset[i] for i in cluster]
        centroid = [sum(t) / len(cluster) for t in zip(*cluster_points)]
        return centroid


def euclidean_distance(v1: list[float], v2: list[float]) -> float:
    """
    Calculate the L2 distance between two vectors

    """
    assert len(v1) == len(v2), f"{v1} and {v2} have different length!"
    return math.sqrt(sum(pow(v1[i] - v2[i], 2) for i in range(len(v1))))


def sum_square_error(c: list[list[float]]) -> float:
    """
    Calc the sum of squared errors.
    """
    mean_c = [statistics.mean([v[i] for v in c]) for i in range(len(c[0]))]
    return sum(euclidean_distance(mean_c, v) ** 2 for v in c)


if __name__ == "__main__":
    dataset: list[list[float]] = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [10.0, 1.0], [10.0, 2.0], [11.0, 1.0]]
    k = 6
    # Bisecting K-means testing
    diana = BisectingKmeans(k).fit(dataset)
    print(diana.cluster_tree_.get_clusters())
    print(diana.labels_)
    # diana.cluster_tree.plot()
    print(diana.predict(dataset))

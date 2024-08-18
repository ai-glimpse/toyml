from __future__ import annotations

import pytest

from toyml.clustering.bisect_kmeans import BisectingKmeans, ClusterTree


class TestBisectKMeansSimple:
    """
    Test the k-means algorithm completely.
    """

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_fit(
        self,
        k: int,
        simple_dataset: list[list[float]],
    ) -> None:
        diana = BisectingKmeans(k)
        print(diana.cluster_tree.get_clusters())
        assert len(diana.cluster_tree.get_clusters()) == 1
        if k <= len(simple_dataset):
            diana = diana.fit(simple_dataset)
            assert len(diana.cluster_tree.get_clusters()) == k
            assert len(set(diana.labels)) == k
            assert all(0 <= label <= k for label in diana.labels) is True

            cluster_index = [i for cluster in diana.cluster_tree.get_clusters() for i in cluster]
            assert len(cluster_index) == len(simple_dataset)
            assert sorted(cluster_index) == sorted(list(range(len(simple_dataset))))
        else:
            with pytest.raises(ValueError):
                diana.fit(simple_dataset)


class TestClusterTree:
    def test_init(self) -> None:
        tree = ClusterTree()
        assert tree.is_root() is True
        assert tree.is_leaf() is True
        assert tree.cluster == []

    def test_leaf_cluster_nodes(self):
        sample_indices = [1, 2, 3, 4, 5]

        tree = ClusterTree(cluster=sample_indices)
        leaf_cluster_nodes = tree.leaf_cluster_nodes()
        assert len(leaf_cluster_nodes) == 1
        assert leaf_cluster_nodes[0].cluster == sample_indices

        tree.add_left_child(ClusterTree(cluster=[1, 2, 3]))
        leaf_cluster_nodes = tree.leaf_cluster_nodes()
        assert len(leaf_cluster_nodes) == 1
        assert leaf_cluster_nodes[0].cluster == [1, 2, 3]

        tree.add_right_child(ClusterTree(cluster=[4, 5]))
        leaf_cluster_nodes = tree.leaf_cluster_nodes()
        assert len(leaf_cluster_nodes) == 2
        assert leaf_cluster_nodes[0].cluster == [1, 2, 3]
        assert leaf_cluster_nodes[1].cluster == [4, 5]

        tree.left.add_left_child(ClusterTree(cluster=[1, 2]))  # type: ignore[union-attr]
        leaf_cluster_nodes = tree.leaf_cluster_nodes()  # type: ignore[union-attr]
        assert len(leaf_cluster_nodes) == 2
        assert leaf_cluster_nodes[0].cluster == [1, 2]
        assert leaf_cluster_nodes[1].cluster == [4, 5]

        tree.left.add_right_child(ClusterTree(cluster=[3]))  # type: ignore[union-attr]
        leaf_cluster_nodes = tree.leaf_cluster_nodes()
        assert len(leaf_cluster_nodes) == 3
        assert leaf_cluster_nodes[0].cluster == [1, 2]
        assert leaf_cluster_nodes[1].cluster == [3]
        assert leaf_cluster_nodes[2].cluster == [4, 5]

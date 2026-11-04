from __future__ import annotations

from pathlib import Path

import pytest

from toyml.clustering.bisect_kmeans import BisectingKmeans, ClusterTree


class TestBisectKMeansSimple:
    """Test the k-means algorithm completely."""

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_fit(
        self,
        k: int,
        simple_dataset: list[list[float]],
    ) -> None:
        diana = BisectingKmeans(k)
        print(diana.cluster_tree_.get_clusters())
        assert len(diana.cluster_tree_.get_clusters()) == 1
        if k <= len(simple_dataset):
            diana = diana.fit(simple_dataset)
            assert len(diana.cluster_tree_.get_clusters()) == k
            assert len(set(diana.labels_)) == k
            assert all(0 <= label <= k for label in diana.labels_) is True

            cluster_index = [i for cluster in diana.cluster_tree_.get_clusters() for i in cluster]
            assert len(cluster_index) == len(simple_dataset)
            assert sorted(cluster_index) == sorted(range(len(simple_dataset)))
        else:
            with pytest.raises(ValueError):  # noqa: PT011
                diana.fit(simple_dataset)

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
    def test_predict(
        self,
        k: int,
        simple_dataset: list[list[float]],
    ) -> None:
        """Test the predict method."""
        # Fit the model
        diana = BisectingKmeans(k).fit(simple_dataset)

        # Predict on the same dataset
        predictions = diana.predict(simple_dataset)

        # Check if the number of predictions matches the dataset size
        assert len(predictions) == len(simple_dataset)

        # Check if all predicted labels are within the valid range
        assert all(0 <= label < k for label in predictions)

        # Check if the predictions match the labels from fit
        assert predictions == diana.labels_

        # Test prediction on new points
        new_points = [[0.5, 0.5], [10.5, 10.5]]
        new_predictions = diana.predict(new_points)

        # Check if the number of new predictions is correct
        assert len(new_predictions) == len(new_points)

        # Check if new predictions are within the valid range
        assert all(0 <= label < k for label in new_predictions)


class TestClusterTree:
    def test_init(self) -> None:
        tree = ClusterTree()
        assert tree.is_root() is True
        assert tree.is_leaf() is True
        assert tree.cluster == []

    def test_leaf_cluster_nodes(self) -> None:
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

        assert tree.left is not None
        tree.left.add_left_child(ClusterTree(cluster=[1, 2]))
        leaf_cluster_nodes = tree.leaf_cluster_nodes()
        assert len(leaf_cluster_nodes) == 2
        assert leaf_cluster_nodes[0].cluster == [1, 2]
        assert leaf_cluster_nodes[1].cluster == [4, 5]

        tree.left.add_right_child(ClusterTree(cluster=[3]))
        leaf_cluster_nodes = tree.leaf_cluster_nodes()
        assert len(leaf_cluster_nodes) == 3
        assert leaf_cluster_nodes[0].cluster == [1, 2]
        assert leaf_cluster_nodes[1].cluster == [3]
        assert leaf_cluster_nodes[2].cluster == [4, 5]

    def test_plot(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import matplotlib.pyplot as plt

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(plt, "show", lambda: None)
        tree = ClusterTree(cluster=[0, 1])
        tree.add_left_child(ClusterTree(cluster=[0]))
        tree.add_right_child(ClusterTree(cluster=[1]))
        try:
            tree.plot()
            assert (tmp_path / "cluster_tree.png").stat().st_size > 0
        finally:
            plt.close("all")


@pytest.mark.parametrize("missing", ["left", "right", "left-centroid", "right-centroid"])
def test_predict_rejects_incomplete_tree(missing: str) -> None:
    model = BisectingKmeans(2)
    left = ClusterTree(cluster=[0], centroid=[0.0])
    right = ClusterTree(cluster=[1], centroid=[1.0])
    model.cluster_tree_ = ClusterTree(cluster=[0, 1], centroid=[0.5])
    if missing != "left":
        model.cluster_tree_.add_left_child(left)
    if missing != "right":
        model.cluster_tree_.add_right_child(right)
    if missing == "left-centroid":
        left.centroid = None
    if missing == "right-centroid":
        right.centroid = None
    with pytest.raises(ValueError, match="Invalid cluster tree structure"):
        model.predict([[0.0]])

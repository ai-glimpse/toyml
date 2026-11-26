from dataclasses import fields
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
lazy import matplotlib.pyplot as plt

from toyml.clustering.bisecting_kmeans import BisectingKMeans, ClusterTree

TRAINING_DATA = st.lists(st.integers(1, 30), min_size=4, max_size=8, unique=True).map(
    lambda xs: [[float(x), float(x % 5)] for x in xs]
)
PROPERTY = settings(max_examples=30, deadline=None, derandomize=True, database=None)


class TestBisectingKMeans:
    """Test the k-means algorithm completely."""

    def test_identical_samples_remain_in_one_leaf(self) -> None:
        """An empty child leaves the parent intact, even when two clusters were requested."""
        samples = [[1.0], [1.0]]
        model = BisectingKMeans(k=2)
        assert model.fit_predict(samples) == [0, 0]
        assert model.cluster_tree_.is_leaf()
        assert model.predict(samples) == [0, 0]

    def test_unsplittable_leaf_does_not_block_other_splits(self) -> None:
        """Keep the duplicates together and continue splitting the two distinct neighbors."""
        samples = [[0.0], [0.0], [10.0], [11.0]]
        model = BisectingKMeans(k=4).fit(samples)
        assert sorted(model.cluster_tree_.get_clusters()) == [[0, 1], [2], [3]]
        assert model.predict(samples) == model.labels_

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_fit(
        self,
        k: int,
        simple_dataset: list[list[float]],
    ) -> None:
        model = BisectingKMeans(k)
        print(model.cluster_tree_.get_clusters())
        assert len(model.cluster_tree_.get_clusters()) == 1
        if k <= len(simple_dataset):
            model = model.fit(simple_dataset)
            assert len(model.cluster_tree_.get_clusters()) == k
            assert len(set(model.labels_)) == k
            assert all(0 <= label <= k for label in model.labels_) is True

            cluster_index = [i for cluster in model.cluster_tree_.get_clusters() for i in cluster]
            assert len(cluster_index) == len(simple_dataset)
            assert sorted(cluster_index) == sorted(range(len(simple_dataset)))
        else:
            with pytest.raises(ValueError):  # noqa: PT011
                model.fit(simple_dataset)

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
    def test_predict(
        self,
        k: int,
        simple_dataset: list[list[float]],
    ) -> None:
        """Test the predict method."""
        # Fit the model
        model = BisectingKMeans(k).fit(simple_dataset)

        # Predict on the same dataset
        predictions = model.predict(simple_dataset)

        # Check if the number of predictions matches the dataset size
        assert len(predictions) == len(simple_dataset)

        # Check if all predicted labels are within the valid range
        assert all(0 <= label < k for label in predictions)

        # Check if the predictions match the labels from fit
        assert predictions == model.labels_

        # Test prediction on new points
        new_points = [[0.5, 0.5], [10.5, 10.5]]
        new_predictions = model.predict(new_points)

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
    model = BisectingKMeans(2)
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


@given(offset=st.integers(-20, 20))
def test_refit_rebuilds_tree_for_shorter_data(offset: int) -> None:
    first = [[float(offset + i)] for i in range(4)]
    second = [[float(offset + 10)], [float(offset + 20)]]
    model = BisectingKMeans(k=2).fit(first).fit(second)
    fresh = BisectingKMeans(k=2).fit(second)
    assert sorted(map(sorted, model.cluster_tree_.get_clusters())) == sorted(
        map(sorted, fresh.cluster_tree_.get_clusters())
    )
    assert len(model.labels_) == 2
    centroids = []
    for node in model.cluster_tree_.leaf_cluster_nodes():
        assert node.centroid is not None
        centroids.append(node.centroid)
    assert sorted(centroids) == second


def test_constructor_excludes_training_state() -> None:
    model = BisectingKMeans(k=2)
    assert all(not f.init for f in fields(model) if f.name.endswith("_") or f.name.startswith("_"))


@given(value=st.integers(-10, 10))
def test_unfitted_bisecting_kmeans_inference_fails(value: int) -> None:
    with pytest.raises(ValueError, match="not.*fitted"):
        BisectingKMeans(k=2).predict([[float(value)]])


@PROPERTY
@given(data=TRAINING_DATA)
def test_fit_predict_returns_independent_labels(data: list[list[float]]) -> None:
    model = BisectingKMeans(k=2)
    labels = model.fit_predict(data)
    original = model.labels_.copy()
    labels[:] = [99] * len(labels)
    assert model.labels_ == original


def test_fit_preserves_input() -> None:
    samples = [[0.0], [0.1], [4.0], [4.1]]
    original = [row.copy() for row in samples]
    BisectingKMeans(k=2).fit(samples)
    assert samples == original

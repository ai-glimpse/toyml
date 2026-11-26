from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from toyml.clustering.agnes import AGNES, ClusterTree

TRAINING_DATA = st.lists(st.integers(1, 30), min_size=4, max_size=8, unique=True).map(
    lambda xs: [[float(x), float(x % 5)] for x in xs]
)
PROPERTY = settings(max_examples=30, deadline=None, derandomize=True, database=None)


class TestAGNES:
    @pytest.fixture
    def sample_dataset(self) -> list[list[float]]:
        return [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [10.0, 0.0], [10.0, 1.0], [10.0, 2.0]]

    def test_fit_predict(self, sample_dataset: list[list[float]]) -> None:
        agnes = AGNES(n_cluster=2)
        labels = agnes.fit_predict(sample_dataset)
        assert len(labels) == len(sample_dataset)
        assert set(labels) == {0, 1}

        # Check that the first 3 points are in the same cluster
        assert len(set(labels[:3])) == 1

        # Check that the last 3 points are in the same cluster
        assert len(set(labels[3:])) == 1

        # Check that the first 3 points and last 3 points are in different clusters
        assert labels[0] != labels[3]

    def test_linkage_methods(self, sample_dataset: list[list[float]]) -> None:
        for linkage in ("single", "complete", "average"):
            agnes = AGNES(n_cluster=2, linkage=linkage)
            labels = agnes.fit_predict(sample_dataset)
            assert len(labels) == len(sample_dataset)
            assert set(labels) == {0, 1}

    def test_invalid_linkage(self) -> None:
        with pytest.raises(ValueError, match="Invalid linkage method"):
            # Deliberately violate the annotation to check runtime validation.
            AGNES(n_cluster=2, linkage="invalid")  # ty: ignore[invalid-argument-type]

    def test_cluster_tree(self) -> None:
        dataset = [[1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0]]
        agnes = AGNES(n_cluster=1).fit(dataset)
        assert isinstance(agnes.cluster_tree_, ClusterTree)
        print(agnes.cluster_tree_.children)
        assert len(agnes.cluster_tree_.children) == 2
        assert len(agnes.linkage_matrix_) == 3  # n-1 merges for n data points

    @pytest.mark.parametrize("n_cluster", [2, 3, 4, 5])
    def test_distance_matrix(self, sample_dataset: list[list[float]], n_cluster: int) -> None:
        agnes = AGNES(n_cluster=n_cluster)
        # before fit
        agnes.distance_matrix_ = agnes._get_init_distance_matrix(sample_dataset)
        assert len(agnes.distance_matrix_) == len(sample_dataset)
        assert all(len(row) == len(sample_dataset) for row in agnes.distance_matrix_)
        assert all(agnes.distance_matrix_[i][i] == 0 for i in range(len(sample_dataset)))
        # after fit: the distance matrix's size should be reduced
        agnes = agnes.fit(sample_dataset)
        assert len(agnes.distance_matrix_) == n_cluster
        assert all(len(row) == n_cluster for row in agnes.distance_matrix_)
        assert all(agnes.distance_matrix_[i][i] == 0 for i in range(n_cluster))

    def test_n_cluster_validation(self) -> None:
        dataset = [[1.0, 0.0], [2.0, 0.0]]
        with pytest.raises(ValueError, match="The number of clusters should be less than the number of data points"):
            AGNES(n_cluster=3).fit(dataset)  # n_cluster > number of data points

    def test_dataset_rows_validation(self) -> None:
        dataset = [[1.0, 0.0], [2.0], [3.0, 0.0], [4.0, 0.0]]
        with pytest.raises(ValueError, match="All samples should have the same dimension"):
            AGNES(n_cluster=1).fit(dataset)

    def test_single_point(self) -> None:
        dataset = [[1.0, 0.0]]
        agnes = AGNES(n_cluster=1).fit(dataset)
        assert agnes.labels_ == [0]

    @pytest.mark.parametrize("n_cluster", [1, 2, 3, 4, 5])
    def test_different_n_clusters(self, sample_dataset: list[list[float]], n_cluster: int) -> None:
        agnes = AGNES(n_cluster=n_cluster).fit(sample_dataset)
        assert len(set(agnes.labels_)) == n_cluster

    def test_plot_dendrogram(self, sample_dataset: list[list[float]], tmp_path: Path) -> None:
        agnes = AGNES(n_cluster=1).fit(sample_dataset)
        figure_path = tmp_path / "test_dendrogram.png"
        agnes.plot_dendrogram(str(figure_path), show=False)
        assert figure_path.exists()

    def test_plot_dendrogram_invalid_n_cluster(self, sample_dataset: list[list[float]]) -> None:
        agnes = AGNES(n_cluster=2).fit(sample_dataset)
        with pytest.raises(ValueError, match="The number of clusters should be 1"):
            agnes.plot_dendrogram()

    def test_linkage_matrix(self, sample_dataset: list[list[float]]) -> None:
        agnes = AGNES(n_cluster=1).fit(sample_dataset)
        assert len(agnes.linkage_matrix_) == len(sample_dataset) - 1
        for row in agnes.linkage_matrix_:
            assert len(row) == 4  # Each row should have 4 elements

    def test_cluster_indices(self, sample_dataset: list[list[float]]) -> None:
        agnes = AGNES(n_cluster=2).fit(sample_dataset)
        all_indices = set()
        for cluster in agnes.clusters_:
            all_indices.update(cluster.sample_indices)
        assert all_indices == set(range(len(sample_dataset)))


def tree_state(value: object) -> object:
    """Compare fitted trees without following their parent links back into cycles."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: tree_state(getattr(value, f.name)) for f in fields(value) if f.name != "parent"}
    if isinstance(value, list):
        return [tree_state(item) for item in value]
    if isinstance(value, dict):
        return {key: tree_state(item) for key, item in value.items()}
    return value


@PROPERTY
@given(first=TRAINING_DATA, second=TRAINING_DATA)
def test_agnes_refit_preserves_configuration_and_replaces_state(
    first: list[list[float]], second: list[list[float]]
) -> None:
    model = AGNES(n_cluster=2)
    fresh = AGNES(n_cluster=2)
    model.fit(first)
    model.fit(second)
    for field in fields(model):
        if field.init:
            assert getattr(model, field.name) == getattr(fresh, field.name)
    fresh.fit(second)
    assert tree_state(model) == tree_state(fresh)


@PROPERTY
@given(data=TRAINING_DATA)
def test_agnes_fit_owns_its_data(data: list[list[float]]) -> None:
    original = [row.copy() for row in data]
    model = AGNES(n_cluster=2)
    model.fit(data)
    assert data == original
    trained = tree_state(model)
    data[0][0] += 100
    data.clear()
    assert tree_state(model) == trained


def test_constructor_excludes_training_state() -> None:
    model = AGNES(n_cluster=2)
    assert all(not f.init for f in fields(model) if f.name.endswith("_") or f.name.startswith("_"))


@PROPERTY
@given(data=TRAINING_DATA)
def test_fit_predict_returns_independent_labels(data: list[list[float]]) -> None:
    model = AGNES(n_cluster=2)
    labels = model.fit_predict(data)
    original = model.labels_.copy()
    labels[:] = [99] * len(labels)
    assert model.labels_ == original


def test_fit_preserves_input() -> None:
    samples = [[0.0], [0.1], [4.0], [4.1]]
    original = [row.copy() for row in samples]
    AGNES(n_cluster=2).fit(samples)
    assert samples == original

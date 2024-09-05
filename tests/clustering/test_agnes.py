from pathlib import Path

import pytest

from toyml.clustering.agnes import AGNES, ClusterTree


class TestAGNES:
    @pytest.fixture  # type: ignore
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
        for linkage in ["single", "complete", "average"]:
            agnes = AGNES(n_cluster=2, linkage=linkage)  # type: ignore
            labels = agnes.fit_predict(sample_dataset)
            assert len(labels) == len(sample_dataset)
            assert set(labels) == {0, 1}

    def test_invalid_linkage(self) -> None:
        with pytest.raises(ValueError, match="Invalid linkage method"):
            AGNES(n_cluster=2, linkage="invalid")  # type: ignore

    def test_cluster_tree(self) -> None:
        dataset = [[1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0]]
        agnes = AGNES(n_cluster=1).fit(dataset)
        assert isinstance(agnes.cluster_tree_, ClusterTree)
        print(agnes.cluster_tree_.children)
        assert len(agnes.cluster_tree_.children) == 2
        assert len(agnes.linkage_matrix) == 3  # n-1 merges for n data points

    @pytest.mark.parametrize("n_cluster", [2, 3, 4, 5])  # type: ignore
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

    @pytest.mark.parametrize("n_cluster", [1, 2, 3, 4, 5])  # type: ignore
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
        assert len(agnes.linkage_matrix) == len(sample_dataset) - 1
        for row in agnes.linkage_matrix:
            assert len(row) == 4  # Each row should have 4 elements

    def test_cluster_indices(self, sample_dataset: list[list[float]]) -> None:
        agnes = AGNES(n_cluster=2).fit(sample_dataset)
        all_indices = set()
        for cluster in agnes.clusters_:
            all_indices.update(cluster.sample_indices)
        assert all_indices == set(range(len(sample_dataset)))

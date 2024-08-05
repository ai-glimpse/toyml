from __future__ import annotations

from toyml.clustering import Kmeans


class TestKMeansSimple:
    def test_fit(
        self,
        simple_dataset: list[list[float]],
    ) -> None:
        k = 2
        max_iter = 10
        dataset = simple_dataset

        kmeans = Kmeans(k, max_iter)
        assert kmeans.clusters_ is None
        assert kmeans.centroids_ is None
        kmeans.fit(dataset)
        assert kmeans.clusters_ is not None
        assert kmeans.centroids_ is not None

        # check clusters
        first_sample_cluster, last_sample_cluster = kmeans.predict(dataset[0]), kmeans.predict(dataset[-1])
        assert kmeans.clusters_[first_sample_cluster] == [0, 1, 2]
        assert kmeans.clusters_[last_sample_cluster] == [3, 4, 5]

        # check prediction
        x_sample = [0] * len(dataset[0])
        assert kmeans.predict(x_sample) == first_sample_cluster

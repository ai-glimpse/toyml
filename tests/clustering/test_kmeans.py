from __future__ import annotations

from typing import Literal

import pytest

from toyml.clustering import Kmeans


class TestKMeansSimple:
    """
    Test the k-means algorithm completely.
    """

    @pytest.mark.parametrize("centroids_init_method,", ["random", "kmeans++"])
    def test_fit(
        self,
        centroids_init_method: Literal["random", "kmeans++"],
        simple_dataset: list[list[float]],
    ) -> None:
        k = 2
        max_iter = 10
        dataset = simple_dataset

        kmeans = Kmeans(k, max_iter, 1e-5, centroids_init_method)
        assert kmeans.clusters is None
        assert kmeans.centroids is None
        kmeans.fit(dataset)
        assert kmeans.clusters is not None
        assert kmeans.centroids is not None

        # check clusters
        first_sample_cluster, last_sample_cluster = kmeans.predict(dataset[0]), kmeans.predict(dataset[-1])
        assert kmeans.clusters[first_sample_cluster] == [0, 1, 2]
        assert kmeans.clusters[last_sample_cluster] == [3, 4, 5]

        # check prediction
        x_sample = [0] * len(dataset[0])
        assert kmeans.predict(x_sample) == first_sample_cluster

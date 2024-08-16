from __future__ import annotations

import pytest

from toyml.clustering import BisectingKmeans


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
        kmeans = BisectingKmeans(k)
        assert len(kmeans.clusters) == 0
        if k <= len(simple_dataset):
            kmeans.fit(simple_dataset)
            assert len(kmeans.clusters) == k
            cluster_index = [i for cluster in kmeans.clusters for i in cluster]
            assert len(cluster_index) == len(simple_dataset)
            assert sorted(cluster_index) == sorted(list(range(len(simple_dataset))))
        else:
            with pytest.raises(ValueError):
                kmeans.fit(simple_dataset)

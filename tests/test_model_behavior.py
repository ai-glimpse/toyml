from collections.abc import Callable

import pytest

from toyml.clustering import AGNES, DBSCAN, BisectingKMeans
from toyml.ensemble import IsolationForest


@pytest.mark.parametrize(
    "fit",
    [
        lambda samples: AGNES(n_cluster=2).fit(samples),
        lambda samples: DBSCAN(eps=0.2, min_samples=2).fit(samples),
        lambda samples: BisectingKMeans(k=2).fit(samples),
        lambda samples: IsolationForest(n_itree=5, random_seed=42).fit(samples),
    ],
    ids=["agnes", "dbscan", "bisecting-kmeans", "isolation-forest"],
)
def test_clustering_fit_preserves_input(fit: Callable[[list[list[float]]], object]) -> None:
    samples = [[0.0], [0.1], [4.0], [4.1]]
    original = [row.copy() for row in samples]
    fit(samples)
    assert samples == original

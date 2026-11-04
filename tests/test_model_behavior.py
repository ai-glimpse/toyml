from collections.abc import Callable
from typing import Literal

import pytest
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import adjusted_rand_score

from toyml.clustering import AGNES, DBSCAN, BisectingKMeans, KMeans
from toyml.ensemble import IsolationForest


@pytest.mark.parametrize("initialization", ["random", "kmeans++"])
def test_kmeans_partition_matches_sklearn(initialization: Literal["random", "kmeans++"]) -> None:
    samples = [[0.0, 0.0], [0.1, 0.2], [0.2, 0.1], [9.0, 9.0], [9.1, 9.2], [9.2, 9.1]]
    actual = KMeans(k=2, centroids_init_method=initialization, random_seed=42).fit_predict(samples)
    expected = SklearnKMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(samples)
    assert adjusted_rand_score(expected, actual) == 1.0


@pytest.mark.parametrize(
    "fit",
    [
        lambda samples: AGNES(n_cluster=2).fit(samples),
        lambda samples: DBSCAN(eps=0.2, min_samples=2).fit(samples),
        lambda samples: BisectingKMeans(k=2).fit(samples),
        lambda samples: KMeans(k=2, random_seed=42).fit(samples),
        lambda samples: IsolationForest(n_itree=5, random_seed=42).fit(samples),
    ],
    ids=["agnes", "dbscan", "bisecting-kmeans", "kmeans", "isolation-forest"],
)
def test_clustering_fit_preserves_input(fit: Callable[[list[list[float]]], object]) -> None:
    samples = [[0.0], [0.1], [4.0], [4.1]]
    original = [row.copy() for row in samples]
    fit(samples)
    assert samples == original

"""Behavior regressions deferred from the layout and naming migration.

Each strict xfail must become an ordinary passing test when its algorithm is fixed.
"""

import pytest

from toyml.clustering import DBSCAN, KMeans
from toyml.ensemble import AdaBoost, IsolationForest
from toyml.ensemble.adaboost import OneDimensionClassifier


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DBSCAN records non-core border points as noise")
def test_dbscan_noise_excludes_border_points() -> None:
    model = DBSCAN(eps=0.21, min_samples=4).fit([[0.0], [0.1], [0.2], [0.3], [0.6]])
    assert model.noises_ == [i for i, label in enumerate(model.labels_) if label == -1]


@pytest.mark.xfail(
    strict=True, raises=ZeroDivisionError, reason="KMeans++ divides by zero when all distance weights are zero"
)
def test_kmeans_plus_plus_handles_identical_samples() -> None:
    labels = KMeans(k=2, centroids_init_method="kmeans++", random_seed=42).fit_predict([[1.0], [1.0], [1.0]])
    assert len(labels) == 3
    assert len(set(labels)) == 1


@pytest.mark.xfail(strict=True, raises=ZeroDivisionError, reason="AdaBoost divides by zero for a perfect weak learner")
def test_adaboost_handles_perfect_weak_classifier() -> None:
    samples = [[0.0], [1.0], [2.0], [3.0]]
    labels = [-1, -1, 1, 1]
    model = AdaBoost(weak_learner=OneDimensionClassifier, n_weak_learner=1).fit(samples, labels)
    assert [model.predict(sample) for sample in samples] == labels


@pytest.mark.xfail(
    strict=True,
    raises=ZeroDivisionError,
    reason="IsolationForest accepts one sample and divides by zero during scoring",
)
def test_isolation_forest_rejects_single_sample() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        IsolationForest(random_seed=42).fit_predict([[1.0]])

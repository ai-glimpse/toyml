"""Behavior regressions deferred from the layout and naming migration.

Each strict xfail must become an ordinary passing test when its algorithm is fixed.
"""

import random
from copy import deepcopy
from random import Random

import pytest
from sklearn.naive_bayes import GaussianNB

from toyml.classification import KNN
from toyml.classification.naive_bayes import GaussianNaiveBayes
from toyml.clustering import DBSCAN, KMeans
from toyml.ensemble import AdaBoost, IsolationForest
from toyml.ensemble.adaboost import OneDimensionClassifier


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="KNN standardization modifies the caller's training data")
def test_knn_fit_preserves_input() -> None:
    samples = [[0.0], [1.0], [2.0]]
    original = deepcopy(samples)
    KNN(k=1).fit(samples, [0, 0, 1])
    assert samples == original


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="KNN standardization modifies the caller's query")
def test_knn_predict_preserves_input() -> None:
    model = KNN(k=1).fit([[0.0], [1.0], [2.0]], [0, 0, 1])
    query = [0.2]
    model.predict(query)
    assert query == [0.2]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DBSCAN returns an empty label list for all-noise inputs")
def test_dbscan_labels_every_noise_sample() -> None:
    assert DBSCAN(eps=0.1, min_samples=2).fit_predict([[0.0], [10.0]]) == [-1, -1]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DBSCAN records non-core border points as noise")
def test_dbscan_noise_excludes_border_points() -> None:
    model = DBSCAN(eps=0.21, min_samples=4).fit([[0.0], [0.1], [0.2], [0.3], [0.6]])
    assert model.noises_ == [i for i, label in enumerate(model.labels_) if label == -1]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DBSCAN retains clusters from its previous fit")
def test_dbscan_refit_matches_fresh_model() -> None:
    samples = [[0.0], [0.1], [4.0], [4.1]]
    model = DBSCAN(eps=0.2, min_samples=2).fit(samples)
    assert model.fit_predict(samples) == DBSCAN(eps=0.2, min_samples=2).fit_predict(samples)


@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="GaussianNaiveBayes uses uniform rather than empirical class priors"
)
def test_gaussian_naive_bayes_imbalanced_priors_match_sklearn() -> None:
    samples = [[-1.0], [1.0]] * 5
    labels = [0] * 8 + [1] * 2
    expected = GaussianNB(var_smoothing=0).fit(samples, labels).predict_proba([[0.0]])[0]
    actual = GaussianNaiveBayes(unbiased_variance=False, var_smoothing=0).fit(samples, labels).predict_proba([0.0])
    assert [actual[0], actual[1]] == pytest.approx(expected)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="KMeans seeds the process-wide random generator")
def test_kmeans_preserves_global_random_state() -> None:
    state = random.getstate()
    try:
        KMeans(k=2, random_seed=42).fit([[0.0], [0.1], [4.0], [4.1]])
        assert random.getstate() == state
    finally:
        random.setstate(state)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="IsolationForest does not seed child-tree randomness")
def test_isolation_forest_seed_is_independent_of_system_entropy(monkeypatch: pytest.MonkeyPatch) -> None:
    samples = [[float(i), float(i % 7)] for i in range(40)]

    def scores(entropy_seed: int) -> list[float]:
        monkeypatch.setattr(random, "Random", lambda seed=None: Random(entropy_seed if seed is None else seed))
        model = IsolationForest(n_itree=10, max_samples=20, random_seed=42).fit(samples)
        return [model.score(sample) for sample in samples]

    assert scores(0) == scores(1)


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

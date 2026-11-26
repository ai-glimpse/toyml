import math
from dataclasses import asdict, fields

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score

from toyml.clustering.dbscan import DBSCAN, Dataset

TRAINING_DATA = st.lists(st.integers(1, 30), min_size=4, max_size=8, unique=True).map(
    lambda xs: [[float(x), float(x % 5)] for x in xs]
)
PROPERTY = settings(max_examples=30, deadline=None, derandomize=True, database=None)


@pytest.fixture
def dataset() -> list[list[float]]:
    return [[1.0, 2.0], [2.0, 2.0], [2.0, 3.0], [8.0, 7.0], [8.0, 8.0], [25.0, 80.0]]


def test_clustering(dataset: list[list[float]]) -> None:
    model = DBSCAN(eps=3, min_samples=2).fit(dataset)
    assert {frozenset(cluster) for cluster in model.clusters_} == {frozenset({0, 1, 2}), frozenset({3, 4})}
    assert model.noises_ == [5]


@pytest.mark.parametrize(("eps", "expected_clusters"), [(0.1, 0), (100.0, 1)])
def test_cluster_counts(dataset: list[list[float]], eps: float, expected_clusters: int) -> None:
    model = DBSCAN(eps=eps, min_samples=2).fit(dataset)
    assert len(model.clusters_) == expected_clusters
    if expected_clusters:
        assert set(model.clusters_[0]) == set(range(len(dataset)))


def test_distance_matrix(dataset: list[list[float]]) -> None:
    samples = Dataset(dataset)
    assert samples.n == 6
    assert len(samples.distance_matrix_) == 6
    assert all(len(row) == 6 for row in samples.distance_matrix_)
    assert samples.distance_matrix_[0][2] == pytest.approx(math.sqrt(2))
    assert set(samples.get_neighbors(0, 2.0)) == {1, 2}
    core_objects, non_core_objects = samples.get_core_objects(3, 2)
    assert core_objects == {0, 1, 2, 3, 4}
    assert non_core_objects == [5]


@pytest.mark.parametrize(("eps", "min_samples"), [(3.0, 2), (100.0, 2), (3.0, 3)])
def test_fit_predict_matches_sklearn(dataset: list[list[float]], eps: float, min_samples: int) -> None:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(dataset)
    expected = SklearnDBSCAN(eps=eps, min_samples=min_samples).fit_predict(dataset).tolist()
    assert adjusted_rand_score(expected, labels) == 1.0
    assert [i for i, label in enumerate(labels) if label == -1] == [
        i for i, label in enumerate(expected) if label == -1
    ]
    assert model.labels_ == labels


@given(offset=st.integers(-20, 20), count=st.integers(2, 8))
def test_refit_to_all_noise_replaces_previous_clusters(offset: int, count: int) -> None:
    model = DBSCAN(eps=0.1, min_samples=2).fit([[0.0], [0.01]])
    samples = [[float(offset + i)] for i in range(count)]
    assert model.fit_predict(samples) == [-1] * count
    assert model.clusters_ == []
    assert model.noises_ == list(range(count))


@PROPERTY
@given(first=TRAINING_DATA, second=TRAINING_DATA)
def test_dbscan_refit_preserves_configuration_and_replaces_state(
    first: list[list[float]], second: list[list[float]]
) -> None:
    model = DBSCAN(eps=2, min_samples=1)
    fresh = DBSCAN(eps=2, min_samples=1)
    model.fit(first)
    model.fit(second)
    for field in fields(model):
        if field.init:
            assert getattr(model, field.name) == getattr(fresh, field.name)
    fresh.fit(second)
    assert asdict(model) == asdict(fresh)


@PROPERTY
@given(data=TRAINING_DATA)
def test_dbscan_fit_owns_its_data(data: list[list[float]]) -> None:
    original = [row.copy() for row in data]
    model = DBSCAN(eps=2, min_samples=1)
    model.fit(data)
    assert data == original
    trained = asdict(model)
    data[0][0] += 100
    data.clear()
    assert asdict(model) == trained


def test_constructor_excludes_training_state() -> None:
    model = DBSCAN(eps=2, min_samples=1)
    assert all(not f.init for f in fields(model) if f.name.endswith("_") or f.name.startswith("_"))


@PROPERTY
@given(data=TRAINING_DATA)
def test_fit_predict_returns_independent_labels(data: list[list[float]]) -> None:
    model = DBSCAN(eps=2, min_samples=1)
    labels = model.fit_predict(data)
    original = model.labels_.copy()
    labels[:] = [99] * len(labels)
    assert model.labels_ == original


def test_fit_preserves_input() -> None:
    samples = [[0.0], [0.1], [4.0], [4.1]]
    original = [row.copy() for row in samples]
    DBSCAN(eps=0.2, min_samples=2).fit(samples)
    assert samples == original


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="DBSCAN records non-core border points as noise")
def test_dbscan_noise_excludes_border_points() -> None:
    model = DBSCAN(eps=0.21, min_samples=4).fit([[0.0], [0.1], [0.2], [0.3], [0.6]])
    assert model.noises_ == [i for i, label in enumerate(model.labels_) if label == -1]

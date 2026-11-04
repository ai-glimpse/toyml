"""Shared contracts for configuration, fitted state, and caller-owned containers."""

from collections.abc import Callable
from dataclasses import fields, is_dataclass

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from toyml.classification.knn import KNN, Standardizer
from toyml.classification.naive_bayes import CategoricalNaiveBayes, GaussianNaiveBayes, MultinomialNaiveBayes
from toyml.clustering import AGNES, DBSCAN, BisectingKMeans, KMeans
from toyml.ensemble.adaboost import AdaBoost, OneDimensionClassifier
from toyml.ensemble.isolation_forest import IsolationForest, IsolationTree

Model = (
    KNN
    | Standardizer
    | GaussianNaiveBayes
    | MultinomialNaiveBayes
    | CategoricalNaiveBayes
    | AGNES
    | DBSCAN
    | KMeans
    | IsolationForest
    | IsolationTree
)
FACTORIES = [
    pytest.param(lambda: KNN(k=1), id="knn"),
    pytest.param(Standardizer, id="standardizer"),
    pytest.param(GaussianNaiveBayes, id="gaussian-nb"),
    pytest.param(MultinomialNaiveBayes, id="multinomial-nb"),
    pytest.param(CategoricalNaiveBayes, id="categorical-nb"),
    pytest.param(lambda: AGNES(n_cluster=2), id="agnes"),
    pytest.param(lambda: DBSCAN(eps=2, min_samples=1), id="dbscan"),
    pytest.param(lambda: KMeans(k=2, random_seed=0), id="kmeans"),
    pytest.param(lambda: IsolationForest(n_itree=3, max_samples=6, random_seed=0), id="forest"),
    pytest.param(lambda: IsolationTree(max_height=4, random_seed=0), id="tree"),
]
DATA = st.lists(st.integers(1, 30), min_size=4, max_size=8, unique=True).map(
    lambda xs: [[float(x), float(x % 5)] for x in xs]
)
PROPERTY = settings(max_examples=30, deadline=None, derandomize=True, database=None)


def fit(model: Model, data: list[list[float]]) -> None:
    if isinstance(model, (KNN, GaussianNaiveBayes, MultinomialNaiveBayes, CategoricalNaiveBayes)):
        model.fit(data, [i % 2 for i in range(len(data))])
    else:
        model.fit(data)


def state(value: object) -> object:
    """Compare fitted trees without following their parent links back into cycles."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: state(getattr(value, f.name)) for f in fields(value) if f.name != "parent"}
    if isinstance(value, list):
        return [state(item) for item in value]
    if isinstance(value, dict):
        return {key: state(item) for key, item in value.items()}
    return value


@pytest.mark.parametrize("factory", FACTORIES)
@PROPERTY
@given(first=DATA, second=DATA)
def test_fit_preserves_configuration_and_replaces_state(
    factory: Callable[[], Model], first: list[list[float]], second: list[list[float]]
) -> None:
    model = factory()
    config = {
        f.name: getattr(model, f.name)
        for f in fields(model)
        if f.init and not f.name.endswith("_") and not f.name.startswith("_")
    }
    fit(model, first)
    fit(model, second)
    assert {name: getattr(model, name) for name in config} == config
    fresh = factory()
    fit(fresh, second)
    assert state(model) == state(fresh)


@pytest.mark.parametrize(
    "factory",
    FACTORIES
    + [
        pytest.param(lambda: BisectingKMeans(k=2), id="bisecting"),
        pytest.param(lambda: AdaBoost(OneDimensionClassifier), id="adaboost"),
        pytest.param(OneDimensionClassifier, id="weak-learner"),
    ],
)
def test_constructor_excludes_training_state(
    factory: Callable[[], Model | BisectingKMeans | AdaBoost | OneDimensionClassifier],
) -> None:
    model = factory()
    assert all(not f.init for f in fields(model) if f.name.endswith("_") or f.name.startswith("_"))


@pytest.mark.parametrize("standardize", [True, False])
@PROPERTY
@given(data=DATA, query=st.tuples(st.integers(1, 30), st.integers(0, 4)))
def test_knn_owns_training_data_and_preserves_queries(
    standardize: bool, data: list[list[float]], query: tuple[int, int]
) -> None:
    original = [row.copy() for row in data]
    labels = [i % 2 for i in range(len(data))]
    model = KNN(k=1, std_transform=standardize).fit(data, labels)
    point = [float(x) for x in query]
    prediction = model.predict(point)
    assert data == original
    assert point == list(query)
    data[0][0] += 1000
    data.append([99.0, 99.0])
    labels[:] = [99] * len(labels)
    assert model.predict(point) == prediction
    assert model.dataset_ is not data
    assert model.labels_ is not labels


@PROPERTY
@given(data=DATA)
def test_standardizer_returns_independent_values(data: list[list[float]]) -> None:
    original = [row.copy() for row in data]
    model = Standardizer().fit(data)
    expected_state = state(model)
    first = model.transform(data)
    assert data == original
    first[0][0] += 100
    assert model.transform(data) != first
    assert state(model) == expected_state
    assert data == original


@pytest.mark.parametrize(
    "factory",
    [
        lambda: KNN(k=1),
        Standardizer,
        GaussianNaiveBayes,
        MultinomialNaiveBayes,
        CategoricalNaiveBayes,
        lambda: KMeans(k=2),
        lambda: BisectingKMeans(k=2),
        lambda: AdaBoost(OneDimensionClassifier),
        OneDimensionClassifier,
        IsolationForest,
        lambda: IsolationTree(max_height=2),
    ],
)
@PROPERTY
@given(value=st.integers(-10, 10))
def test_unfitted_inference_fails_explicitly(
    factory: Callable[[], Model | BisectingKMeans | AdaBoost | OneDimensionClassifier], value: int
) -> None:
    model = factory()
    if isinstance(model, Standardizer):
        with pytest.raises(ValueError, match="not.*fitted"):
            model.transform([[float(value)]])
    elif isinstance(model, BisectingKMeans):
        with pytest.raises(ValueError, match="not.*fitted"):
            model.predict([[float(value)]])
    elif isinstance(model, IsolationTree):
        with pytest.raises(ValueError, match="not.*fitted"):
            model.get_sample_path_length([float(value)])
    else:
        assert not isinstance(model, (AGNES, DBSCAN))
        with pytest.raises(ValueError, match="not.*fitted"):
            model.predict([float(value)])


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AGNES(n_cluster=2),
        lambda: DBSCAN(eps=2, min_samples=1),
        lambda: KMeans(k=2, random_seed=0),
        lambda: BisectingKMeans(k=2),
    ],
)
@PROPERTY
@given(data=DATA)
def test_fit_predict_returns_independent_labels(
    factory: Callable[[], AGNES | DBSCAN | KMeans | BisectingKMeans], data: list[list[float]]
) -> None:
    model = factory()
    labels = model.fit_predict(data)
    original = model.labels_.copy()
    labels[:] = [99] * len(labels)
    assert model.labels_ == original


@pytest.mark.parametrize("factory", FACTORIES)
@PROPERTY
@given(data=DATA)
def test_fit_leaves_input_unchanged_and_retains_no_mutable_rows(
    factory: Callable[[], Model], data: list[list[float]]
) -> None:
    original = [row.copy() for row in data]
    model = factory()
    fit(model, data)
    assert data == original
    trained = state(model)
    data[0][0] += 100
    data.clear()
    assert state(model) == trained

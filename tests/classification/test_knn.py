import math
from dataclasses import asdict, fields

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from sklearn.neighbors import KNeighborsClassifier

from toyml.classification.knn import KNN, Standardizer, euclidean_distance

TRAINING_DATA = st.lists(st.integers(1, 30), min_size=4, max_size=8, unique=True).map(
    lambda xs: [[float(x), float(x % 5)] for x in xs]
)
PROPERTY = settings(max_examples=30, deadline=None, derandomize=True, database=None)


class TestKNN:
    @pytest.mark.parametrize(("k", "expected_label"), [(1, 0), (3, 1), (5, 1)])
    def test_votes_over_nearest_k_neighbors(self, k: int, expected_label: int) -> None:
        dataset = [[4.0], [2.0], [0.0], [3.0], [1.0]]
        labels = [0, 1, 0, 1, 1]
        query = [0.1]

        prediction = KNN(k=k, std_transform=False).fit(dataset, labels).predict(query)
        reference = KNeighborsClassifier(n_neighbors=k).fit(dataset, labels).predict([query])[0]

        assert prediction == expected_label
        assert prediction == reference

    @given(
        dataset=st.lists(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False, max_value=100, min_value=-100),
                min_size=4,
                max_size=4,
            ),
            min_size=2,
            max_size=10,
        ),
        labels=st.lists(st.integers(min_value=0, max_value=9), min_size=2, max_size=10),
        k=st.integers(min_value=1, max_value=10),
        std_transform=st.booleans(),
    )
    def test_knn_fit_predict(self, dataset: list[list[float]], labels: list[int], k: int, std_transform: bool) -> None:
        # Ensure dataset and labels have the same length
        dataset = dataset[: len(labels)]
        labels = labels[: len(dataset)]

        query = dataset[0].copy()
        knn = KNN(k=min(k, len(dataset)), std_transform=std_transform)
        knn.fit(dataset, labels)

        prediction = knn.predict(query)
        assert prediction in labels

    @pytest.mark.parametrize(
        ("dataset", "labels", "k", "test_point", "expected_label"),
        [
            ([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1], 3, [2.5], 0),
            ([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], ["A", "A", "B", "B"], 2, [3.5, 3.5], "B"),
        ],
    )
    def test_knn_specific_cases(
        self,
        dataset: list[list[float]],
        labels: list[int],
        k: int,
        test_point: list[float],
        expected_label: int,
    ) -> None:
        knn = KNN(k=k, std_transform=False)
        knn.fit(dataset, labels)
        prediction = knn.predict(test_point)
        assert prediction == expected_label

    @PROPERTY
    @given(point=st.lists(st.integers(-10, 10).map(float), min_size=1, max_size=2))
    @example(point=[1.0, 2.0])
    def test_knn_not_fitted(self, point: list[float]) -> None:
        knn = KNN(k=1)
        with pytest.raises(ValueError, match="The model is not fitted yet!"):
            knn.predict(point)


class TestStandardizer:
    @pytest.mark.parametrize(
        ("dataset", "expected_means", "expected_stds"),
        [
            ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [3.0, 4.0], [2.0, 2.0]),
            ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [1.0, 1.0], [1.0, 1.0]),
            ([[1.0], [2.0], [3.0]], [2.0], [1.0]),
        ],
    )
    def test_fit(self, dataset: list[list[float]], expected_means: list[float], expected_stds: list[float]) -> None:
        standardizer = Standardizer()
        standardizer.fit(dataset)

        assert standardizer._dimension == len(dataset[0])
        assert pytest.approx(standardizer._means) == expected_means
        assert pytest.approx(standardizer._stds) == expected_stds

    @pytest.mark.parametrize(
        ("dataset", "input_data", "expected_output"),
        [
            ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[2.0, 3.0]], [[-0.5, -0.5]]),
            ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [[1.5, 1.5]], [[0.5, 0.5]]),
            ([[1.0], [2.0], [3.0]], [[2.5]], [[0.5]]),
        ],
    )
    def test_transform(
        self,
        dataset: list[list[float]],
        input_data: list[list[float]],
        expected_output: list[list[float]],
    ) -> None:
        standardizer = Standardizer()
        standardizer.fit(dataset)

        transformed = standardizer.transform(input_data)
        np.testing.assert_array_almost_equal(transformed, expected_output)

    @pytest.mark.parametrize(
        ("dataset", "expected_output"),
        [
            ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
            ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
            ([[1.0], [2.0], [3.0]], [[-1.0], [0.0], [1.0]]),
        ],
    )
    def test_fit_transform(self, dataset: list[list[float]], expected_output: list[list[float]]) -> None:
        standardizer = Standardizer()
        transformed = standardizer.fit_transform(dataset)
        np.testing.assert_array_almost_equal(transformed, expected_output)

    @PROPERTY
    @given(point=st.lists(st.integers(-10, 10).map(float), min_size=1, max_size=2))
    @example(point=[1.0, 2.0])
    def test_not_fitted(self, point: list[float]) -> None:
        standardizer = Standardizer()
        with pytest.raises(ValueError, match="The model is not fitted yet!"):
            standardizer.transform([point])

    def test_inconsistent_dimensions(self) -> None:
        standardizer = Standardizer()
        with pytest.raises(ValueError, match="All rows must have the same number of columns"):
            standardizer.fit([[1.0, 2.0], [3.0, 4.0, 5.0]])


@given(offset=st.integers(-100, 100))
def test_refit_without_standardization_drops_previous_standardizer(offset: int) -> None:
    samples = [[float(offset + i)] for i in range(4)]
    model = KNN(k=1).fit(samples, [0, 0, 1, 1])
    model.std_transform = False
    model.fit(samples, [0, 0, 1, 1])
    assert model.standardizer_ is None
    assert model.predict(samples[0]) == 0


@given(exponent=st.integers(-250, -10))
def test_one_neighbor_recognizes_exact_training_point(exponent: int) -> None:
    point = [10.0**exponent]
    model = KNN(k=1, std_transform=False).fit([[0.0], point], [0, 1])
    assert model.predict(point) == 1


@given(exponent=st.integers(-250, 250))
def test_standardization_removes_positive_feature_rescaling(exponent: int) -> None:
    samples = [[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]]
    scaled = [[x * 10.0**exponent, y] for x, y in samples]
    expected = Standardizer().fit_transform(samples)
    actual = Standardizer().fit_transform(scaled)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert KNN(k=1).fit(scaled, [0, 1, 2]).predict([0.0, 1.0]) == 0


@given(value=st.integers(-100, 100), dimensions=st.integers(1, 5))
def test_single_sample_and_constant_features(value: int, dimensions: int) -> None:
    sample = [float(value)] * dimensions
    assert Standardizer().fit_transform([sample]) == [[0.0] * dimensions]
    assert Standardizer().fit_transform([sample, sample]) == [[0.0] * dimensions] * 2
    assert KNN(k=1).fit([sample], [7]).predict(sample) == 7


@pytest.mark.parametrize("standardize", [True, False])
@given(k=st.integers(max_value=0))
def test_rejects_nonpositive_neighbor_count(standardize: bool, k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        KNN(k=k, std_transform=standardize).fit([[0.0], [1.0]], [0, 1])


@pytest.mark.parametrize("k", [True, False, 1.5, "1", None])
def test_rejects_noninteger_neighbor_count(k: float | str | None) -> None:
    with pytest.raises(ValueError, match="k"):
        KNN(k=k).fit([[0.0]], [0])  # ty: ignore[invalid-argument-type]


@given(count=st.integers(1, 10), extra=st.integers(1, 10))
def test_rejects_more_neighbors_than_training_samples(count: int, extra: int) -> None:
    with pytest.raises(ValueError, match="k"):
        KNN(k=count + extra).fit([[float(i)] for i in range(count)], list(range(count)))


@pytest.mark.parametrize("standardize", [True, False])
@given(count=st.integers(1, 8), extra=st.integers(1, 8))
def test_rejects_mismatched_label_counts(standardize: bool, count: int, extra: int) -> None:
    samples = [[float(i)] for i in range(count)]
    with pytest.raises(ValueError, match="label"):
        KNN(k=1, std_transform=standardize).fit(samples, list(range(count + extra)))
    with pytest.raises(ValueError, match="label"):
        KNN(k=1, std_transform=standardize).fit(samples, list(range(count - 1)))


@pytest.mark.parametrize("samples", [[], [[]], [[0.0], [1.0, 2.0]], [[math.nan]], [[math.inf]], [[-math.inf]]])
@pytest.mark.parametrize("standardize", [True, False])
def test_rejects_invalid_training_features(samples: list[list[float]], standardize: bool) -> None:
    with pytest.raises(ValueError, match="sample|feature|dimension|columns|numeric"):
        KNN(k=1, std_transform=standardize).fit(samples, [0] * len(samples))
    with pytest.raises(ValueError, match="sample|feature|dimension|columns|numeric"):
        Standardizer().fit(samples)


@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("query", [[], [0.0, 1.0], [math.nan], [math.inf], [-math.inf]])
def test_rejects_invalid_queries(standardize: bool, query: list[float]) -> None:
    model = KNN(k=1, std_transform=standardize).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="sample|feature|dimension|columns|numeric"):
        model.predict(query)


@pytest.mark.parametrize("method", ["transform", "standardization"])
@given(dimension=st.integers(1, 5), extra=st.integers(1, 5))
def test_standardizer_rejects_wrong_feature_count(method: str, dimension: int, extra: int) -> None:
    model = Standardizer().fit([[0.0] * dimension, [1.0] * dimension])
    transform = getattr(model, method)
    with pytest.raises(ValueError, match="dimension"):
        transform([[0.0] * (dimension + extra)])


@pytest.mark.parametrize("standardize", [True, False])
def test_invalid_refit_preserves_previous_model(standardize: bool) -> None:
    model = KNN(k=1, std_transform=standardize).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="at least one sample"):
        model.fit([], [])
    assert model.predict([1.0]) == 1


@pytest.mark.parametrize("value", ["invalid", None, 10**400])
def test_rejects_nonnumeric_or_unrepresentable_features(value: object) -> None:
    samples = [[value]]
    with pytest.raises(ValueError, match="finite numeric"):
        KNN(k=1).fit(samples, [0])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="finite numeric"):
        Standardizer().fit(samples)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("flag", [None, "yes", 1])
def test_rejects_nonboolean_standardization_flag(flag: object) -> None:
    with pytest.raises(TypeError, match="std_transform"):
        KNN(k=1, std_transform=flag).fit([[0.0]], [0])  # ty: ignore[invalid-argument-type]


def test_rejects_unhashable_labels() -> None:
    with pytest.raises(ValueError, match="hashable"):
        KNN(k=1).fit([[0.0]], [[]])


@given(k=st.one_of(st.integers(max_value=0), st.integers(min_value=3)))
def test_prediction_revalidates_changed_neighbor_count(k: int) -> None:
    model = KNN(k=1).fit([[0.0], [1.0]], [0, 1])
    model.k = k
    with pytest.raises(ValueError, match="k"):
        model.predict([0.0])


@pytest.mark.parametrize("samples", [[], [[]], [[math.nan]], [[math.inf]], [[-math.inf]]])
@pytest.mark.parametrize("method", ["transform", "standardization"])
def test_standardizer_rejects_invalid_transform_inputs(samples: list[list[float]], method: str) -> None:
    model = Standardizer().fit([[0.0], [1.0]])
    with pytest.raises(ValueError, match="sample|feature|numeric"):
        getattr(model, method)(samples)


@given(value=st.integers(-100, 100))
def test_invalid_standardizer_refit_preserves_statistics(value: int) -> None:
    samples = [[float(value)], [float(value + 1)]]
    model = Standardizer().fit(samples)
    expected = model.transform(samples)
    with pytest.raises(ValueError, match="at least one sample"):
        model.fit([])
    assert model.transform(samples) == expected


def test_rejects_unrepresentable_standardization_statistics() -> None:
    with pytest.raises(ValueError, match="floating-point range"):
        Standardizer().fit([[-1.7e308], [1.7e308]])


def test_rejects_unrepresentable_standardized_query() -> None:
    model = KNN(k=1).fit([[0.0], [1e-250]], [0, 1])
    with pytest.raises(ValueError, match="floating-point range"):
        model.predict([1e308])


def test_standardization_overflow_preserves_previous_model() -> None:
    model = KNN(k=1).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="floating-point range"):
        model.fit([[-1.7e308], [1.7e308]], [2, 3])
    assert model.predict([1.0]) == 1


VECTORS = st.lists(st.integers(-100, 100), min_size=1, max_size=8)


@given(values=VECTORS, exponent=st.integers(-250, 250))
def test_distance_preserves_small_and_large_scales(values: list[int], exponent: int) -> None:
    point = [value * 10.0**exponent for value in values]
    origin = [0.0] * len(point)
    actual = euclidean_distance(origin, point)
    expected = math.dist(origin, point)
    assert actual == pytest.approx(expected, rel=1e-14, abs=0.0)
    assert euclidean_distance(point, origin) == actual
    assert euclidean_distance(point, point) == 0.0


def test_distance_handles_overflowing_coordinate_difference() -> None:
    assert euclidean_distance([-1e308], [1e308]) == math.inf


@given(pairs=st.lists(st.tuples(st.floats(-1e150, 1e150), st.floats(-1e150, 1e150)), min_size=1, max_size=8))
def test_distance_matches_reference_for_arbitrary_points(pairs: list[tuple[float, float]]) -> None:
    first = [x for x, _ in pairs]
    second = [y for _, y in pairs]
    assert euclidean_distance(first, second) == pytest.approx(math.dist(first, second), rel=1e-14, abs=0.0)


@PROPERTY
@given(first=TRAINING_DATA, second=TRAINING_DATA)
def test_knn_refit_preserves_configuration_and_replaces_state(
    first: list[list[float]], second: list[list[float]]
) -> None:
    model = KNN(k=1)
    fresh = KNN(k=1)
    model.fit(first, [i % 2 for i in range(len(first))])
    model.fit(second, [i % 2 for i in range(len(second))])
    for field in fields(model):
        if field.init:
            assert getattr(model, field.name) == getattr(fresh, field.name)
    fresh.fit(second, [i % 2 for i in range(len(second))])
    assert asdict(model) == asdict(fresh)


@PROPERTY
@given(data=TRAINING_DATA)
def test_knn_fit_owns_its_data(data: list[list[float]]) -> None:
    original = [row.copy() for row in data]
    model = KNN(k=1)
    model.fit(data, [i % 2 for i in range(len(data))])
    assert data == original
    trained = asdict(model)
    data[0][0] += 100
    data.clear()
    assert asdict(model) == trained


@PROPERTY
@given(first=TRAINING_DATA, second=TRAINING_DATA)
def test_standardizer_refit_preserves_configuration_and_replaces_state(
    first: list[list[float]], second: list[list[float]]
) -> None:
    model = Standardizer()
    fresh = Standardizer()
    model.fit(first)
    model.fit(second)
    for field in fields(model):
        if field.init:
            assert getattr(model, field.name) == getattr(fresh, field.name)
    fresh.fit(second)
    assert asdict(model) == asdict(fresh)


@PROPERTY
@given(data=TRAINING_DATA)
def test_standardizer_fit_owns_its_data(data: list[list[float]]) -> None:
    original = [row.copy() for row in data]
    model = Standardizer()
    model.fit(data)
    assert data == original
    trained = asdict(model)
    data[0][0] += 100
    data.clear()
    assert asdict(model) == trained


@pytest.mark.parametrize("model", [KNN(k=1), Standardizer()])
def test_constructor_excludes_training_state(model: KNN | Standardizer) -> None:
    assert all(not f.init for f in fields(model) if f.name.endswith("_") or f.name.startswith("_"))


@pytest.mark.parametrize("standardize", [True, False])
@PROPERTY
@given(data=TRAINING_DATA, query=st.tuples(st.integers(1, 30), st.integers(0, 4)))
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
@given(data=TRAINING_DATA)
def test_standardizer_returns_independent_values(data: list[list[float]]) -> None:
    original = [row.copy() for row in data]
    model = Standardizer().fit(data)
    expected_state = asdict(model)
    first = model.transform(data)
    assert data == original
    first[0][0] += 100
    assert model.transform(data) != first
    assert asdict(model) == expected_state
    assert data == original

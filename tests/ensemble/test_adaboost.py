import pytest
from hypothesis import given
from hypothesis import strategies as st

from toyml.ensemble.adaboost import AdaBoost, OneDimensionClassifier


class TestOneDimensionClassifier:
    @pytest.fixture
    def sample_dataset(self) -> tuple[list[list[float]], list[float], list[int]]:
        dataset: list[list[float]] = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
        weights: list[float] = [1 / 6] * 6
        labels: list[int] = [1, 1, 1, -1, -1, -1]
        return dataset, weights, labels

    def test_fit(self, sample_dataset: tuple[list[list[float]], list[float], list[int]]) -> None:
        dataset, weights, labels = sample_dataset
        classifier = OneDimensionClassifier()
        classifier.fit(dataset, weights, labels)

        assert classifier.error_rate_ < 1.0
        assert classifier._best_cut is not None
        assert classifier._sign_mode in [
            OneDimensionClassifier.SignMode.POS_NEG,
            OneDimensionClassifier.SignMode.NEG_POS,
        ]

    def test_predict(self, sample_dataset: tuple[list[list[float]], list[float], list[int]]) -> None:
        dataset, weights, labels = sample_dataset
        classifier = OneDimensionClassifier().fit(dataset, weights, labels)

        assert classifier.predict([0]) == 1
        assert classifier.predict([5]) == -1

    def test_get_error_rate(self, sample_dataset: tuple[list[list[float]], list[float], list[int]]) -> None:
        dataset, weights, labels = sample_dataset
        classifier = OneDimensionClassifier().fit(dataset, weights, labels)

        assert 0 <= classifier.get_error_rate() <= 1

    def test_get_predict_labels(self, sample_dataset: tuple[list[list[float]], list[float], list[int]]) -> None:
        dataset, weights, labels = sample_dataset
        classifier = OneDimensionClassifier().fit(dataset, weights, labels)

        predict_labels = classifier.get_predict_labels()
        assert len(predict_labels) == len(dataset)
        assert all(label in [-1, 1] for label in predict_labels)


class TestAdaBoost:
    @pytest.fixture
    def sample_dataset(self) -> tuple[list[list[float]], list[int]]:
        dataset: list[list[float]] = [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]]
        labels: list[int] = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
        return dataset, labels

    def test_fit(self, sample_dataset: tuple[list[list[float]], list[int]]) -> None:
        dataset, labels = sample_dataset
        adaboost = AdaBoost(weak_learner=OneDimensionClassifier, n_weak_learner=3)
        adaboost.fit(dataset, labels)

        assert len(adaboost._weak_learner_predicts) == 3
        assert len(adaboost._alphas) == 3
        assert adaboost.predict_labels_ is not None
        assert adaboost.training_error_rate_ is not None

    def test_predict(self, sample_dataset: tuple[list[list[float]], list[int]]) -> None:
        dataset, labels = sample_dataset
        adaboost = AdaBoost(weak_learner=OneDimensionClassifier, n_weak_learner=3).fit(dataset, labels)

        assert adaboost.predict([1.5]) in [-1, 1]
        assert adaboost.predict([8.5]) in [-1, 1]

    def test_error_rate(self, sample_dataset: tuple[list[list[float]], list[int]]) -> None:
        dataset, labels = sample_dataset
        adaboost = AdaBoost(weak_learner=OneDimensionClassifier, n_weak_learner=3).fit(dataset, labels)
        assert adaboost.training_error_rate_ is not None
        assert 0 <= adaboost.training_error_rate_ <= 1

    def test_predict_labels(self, sample_dataset: tuple[list[list[float]], list[int]]) -> None:
        dataset, labels = sample_dataset
        adaboost = AdaBoost(weak_learner=OneDimensionClassifier, n_weak_learner=3).fit(dataset, labels)

        assert adaboost.predict_labels_ is not None
        assert len(adaboost.predict_labels_) == len(dataset)
        assert all(label in [-1, 1] for label in adaboost.predict_labels_)


@given(offset=st.integers(-10, 10))
def test_training_inputs_and_returned_labels_are_independent(offset: int) -> None:
    samples = [[float(offset + i)] for i in range(10)]
    labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    model = AdaBoost(OneDimensionClassifier, n_weak_learner=3).fit(samples, labels)
    predictions = [model.predict(row) for row in samples]
    assert samples == [[float(offset + i)] for i in range(10)]
    assert labels == [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    samples[0][0] += 100
    labels.clear()
    assert [model.predict([float(offset + i)]) for i in range(10)] == predictions
    assert not hasattr(model, "_labels")
    weak = OneDimensionClassifier().fit([[0.0], [1.0]], [0.5, 0.5], [-1, 1])
    result = weak.get_predict_labels()
    result.clear()
    assert weak.get_predict_labels() == [-1, 1]


@given(first_offset=st.integers(-10, 10), second_offset=st.integers(-10, 10))
def test_refit_replaces_weak_learners(first_offset: int, second_offset: int) -> None:
    labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    first = [[float(first_offset + i)] for i in range(10)]
    second = [[float(second_offset + i)] for i in range(10)]
    model = AdaBoost(OneDimensionClassifier, n_weak_learner=3).fit(first, labels).fit(second, labels)
    fresh = AdaBoost(OneDimensionClassifier, n_weak_learner=3).fit(second, labels)
    assert model._alphas == fresh._alphas
    assert model._weights == fresh._weights
    assert model.predict_labels_ == fresh.predict_labels_
    assert [model.predict(row) for row in second] == [fresh.predict(row) for row in second]


def test_unfitted_weak_learner_accessors_fail_explicitly() -> None:
    model = OneDimensionClassifier()
    with pytest.raises(ValueError, match="not fitted"):
        model.get_error_rate()
    with pytest.raises(ValueError, match="not fitted"):
        model.get_predict_labels()

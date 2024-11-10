import math

import pytest

from sklearn.naive_bayes import GaussianNB

from toyml.classification.naive_bayes import GaussianNaiveBayes


@pytest.fixture
def wikipedia_person_classification_dataset_label() -> tuple[list[list[float]], list[int]]:
    """
    References: https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Examples
    """
    # features: height(feet), weight(lbs), foot size(inches)
    dataset = [
        [6.00, 180, 12],
        [5.92, 190, 11],
        [5.58, 170, 12],
        [5.92, 165, 10],
        [5.00, 100, 6],
        [5.50, 150, 8],
        [5.42, 130, 7],
        [5.75, 150, 9],
    ]
    # 0: male; 1: female
    label = [0, 0, 0, 0, 1, 1, 1, 1]
    return dataset, label


@pytest.fixture
def wikipedia_person_classification_sample() -> list[float]:
    """
    References: https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Examples
    """
    return [6, 130, 8]


class TestGaussianNaiveBayesIntegration:
    def test_same_result_with_wikipedia(
        self,
        wikipedia_person_classification_dataset_label: tuple[list[list[float]], list[int]],
        wikipedia_person_classification_sample: list[float],
    ) -> None:
        dataset, label = wikipedia_person_classification_dataset_label
        sut = GaussianNaiveBayes(unbiased_variance=True, var_smoothing=0).fit(dataset, label)

        sut_label = sut.predict(wikipedia_person_classification_sample)
        sut_prob = sut.predict_proba(wikipedia_person_classification_sample, normalization=False)

        assert sut_label == 1
        assert math.isclose(sut_prob[0], 6.1984 * 1e-9, abs_tol=1e-6)
        assert math.isclose(sut_prob[1], 5.3778 * 1e-4, abs_tol=1e-6)

    def test_same_result_with_sklearn(
        self,
        wikipedia_person_classification_dataset_label: tuple[list[list[float]], list[int]],
        wikipedia_person_classification_sample: list[float],
    ) -> None:
        dataset, label = wikipedia_person_classification_dataset_label
        sklearn_clf = GaussianNB()
        sklearn_clf.fit(dataset, label)
        # use the same variance calculation config with sklearn
        sut = GaussianNaiveBayes(unbiased_variance=False, var_smoothing=1e-9).fit(dataset, label)

        # test same labels
        sut_label = sut.predict(wikipedia_person_classification_sample)
        sklearn_label = sklearn_clf.predict([wikipedia_person_classification_sample])
        assert sut_label == sklearn_label[0]

        # test same log probs
        sut_log_prob = sut.predict_log_proba(wikipedia_person_classification_sample)
        sklearn_log_prob = sklearn_clf.predict_log_proba([wikipedia_person_classification_sample])
        assert math.isclose(sut_log_prob[0], sklearn_log_prob[0][0])
        assert math.isclose(sut_log_prob[1], sklearn_log_prob[0][1])

        # test same probs
        sut_prob = sut.predict_proba(wikipedia_person_classification_sample)
        sklearn_prob = sklearn_clf.predict_proba([wikipedia_person_classification_sample])
        assert math.isclose(sut_prob[0], sklearn_prob[0][0])
        assert math.isclose(sut_prob[1], sklearn_prob[0][1])
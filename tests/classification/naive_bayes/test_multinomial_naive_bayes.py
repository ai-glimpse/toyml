import math

import numpy as np
import pytest

from sklearn.naive_bayes import MultinomialNB

from toyml.classification.naive_bayes import MultinomialNaiveBayes


@pytest.fixture
def sklearn_example_random_dataset_label() -> tuple[list[list[int]], list[int]]:
    """
    References: https://scikit-learn.org/1.5/modules/generated/sklearn.naive_bayes.MultinomialNB.html#multinomialnb
    """
    rng = np.random.RandomState(1)
    dataset = rng.randint(5, size=(6, 100)).tolist()
    label = np.array([1, 2, 3, 4, 5, 6]).tolist()
    return dataset, label


class TestMultinomialNaiveBayesIntegration:
    def test_same_result_with_sklearn(
        self,
        sklearn_example_random_dataset_label: tuple[list[list[int]], list[int]],
    ) -> None:
        dataset, label = sklearn_example_random_dataset_label
        sklearn_clf = MultinomialNB()
        sklearn_clf.fit(dataset, label)
        # use the same variance calculation config with sklearn
        sut = MultinomialNaiveBayes(alpha=1).fit(dataset, label)
        # test same labels
        test_sample = dataset[2]
        sklearn_label = sklearn_clf.predict([test_sample])
        sut_label = sut.predict(test_sample)

        assert sut_label == sklearn_label[0]

        # test same log probs
        sut_log_prob = sut.predict_log_proba(test_sample)
        sklearn_log_prob = sklearn_clf.predict_log_proba([test_sample])
        for i in range(6):
            assert math.isclose(sut_log_prob[i + 1], sklearn_log_prob[0][i])

        # # test same probs
        sut_prob = sut.predict_proba(test_sample)
        sklearn_prob = sklearn_clf.predict_proba([test_sample])
        for i in range(6):
            assert math.isclose(sut_prob[i + 1], sklearn_prob[0][i])

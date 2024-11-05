from __future__ import annotations

import math
import statistics

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class GaussianNaiveBayes:
    """
    Gaussian naive bayes classification algorithm implementation.

    Examples:
        >>> label = [0, 0, 0, 0, 1, 1, 1, 1]
        >>> dataset = [[6.00, 180, 12], [5.92, 190, 11], [5.58, 170, 12], [5.92, 165, 10], [5.00, 100, 6], [5.50, 150, 8], [5.42, 130, 7], [5.75, 150, 9]]
        >>> clf = GaussianNaiveBayes().fit(dataset, label)
        >>> clf.predict([6.00, 130, 8])
        1

    """

    labels_: list[int] = field(default_factory=list)
    """The labels in training dataset"""
    class_count_: int = 0
    """The number of classes in training dataset"""
    class_prior_: dict[int, float] = field(default_factory=dict)
    """The prior probability of each class in training dataset"""
    means_: dict[int, list[float]] = field(default_factory=dict)
    """The means of each class in training dataset"""
    variances_: dict[int, list[float]] = field(default_factory=dict)
    """The variance of each class in training dataset"""

    def fit(self, dataset: list[list[float]], labels: list[int]) -> GaussianNaiveBayes:
        """Fit the naive bayes model"""
        self.labels_ = sorted(set(labels))
        self.class_count_ = len(set(labels))
        self.class_prior_ = {label: 1 / self.class_count_ for label in self.labels_}
        self.means_, self.variances_ = self._get_classes_means_variances(dataset, labels)
        return self

    def predict(self, sample: list[float]) -> int:
        label_likelihoods = self._likelihood(sample)
        raw_label_posteriors: dict[int, float] = {}
        for label, likelihood in label_likelihoods.items():
            raw_label_posteriors[label] = likelihood * self.class_prior_[label]
        evidence = sum(raw_label_posteriors.values())
        label_posteriors = {label: raw_posterior / evidence for label, raw_posterior in raw_label_posteriors.items()}
        label = max(label_posteriors, key=lambda k: label_posteriors[k])
        return label

    def _likelihood(self, sample: list[float]) -> dict[int, float]:
        """
        Calculate the likelihood of each sample in each class
        """
        label_likelihoods: dict[int, float] = {}
        for label in self.labels_:
            label_means = self.means_[label]
            label_vars = self.variances_[label]
            likelihood = 1.0
            for i, xi in enumerate(sample):
                # TODO: try to calculate the log-likelihood
                likelihood *= (1 / math.sqrt(2 * math.pi * label_vars[i])) * math.exp(
                    -((xi - label_means[i]) ** 2) / (2 * label_vars[i])
                )
            label_likelihoods[label] = likelihood
        return label_likelihoods

    def _get_classes_means_variances(
        self,
        dataset: list[list[float]],
        labels: list[int],
    ) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
        means, variances = {}, {}
        for label in self.labels_:
            label_samples = [sample for (sample, sample_label) in zip(dataset, labels) if sample_label == label]
            means[label] = self._dataset_column_means(label_samples)
            variances[label] = self._dataset_column_variances(label_samples)
        return means, variances

    @staticmethod
    def _dataset_column_means(dataset: list[list[float]]) -> list[float]:
        """
        Calculate vectors mean
        """
        return [statistics.mean(column) for column in zip(*dataset, strict=True)]

    @staticmethod
    def _dataset_column_variances(dataset: list[list[float]]) -> list[float]:
        """
        Calculate vectors(every column) standard variance
        """
        return [statistics.variance(column) for column in zip(*dataset, strict=True)]

    def _get_classes_vars(self, dataset: list[list[float]], labels: list[int]) -> dict[int, list[float]]:
        dimension_num = len(dataset[0])

        label_count = {label: 0 for label in self.labels_}
        label_dimension_sum_of_squares = {label: [0.0] * dimension_num for label in self.labels_}
        for label, sample in zip(labels, dataset):
            label_count[label] += 1
            for dim in range(dimension_num):
                label_dimension_sum_of_squares[label][dim] += (sample[dim] - self.means_[label][dim]) ** 2
        # TODO: simple sample variance case handle
        variances = {
            label: [
                dimension_sum_of_square / (label_count[label] - 1)
                for dimension_sum_of_square in dimension_sum_of_squares
            ]
            for label, dimension_sum_of_squares in label_dimension_sum_of_squares.items()
        }
        return variances


@dataclass
class MultinomialNaiveBayes:
    alpha: float = 1.0
    """Additive (Laplace/Lidstone) smoothing parameter"""
    labels_: list[int] = field(default_factory=list)
    """The labels in training dataset"""
    class_count_: int = 0
    """The number of classes in training dataset"""
    class_prior_: dict[int, float] = field(default_factory=dict)
    """The prior probability of each class in training dataset"""
    class_feature_count_: dict[int, list[int]] = field(default_factory=dict)
    """The feature value counts of each class in training dataset"""
    class_feature_prob_: dict[int, list[float]] = field(default_factory=dict)
    """The feature value probability of each class in training dataset"""

    def fit(self, dataset: list[list[int]], labels: list[int]) -> MultinomialNaiveBayes:
        self.labels_ = sorted(set(labels))
        self.class_count_ = len(set(labels))
        # get the prior from training dataset labels
        self.class_prior_ = {label: count / len(dataset) for label, count in Counter(labels).items()}
        self.class_feature_count_, self.class_feature_prob_ = self._get_classes_feature_count_prob(dataset, labels)
        return self

    def predict(self, sample: list[int]) -> int:
        label_likelihoods = self._likelihood(sample)
        raw_label_posteriors: dict[int, float] = {}
        for label, likelihood in label_likelihoods.items():
            raw_label_posteriors[label] = likelihood + math.log(self.class_prior_[label])
        raw_label_posteriors_shift = {
            label: likelihood - max(raw_label_posteriors.values()) for label, likelihood in label_likelihoods.items()
        }
        print(raw_label_posteriors_shift)
        evidence = sum(raw_label_posteriors_shift.values())
        label_posteriors = {
            label: raw_posterior / evidence for label, raw_posterior in raw_label_posteriors_shift.items()
        }
        label = max(label_posteriors, key=lambda k: label_posteriors[k])
        return label

    def _likelihood(self, sample: list[int]) -> dict[int, float]:
        """
        Calculate the likelihood of each sample in each class
        """
        label_likelihoods: dict[int, float] = {}
        for label in self.labels_:
            likelihood = 0.0
            for i, xi in enumerate(sample):
                # calculate the log-likelihood
                likelihood += xi * math.log(self.class_feature_prob_[label][i])
            label_likelihoods[label] = likelihood
        return label_likelihoods

    def _get_classes_feature_count_prob(
        self,
        dataset: list[list[int]],
        labels: list[int],
    ) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
        dimension_num = len(dataset[0])
        feature_count, feature_prob = {}, {}
        for label in self.labels_:
            label_samples = [sample for (sample, sample_label) in zip(dataset, labels) if sample_label == label]
            counts = self._dataset_feature_counts(label_samples)
            feature_count[label] = counts
            feature_prob[label] = [
                (value_count + self.alpha) / (sum(counts) + self.alpha * dimension_num) for value_count in counts
            ]

        return feature_count, feature_prob

    @staticmethod
    def _dataset_feature_counts(dataset: list[list[int]]) -> list[int]:
        """
        Calculate feature value counts
        """
        return [sum(column) for column in zip(*dataset, strict=True)]


if __name__ == "__main__":
    import numpy as np

    rng = np.random.RandomState(1)
    X = rng.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(X, y)
    print(clf.predict_log_proba(X[2:3]))

    clf1 = MultinomialNaiveBayes()
    clf1.fit([[int(v) for v in s] for s in X], [int(v) for v in y])
    sample = [int(x) for x in X[2:3][0]]
    print(sample)
    print(clf1.predict(sample))

from __future__ import annotations

import math

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier.

    Examples:
        >>> import random
        >>> rng = random.Random(0)
        >>> dataset = [[rng.randint(0, 5) for _ in range(100)] for _ in range(6)]
        >>> label = [1, 2, 3, 4, 5, 6]
        >>> clf = MultinomialNaiveBayes().fit(dataset, label)
        >>> clf.predict(dataset[2])
        3

    """

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
    class_feature_log_prob_: dict[int, list[float]] = field(default_factory=dict)
    """The feature value probability of each class in training dataset"""

    def fit(self, dataset: list[list[int]], labels: list[int]) -> MultinomialNaiveBayes:
        self.labels_ = sorted(set(labels))
        self.class_count_ = len(set(labels))
        # get the prior from training dataset labels
        self.class_prior_ = {label: count / len(dataset) for label, count in Counter(labels).items()}
        self.class_feature_count_, self.class_feature_log_prob_ = self._get_classes_feature_count_prob(dataset, labels)
        return self

    def predict(self, sample: list[int]) -> int:
        label_posteriors = self.predict_log_proba(sample)
        label = max(label_posteriors, key=lambda k: label_posteriors[k])
        return label

    def predict_proba(self, sample: list[int]) -> dict[int, float]:
        label_posteriors = self.predict_log_proba(sample)
        return {label: math.exp(log_prob) for label, log_prob in label_posteriors.items()}

    def predict_log_proba(self, sample: list[int]) -> dict[int, float]:
        label_likelihoods = self._likelihood(sample)
        raw_label_posteriors: dict[int, float] = {}
        for label, likelihood in label_likelihoods.items():
            raw_label_posteriors[label] = likelihood + math.log(self.class_prior_[label])
        # ref: https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/naive_bayes.py#L97
        max_log_prob = max(raw_label_posteriors.values())
        logsumexp_prob = max_log_prob + math.log(
            sum(math.exp(log_prob - max_log_prob) for log_prob in raw_label_posteriors.values())
        )
        label_posteriors = {
            label: raw_posterior - logsumexp_prob for label, raw_posterior in raw_label_posteriors.items()
        }
        return label_posteriors

    def _likelihood(self, sample: list[int]) -> dict[int, float]:
        """
        Calculate the likelihood of each sample in each class
        """
        label_likelihoods: dict[int, float] = {}
        for label in self.labels_:
            likelihood = 0.0
            for i, xi in enumerate(sample):
                # calculate the log-likelihood
                likelihood += xi * self.class_feature_log_prob_[label][i]
            label_likelihoods[label] = likelihood
        return label_likelihoods

    def _get_classes_feature_count_prob(
        self,
        dataset: list[list[int]],
        labels: list[int],
    ) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
        feature_count, feature_prob = {}, {}
        for label in self.labels_:
            label_samples = [sample for (sample, sample_label) in zip(dataset, labels) if sample_label == label]
            counts = self._dataset_feature_counts(label_samples)
            feature_count[label] = counts
            feature_prob[label] = [math.log(value_count / sum(counts)) for value_count in counts]

        return feature_count, feature_prob

    def _dataset_feature_counts(self, dataset: list[list[int]]) -> list[int]:
        """
        Calculate feature value counts
        """
        return [sum(column) + self.alpha for column in zip(*dataset, strict=True)]

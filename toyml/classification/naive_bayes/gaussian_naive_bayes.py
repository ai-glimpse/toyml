from __future__ import annotations

import math
import statistics

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

    unbiased_variance: bool = True
    """Use the unbiased variance estimation or not. Default is True."""
    var_smoothing: float = 1e-9
    """Portion of the largest variance of all features that is added to variances for calculation stability."""
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
    epsilon_: float = 0
    """The absolute additive value to variances."""

    def fit(self, dataset: list[list[float]], labels: list[int]) -> GaussianNaiveBayes:
        """Fit the naive bayes model"""
        self.labels_ = sorted(set(labels))
        self.class_count_ = len(set(labels))
        self.class_prior_ = {label: 1 / self.class_count_ for label in self.labels_}
        self.epsilon_ = self.var_smoothing * max(self._variance(col) for col in zip(*dataset))
        self.means_, self.variances_ = self._get_classes_means_variances(dataset, labels)
        return self

    def predict(self, sample: list[float]) -> int:
        label_posteriors = self.predict_proba(sample)
        label = max(label_posteriors, key=lambda k: label_posteriors[k])
        return label

    def predict_proba(self, sample: list[float], normalization: bool = True) -> dict[int, float]:
        label_posteriors = self.predict_log_proba(sample, normalization)
        return {label: math.exp(log_prob) for label, log_prob in label_posteriors.items()}

    def predict_log_proba(self, sample: list[float], normalization: bool = True) -> dict[int, float]:
        label_likelihoods = self._log_likelihood(sample)
        raw_label_posteriors: dict[int, float] = {}
        for label, likelihood in label_likelihoods.items():
            raw_label_posteriors[label] = likelihood + math.log(self.class_prior_[label])
        if normalization is False:
            return raw_label_posteriors
        # ref: https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/naive_bayes.py#L97
        max_log_prob = max(raw_label_posteriors.values())
        logsumexp_prob = max_log_prob + math.log(
            sum(math.exp(log_prob - max_log_prob) for log_prob in raw_label_posteriors.values())
        )
        label_posteriors = {
            label: raw_posterior - logsumexp_prob for label, raw_posterior in raw_label_posteriors.items()
        }
        return label_posteriors

    def _log_likelihood(self, sample: list[float]) -> dict[int, float]:
        """
        Calculate the likelihood of each sample in each class
        """
        label_likelihoods: dict[int, float] = {}
        for label in self.labels_:
            label_means = self.means_[label]
            label_vars = self.variances_[label]
            log_likelihood = 0.0
            for i, xi in enumerate(sample):
                # calculate the log-likelihood
                log_likelihood += -0.5 * math.log(2 * math.pi * label_vars[i]) - (
                    (xi - label_means[i]) ** 2 / (2 * label_vars[i])
                )
            label_likelihoods[label] = log_likelihood
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

    def _dataset_column_variances(self, dataset: list[list[float]]) -> list[float]:
        """
        Calculate vectors(every column) standard variance
        """
        return [self._variance(column) + self.epsilon_ for column in zip(*dataset, strict=True)]

    def _variance(self, xs: list[float] | tuple[float, ...]) -> float:
        mean = statistics.mean(xs)
        ss = sum((x - mean) ** 2 for x in xs)
        if self.unbiased_variance is True:
            variance = ss / (len(xs) - 1)
        else:
            variance = ss / len(xs)
        return variance
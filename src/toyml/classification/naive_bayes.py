from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class BaseNaiveBayes(ABC):
    class_prior_: dict[int, float] = field(default_factory=dict, init=False)
    """The prior probability of each class in training dataset"""

    @abstractmethod
    def fit(self, dataset: list[list[float]], labels: list[int]) -> BaseNaiveBayes:
        raise NotImplementedError

    def predict(self, sample: list[float]) -> int:
        """Predict the class label for a given sample.

        Args:
            sample: A single sample to predict, represented as a list of feature values.

        Returns:
            int: Predicted class label.
        """
        class_posteriors = self.predict_proba(sample)
        label = max(class_posteriors, key=lambda label: class_posteriors[label])
        return label

    def predict_proba(self, sample: list[float], normalization: bool = True) -> dict[int, float]:
        """Predict class probabilities for a given sample.

        Args:
            sample: A single sample to predict, represented as a list of feature values.
            normalization: Whether to normalize the probabilities. If False, return
                joint probabilities (or densities for Gaussian features). Default is True.

        Returns:
            dict[int, float]: Posterior probabilities by class label when normalized;
                otherwise, unnormalized joint probabilities or densities.
        """
        class_log_probs = self.predict_log_proba(sample, normalization)
        return {label: math.exp(log_prob) for label, log_prob in class_log_probs.items()}

    def predict_log_proba(self, sample: list[float], normalization: bool = True) -> dict[int, float]:
        """Predict log probabilities for a given sample.

        Args:
            sample: A single sample to predict, represented as a list of feature values.
            normalization: Whether to normalize the log probabilities. If False, return
                joint log probabilities (or log densities for Gaussian features). Default is True.

        Returns:
            dict[int, float]: Log posterior probabilities by class label when normalized;
                otherwise, unnormalized joint log probabilities or log densities.
        """
        if not self.class_prior_:
            msg = "The model is not fitted yet!"
            raise ValueError(msg)
        class_log_likelihoods = self._log_likelihood(sample)
        class_log_joint_probs: dict[int, float] = {}
        for label, log_likelihood in class_log_likelihoods.items():
            class_log_joint_probs[label] = log_likelihood + math.log(self.class_prior_[label])
        if normalization is False:
            return class_log_joint_probs
        # ref: https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/naive_bayes.py#L97
        max_log_joint_prob = max(class_log_joint_probs.values())
        log_evidence = max_log_joint_prob + math.log(
            sum(math.exp(log_joint_prob - max_log_joint_prob) for log_joint_prob in class_log_joint_probs.values()),
        )
        class_log_posteriors = {
            label: log_joint_prob - log_evidence for label, log_joint_prob in class_log_joint_probs.items()
        }
        return class_log_posteriors

    @abstractmethod
    def _log_likelihood(self, sample: list[float]) -> dict[int, float]:
        raise NotImplementedError


@dataclass
class GaussianNaiveBayes(BaseNaiveBayes):
    """Gaussian naive bayes classification algorithm implementation.

    Examples:
        >>> label = [0, 0, 0, 0, 1, 1, 1, 1]
        >>> dataset = [
        ...     [6.00, 180, 12],
        ...     [5.92, 190, 11],
        ...     [5.58, 170, 12],
        ...     [5.92, 165, 10],
        ...     [5.00, 100, 6],
        ...     [5.50, 150, 8],
        ...     [5.42, 130, 7],
        ...     [5.75, 150, 9],
        ... ]
        >>> clf = GaussianNaiveBayes().fit(dataset, label)
        >>> clf.predict([6.00, 130, 8])
        1

    """

    unbiased_variance: bool = True
    """Use the unbiased variance estimation or not. Default is True."""
    var_smoothing: float = 1e-9
    """Portion of the largest variance of all features that is added to variances for calculation stability."""
    labels_: list[int] = field(default_factory=list, init=False)
    """The labels in training dataset"""
    class_count_: int = field(default=0, init=False)
    """The number of classes in training dataset"""
    class_prior_: dict[int, float] = field(default_factory=dict, init=False)
    """Class prior probabilities estimated from training label frequencies."""
    means_: dict[int, list[float]] = field(default_factory=dict, init=False)
    """The means of each class in training dataset"""
    variances_: dict[int, list[float]] = field(default_factory=dict, init=False)
    """The variance of each class in training dataset"""
    epsilon_: float = field(default=0, init=False)
    """The absolute additive value to variances."""

    def fit(self, dataset: list[list[float]], labels: list[int]) -> GaussianNaiveBayes:
        """Fit the Gaussian Naive Bayes classifier.

        Args:
            dataset: Training data, where each row is a sample and each column is a feature.
            labels: Target labels for training data.

        Returns:
            self: Returns the instance itself.
        """
        self.labels_ = sorted(set(labels))
        self.class_count_ = len(set(labels))
        self.class_prior_ = {label: count / len(dataset) for label, count in Counter(labels).items()}
        self.epsilon_ = self.var_smoothing * max(self._variance(col) for col in zip(*dataset, strict=False))
        self.means_, self.variances_ = self._get_classes_means_variances(dataset, labels)
        return self

    def _log_likelihood(self, sample: list[float]) -> dict[int, float]:
        """Calculate the log-likelihood of the sample for each class."""
        class_log_likelihoods: dict[int, float] = {}
        for label in self.labels_:
            label_means = self.means_[label]
            label_variances = self.variances_[label]
            log_likelihood = 0.0
            for feature_index, feature_value in enumerate(sample):
                log_likelihood += -0.5 * math.log(2 * math.pi * label_variances[feature_index]) - (
                    (feature_value - label_means[feature_index]) ** 2 / (2 * label_variances[feature_index])
                )
            class_log_likelihoods[label] = log_likelihood
        return class_log_likelihoods

    def _get_classes_means_variances(
        self,
        dataset: list[list[float]],
        labels: list[int],
    ) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
        means: dict[int, list[float]] = {}
        variances: dict[int, list[float]] = {}
        for label in self.labels_:
            label_samples = [
                sample for (sample, sample_label) in zip(dataset, labels, strict=False) if sample_label == label
            ]
            means[label] = self._dataset_column_means(label_samples)
            variances[label] = self._dataset_column_variances(label_samples)
        return means, variances

    @staticmethod
    def _dataset_column_means(dataset: list[list[float]]) -> list[float]:
        """Calculate vectors mean."""
        return [statistics.mean(column) for column in zip(*dataset, strict=True)]

    def _dataset_column_variances(self, dataset: list[list[float]]) -> list[float]:
        """Calculate the smoothed variance of each feature."""
        return [self._variance(column) + self.epsilon_ for column in zip(*dataset, strict=True)]

    def _variance(self, xs: list[float] | tuple[float, ...]) -> float:
        n = len(xs)
        mean = statistics.mean(xs)
        ss = sum((x - mean) ** 2 for x in xs)
        if self.unbiased_variance is True:
            if n > 1:  # noqa: SIM108
                variance = ss / (len(xs) - 1)
            else:
                variance = 0.0  # Variance is zero when there's only one sample
        else:
            variance = ss / len(xs)
        return variance


@dataclass
class MultinomialNaiveBayes(BaseNaiveBayes):
    """Multinomial Naive Bayes classifier.

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
    labels_: list[int] = field(default_factory=list, init=False)
    """The labels in training dataset"""
    class_count_: int = field(default=0, init=False)
    """The number of classes in training dataset"""
    class_prior_: dict[int, float] = field(default_factory=dict, init=False)
    """The prior probability of each class in training dataset"""
    class_feature_count_: dict[int, list[float]] = field(default_factory=dict, init=False)
    """Feature counts with additive smoothing for each class."""
    class_feature_log_prob_: dict[int, list[float]] = field(default_factory=dict, init=False)
    """Feature conditional log probabilities for each class."""

    def fit(self, dataset: list[list[float]], labels: list[int]) -> MultinomialNaiveBayes:
        """Fit the Multinomial Naive Bayes classifier.

        Args:
            dataset: Training data, where each row is a sample and each column is a feature.
                     Features should be represented as counts (non-negative integers).
            labels: Target labels for training data.

        Returns:
            self: Returns the instance itself.
        """
        self.labels_ = sorted(set(labels))
        self.class_count_ = len(set(labels))
        # get the prior from training dataset labels
        self.class_prior_ = {label: count / len(dataset) for label, count in Counter(labels).items()}
        self.class_feature_count_, self.class_feature_log_prob_ = self._get_classes_feature_counts_log_probs(
            dataset, labels
        )
        return self

    def _log_likelihood(self, sample: list[float]) -> dict[int, float]:
        """Calculate the log-likelihood of the sample for each class."""
        class_log_likelihoods: dict[int, float] = {}
        for label in self.labels_:
            log_likelihood = 0.0
            for feature_index, feature_value in enumerate(sample):
                log_likelihood += feature_value * self.class_feature_log_prob_[label][feature_index]
            class_log_likelihoods[label] = log_likelihood
        return class_log_likelihoods

    def _get_classes_feature_counts_log_probs(
        self,
        dataset: list[list[float]],
        labels: list[int],
    ) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
        feature_count: dict[int, list[float]] = {}
        feature_log_probs: dict[int, list[float]] = {}
        for label in self.labels_:
            label_samples = [
                sample for (sample, sample_label) in zip(dataset, labels, strict=False) if sample_label == label
            ]
            counts = self._dataset_feature_counts(label_samples)
            feature_count[label] = counts
            feature_log_probs[label] = [math.log(value_count / sum(counts)) for value_count in counts]

        return feature_count, feature_log_probs

    def _dataset_feature_counts(self, dataset: list[list[float]]) -> list[float]:
        """Calculate feature counts with additive smoothing."""
        return [sum(column) + self.alpha for column in zip(*dataset, strict=True)]


@dataclass
class CategoricalNaiveBayes(BaseNaiveBayes):
    """Categorical Naive Bayes classifier.

    Examples:
        >>> import random
        >>> rng = random.Random(0)
        >>> dataset = [[rng.randint(0, 5) for _ in range(100)] for _ in range(6)]
        >>> label = [1, 2, 3, 4, 5, 6]
        >>> clf = CategoricalNaiveBayes().fit(dataset, label)
        >>> clf.predict(dataset[2])
        3
    """

    alpha: float = 1.0
    """Additive (Laplace/Lidstone) smoothing parameter"""
    labels_: list[int] = field(default_factory=list, init=False)
    """The labels in training dataset"""
    class_count_: int = field(default=0, init=False)
    """The number of classes in training dataset"""
    class_prior_: dict[int, float] = field(default_factory=dict, init=False)
    """The prior probability of each class in training dataset"""
    class_feature_count_: dict[int, dict[int, dict[float, float]]] = field(default_factory=dict, init=False)
    """Smoothed counts indexed by class label, feature index, then feature value."""
    class_feature_log_prob_: dict[int, dict[int, dict[float, float]]] = field(default_factory=dict, init=False)
    """Conditional log probabilities indexed by class label, feature index, then feature value."""

    def fit(self, dataset: list[list[float]], labels: list[int]) -> CategoricalNaiveBayes:
        """Fit the Categorical Naive Bayes classifier.

        Args:
            dataset: Training data, where each row is a sample and each column is a feature.
            labels: Target labels for training data.

        Returns:
            self: Returns the instance itself.
        """
        self.labels_ = sorted(set(labels))
        self.class_count_ = len(set(labels))
        # get the prior from training dataset labels
        self.class_prior_ = {label: count / len(dataset) for label, count in Counter(labels).items()}
        self.class_feature_count_, self.class_feature_log_prob_ = self._get_classes_feature_counts_log_probs(
            dataset, labels
        )
        return self

    def _log_likelihood(self, sample: list[float]) -> dict[int, float]:
        """Calculate the log-likelihood of the sample for each class."""
        class_log_likelihoods: dict[int, float] = {}
        for label in self.labels_:
            log_likelihood = 0.0
            for feature_index, feature_value in enumerate(sample):
                log_likelihood += self.class_feature_log_prob_[label][feature_index].get(feature_value, 0)
            class_log_likelihoods[label] = log_likelihood
        return class_log_likelihoods

    def _get_classes_feature_counts_log_probs(
        self,
        dataset: list[list[float]],
        labels: list[int],
    ) -> tuple[
        dict[int, dict[int, dict[float, float]]],
        dict[int, dict[int, dict[float, float]]],
    ]:
        smoothed_feature_counts: dict[int, dict[float, float]] = {}
        for feature_index, column in enumerate(zip(*dataset, strict=False)):
            smoothed_feature_counts[feature_index] = dict.fromkeys(set(column), self.alpha)

        feature_count: dict[int, dict[int, dict[float, float]]] = {}
        feature_log_probs: dict[int, dict[int, dict[float, float]]] = {}
        for label in self.labels_:
            label_samples = [
                sample for (sample, sample_label) in zip(dataset, labels, strict=False) if sample_label == label
            ]
            counts = self._dataset_feature_counts(label_samples, smoothed_feature_counts)
            feature_count[label] = counts
            feature_log_probs[label] = {}
            for feature_index, feature_value_count in counts.items():
                feature_log_probs[label][feature_index] = {
                    feature_value: math.log(count / sum(feature_value_count.values()))
                    for feature_value, count in feature_value_count.items()
                }

        return feature_count, feature_log_probs

    @staticmethod
    def _dataset_feature_counts(
        dataset: list[list[float]],
        smoothed_feature_counts: dict[int, dict[float, float]],
    ) -> dict[int, dict[float, float]]:
        """Calculate feature counts with additive smoothing."""
        # Each class owns its per-feature counts; numeric keys and counts can be shared.
        feature_value_count = {
            feature_index: counts.copy() for feature_index, counts in smoothed_feature_counts.items()
        }
        for feature_index, column in enumerate(zip(*dataset, strict=True)):
            for value, count in Counter(column).items():
                feature_value_count[feature_index][value] += count
        return feature_value_count

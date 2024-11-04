from __future__ import annotations

import math

from dataclasses import dataclass, field


@dataclass
class GaussianNaiveBayes:
    labels_: list[int] = field(default_factory=list)
    """The labels in training dataset"""
    class_count_: int = 0
    """The number of classes in training dataset"""
    class_prior_: dict[int, float] = field(default_factory=dict)
    """The prior probability of each class in training dataset"""
    means_: dict[int, list[float]] = field(default_factory=dict)
    """The means of each class in training dataset"""
    vars_: dict[int, list[float]] = field(default_factory=dict)
    """The variance of each class in training dataset"""

    def fit(self, dataset: list[list[float]], labels: list[int]) -> GaussianNaiveBayes:
        """Fit the naive bayes model"""
        self.labels_ = sorted(set(labels))
        self.class_count_ = len(set(labels))
        self.class_prior_ = {label: 1 / self.class_count_ for label in self.labels_}
        self.means_ = self._get_classes_means(dataset, labels)
        self.vars_ = self._get_classes_vars(dataset, labels)
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
            label_vars = self.vars_[label]
            likelihood = 1.0
            for i, xi in enumerate(sample):
                # TODO: try to calculate the log-likelihood
                likelihood *= (1 / math.sqrt(2 * math.pi * label_vars[i])) * math.exp(
                    -((xi - label_means[i]) ** 2) / (2 * label_vars[i])
                )
            label_likelihoods[label] = likelihood
        return label_likelihoods

    def _get_classes_means(self, dataset: list[list[float]], labels: list[int]) -> dict[int, list[float]]:
        dimension_num = len(dataset[0])

        label_count = {label: 0 for label in self.labels_}
        label_dimension_sums = {label: [0.0] * dimension_num for label in self.labels_}
        for label, sample in zip(labels, dataset):
            label_count[label] += 1
            for dim, xi in enumerate(sample):
                label_dimension_sums[label][dim] += xi
        means = {
            label: [dimension_sum / label_count[label] for dimension_sum in dimension_sums]
            for label, dimension_sums in label_dimension_sums.items()
        }
        return means

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


if __name__ == "__main__":
    label = [0, 0, 0, 0, 1, 1, 1, 1]
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

    clf = GaussianNaiveBayes().fit(dataset, label)

    sample = [6.00, 130, 8]
    predict_label = clf.predict(sample)
    print(predict_label)

from __future__ import annotations

import math
import random

from dataclasses import dataclass, field
from typing import Optional


def bst_expect_length(n: int) -> float:
    if n <= 1:
        return 0
    if n == 2:
        return 1
    return 2 * (math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)


@dataclass
class IsolationTree:
    """
    The isolation tree.

    Note:
        The isolation tree is a proper(full) binary tree, which has either 0 or 2 children.
    """

    max_height: int
    """The maximum height of the tree."""
    random_seed: int | None = None
    """The random seed used to initialize the centroids."""
    sample_size_: int | None = None
    """The sample size."""
    feature_num_: int | None = None
    """The number of features at each sample."""
    left_: IsolationTree | None = None
    """The left child of the tree."""
    right_: IsolationTree | None = None
    """The right child of the tree."""
    split_at_: int | None = None
    """The index of feature which is used to split the tree's samples into children."""
    split_value_: float | None = None
    """The value of split_at feature that use to split samples"""

    def __post_init__(self) -> None:
        self.random_state = random.Random(self.random_seed)
        if self.max_height < 0:
            raise ValueError(f"The max height of {self.__class__.__name__} must >= 0, not get {self.max_height}")

    def fit(self, samples: list[list[float]]) -> IsolationTree:
        """
        Fit the isolation tree.
        """
        self.sample_size_ = len(samples)
        self.feature_num_ = len(samples[0])
        # exNode
        if self.max_height == 0 or self.sample_size_ == 1:
            return self
        # inNode
        left_itree, right_itree = self._get_left_right_child_itree(samples)
        self.left_, self.right_ = left_itree, right_itree
        return self

    def get_sample_path_length(self, sample: list[float]) -> float:
        """
        Get the sample's path length to the external(leaf) node.

        Args:
            sample: The data sample.

        Returns:
            The path length of the sample.
        """
        if self.is_external_node():
            assert self.sample_size_ is not None
            if self.sample_size_ == 1:
                return 0
            else:
                return bst_expect_length(self.sample_size_)

        assert self.split_at_ is not None and self.split_value_ is not None
        if sample[self.split_at_] < self.split_value_:
            assert self.left_ is not None
            return 1 + self.left_.get_sample_path_length(sample)
        else:
            assert self.right_ is not None
            return 1 + self.right_.get_sample_path_length(sample)

    def is_external_node(self) -> bool:
        """
        The tree node is external(leaf) node or not.
        """
        if self.left_ is None and self.right_ is None:
            return True
        return False

    def _get_left_right_child_itree(
        self, samples: list[list[float]]
    ) -> tuple[Optional[IsolationTree], Optional[IsolationTree]]:
        assert self.feature_num_ is not None
        split_at_list = list(range(self.feature_num_))
        self.random_state.shuffle(split_at_list)
        for split_at in split_at_list:
            split_at_feature_values = [sample[split_at] for sample in samples]
            split_at_min, split_at_max = min(split_at_feature_values), max(split_at_feature_values)
            if math.isclose(split_at_min, split_at_max):
                continue
            split_value = self.random_state.uniform(split_at_min, split_at_max)
            left_samples, right_samples = self._get_sub_samples_by_split(samples, split_at, split_value)
            # need to keep proper binary tree property: all internal nodes have exactly two children
            if len(left_samples) > 0 and len(right_samples) > 0:
                self.split_at_, self.split_value_ = split_at, split_value
                left_itree = IsolationTree(max_height=self.max_height - 1).fit(left_samples)
                right_itree = IsolationTree(max_height=self.max_height - 1).fit(right_samples)
                return left_itree, right_itree
        # cannot split the samples by any features
        return None, None

    @staticmethod
    def _get_sub_samples_by_split(
        samples: list[list[float]],
        split_at: int,
        split_value: float,
    ) -> tuple[list[list[float]], list[list[float]]]:
        left_samples, right_samples = [], []
        for sample in samples:
            if sample[split_at] < split_value:
                left_samples.append(sample)
            else:
                right_samples.append(sample)
        return left_samples, right_samples


@dataclass
class IsolationForest:
    """
    Isolation Forest.

    Examples:
        >>> from toyml.ensemble.iforest import IsolationForest
        >>> dataset = [[-1.1], [0.3], [0.5], [100.0]]
        >>> IsolationForest(n_itree=100, max_samples=4).fit_predict(dataset)
        [1, 1, 1, -1]

    References:
        Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." 2008 eighth ieee international conference on data mining. IEEE, 2008.
    """

    n_itree: int = 100
    """The number of isolation tree in the ensemble."""
    max_samples: int = 256
    """The number of samples to draw from X to train each base estimator."""
    score_threshold: float = 0.5
    """The score threshold that is used to define outlier:
    If sample's anomaly score > score_threshold, then the sample is detected as outlier(predict return -1);
    otherwise, the sample is normal(predict return 1)"""
    random_seed: int | None = None
    """The random seed used to initialize the centroids."""
    itrees_: list[IsolationTree] = field(default_factory=list)
    """The isolation trees in the forest."""

    def __post_init__(self) -> None:
        self.random_state = random.Random(self.random_seed)

    def fit(self, dataset: list[list[float]]) -> IsolationForest:
        """
        Fit the isolation forest model.
        """
        if self.max_samples > len(dataset):
            self.max_samples = len(dataset)

        self.itrees_ = self._fit_itrees(dataset)
        return self

    def score(self, sample: list[float]) -> float:
        """
        Predict the sample's anomaly score.

        Args:
            sample: The data sample.

        Returns:
            The anomaly score.
        """
        assert len(self.itrees_) == self.n_itree, "Please fit the model before score sample!"
        itree_path_lengths = [itree.get_sample_path_length(sample) for itree in self.itrees_]
        expect_path_length = sum(itree_path_lengths) / len(itree_path_lengths)
        score = 2 ** (-expect_path_length / bst_expect_length(self.max_samples))
        return score

    def predict(self, sample: list[float]) -> int:
        """
        Predict the sample is outlier ot not.

        Args:
            sample: The data sample.

        Returns:
            Outlier: -1; Normal: 1.
        """
        score = self.score(sample)
        # outlier
        if score > self.score_threshold:
            return -1
        else:
            return 1

    def fit_predict(self, dataset: list[list[float]]) -> list[int]:
        self.fit(dataset)
        return [self.predict(sample) for sample in dataset]

    def _fit_itrees(self, dataset: list[list[float]]) -> list[IsolationTree]:
        itrees = [self._fit_itree(dataset) for _ in range(self.n_itree)]
        return itrees

    def _fit_itree(self, dataset: list[list[float]]) -> IsolationTree:
        samples = self.random_state.sample(dataset, self.max_samples)
        itree_max_height = math.ceil(math.log2(len(samples)))
        return IsolationTree(max_height=itree_max_height, random_seed=self.random_seed).fit(samples)


if __name__ == "__main__":
    simple_dataset = [[-1.1], [0.3], [0.5], [100.0]]
    result = IsolationForest(random_seed=42).fit_predict(simple_dataset)
    print(result)

import abc
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class BaseWeakLearner:
    @abc.abstractmethod
    def fit(self, dataset: list[list[float]], weights: list[float], labels: list[int]) -> BaseWeakLearner:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, x: list[float]) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_predict_labels(self) -> list[int]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_error_rate(self) -> float:
        raise NotImplementedError


@dataclass
class AdaBoost:
    """The implementation of AdaBoost algorithm.

    Examples:
        >>> from toyml.ensemble.adaboost import AdaBoost, OneDimensionClassifier
        >>> dataset = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
        >>> labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
        >>> ada = AdaBoost(weak_learner=OneDimensionClassifier, n_weak_learner=3).fit(dataset, labels)
        >>> print(f"Training dataset error rate: {ada.training_error_rate_}")
        Training dataset error rate: 0.0
        >>> test_sample = [1.5]
        >>> print(f"The label of {test_sample} is {ada.predict(test_sample)}")
        The label of [1.5] is 1

    References:
        1. Li Hang
        2. Zhou Zhihua
    """

    weak_learner: type[BaseWeakLearner]
    """
    The weak learner to be used in the AdaBoost algorithm.
    """
    n_weak_learner: int = 5
    """
    The number of weak learners to be used in the AdaBoost algorithm.
    """
    predict_labels_: list[int] | None = field(default=None, init=False)
    """
    The prediction labels of the training dataset.
    """
    training_error_rate_: float | None = field(default=None, init=False)
    """
    The error rate of the training dataset.
    """

    _n: int = field(default=-1, init=False)
    """
    The number of samples in the training dataset.
    """
    _weights: list[float] = field(default_factory=list, init=False)
    """
    The weights of samples in the training dataset.
    """
    _base_clf_labels: list[list[int]] = field(default_factory=list, init=False)
    """
    The prediction labels of the base classifiers.
    """
    _weak_learner_predicts: list[Callable[..., Any]] = field(default_factory=list, init=False)
    """
    The prediction functions of the weak learners.
    """
    _alphas: list[float] = field(default_factory=list, init=False)
    """
    The alpha values of the weak learners.
    """

    def fit(
        self,
        dataset: list[list[float]],
        labels: list[int],
    ) -> AdaBoost:
        """Fit the AdaBoost model."""
        # for model training(Gm)
        self._n = len(labels)
        self._weights = [1.0 / self._n] * self._n
        self.predict_labels_ = None
        self.training_error_rate_ = None
        self._base_clf_labels = []
        # base clf models
        self._weak_learner_predicts: list[Callable[..., int]] = []
        self._alphas = []

        for _ in range(self.n_weak_learner):
            model = self.weak_learner().fit(dataset, self._weights, labels)
            error_rate = model.get_error_rate()
            if error_rate >= 0.5:
                if not self._weak_learner_predicts:
                    msg = "Weak learner error rate must be less than 0.5 to fit AdaBoost."
                    raise ValueError(msg)
                break
            predictions = model.get_predict_labels()
            self._base_clf_labels.append(predictions)
            self._weak_learner_predicts.append(model.predict)
            if error_rate == 0:
                # Use a finite vote and stop before dividing by zero, as in sklearn.
                self._alphas.append(1.0)
                break
            alpha = 0.5 * math.log((1 - error_rate) / error_rate)
            self._alphas.append(alpha)
            # update the weights
            weights = [0.0] * self._n
            for i in range(self._n):
                weights[i] = self._weights[i] * math.exp(-alpha * labels[i] * predictions[i])
            self._weights = [weight / sum(weights) for weight in weights]
        # collect training dataset result
        self.predict_labels_ = [self.predict(x) for x in dataset]
        self.training_error_rate_ = sum(self.predict_labels_[i] != labels[i] for i in range(self._n)) / self._n
        return self

    def predict(self, x: list[float]) -> int:
        """Predict the label of the input sample."""
        if not self._weak_learner_predicts:
            msg = "The model is not fitted yet!"
            raise ValueError(msg)
        ensemble_predict = 0
        for alpha, model_predict in zip(self._alphas, self._weak_learner_predicts, strict=True):
            ensemble_predict += alpha * model_predict(x)
        if ensemble_predict >= 0:
            return 1
        return -1


@dataclass
class OneDimensionClassifier(BaseWeakLearner):
    """Binary classifier with one dimension feature.

    Ref: Li Hang, 1 ed, E8.1.3
    """

    # Keep Enum's existing string representation for compatibility.
    class SignMode(str, Enum):  # noqa: D106, UP042
        POS_NEG = "POS_NEG"
        NEG_POS = "NEG_POS"

    _sign_mode: SignMode = field(default=SignMode.POS_NEG, init=False)
    _best_cut: float | None = field(default=None, init=False)
    error_rate_: float = field(default=math.inf, init=False)
    predict_labels_: list[int] | None = field(default=None, init=False)

    def fit(
        self,
        dataset: list[list[float]],
        weights: list[float],
        labels: list[int],
    ) -> OneDimensionClassifier:
        """Fit the one-dimension classifier."""
        # search for the best cut point
        sign_mode, best_cut, best_error_rate = self.get_best_cut(dataset, weights, labels)
        self.error_rate_ = best_error_rate
        self._best_cut = best_cut
        self._sign_mode = sign_mode
        # get labels
        self.predict_labels_ = [0] * len(labels)
        for i, x in enumerate(dataset):
            self.predict_labels_[i] = self.predict(x)
        return self

    def predict(self, x: list[float]) -> int:
        """Predict the label of the input sample."""
        if self._best_cut is None:
            msg = "The model is not fitted yet!"
            raise ValueError(msg)
        if self._sign_mode == "POS_NEG":
            if x[0] <= self._best_cut:
                return 1
            return -1
        if x[0] <= self._best_cut:
            return -1
        return 1

    def get_error_rate(self) -> float:
        """Get the error rate of the training dataset."""
        if self.predict_labels_ is None:
            msg = "The model is not fitted yet!"
            raise ValueError(msg)
        return self.error_rate_

    def get_predict_labels(self) -> list[int]:
        """Get the prediction labels of the training dataset."""
        if self.predict_labels_ is None:
            msg = "The model is not fitted yet!"
            raise ValueError(msg)
        return self.predict_labels_.copy()

    @staticmethod
    def _get_candidate_cuts(points: list[float]) -> list[float]:
        """Get the candidate cuts of the training dataset."""
        min_x_int = math.floor(min(points))
        max_x_int = math.ceil(max(points))
        return [i + 0.5 for i in range(min_x_int, max_x_int)]

    def get_best_cut(
        self,
        dataset: list[list[float]],
        weights: list[float],
        labels: list[int],
    ) -> tuple[SignMode, float, float]:
        """Get the best cut of the training dataset."""
        points = [x[0] for x in dataset]
        candidate_cuts = self._get_candidate_cuts(points)
        # (func_mode, cut, error_rate)
        candidate_cuts_result = []
        for cut in candidate_cuts:
            pos_neg_error_rate = self._get_cut_error_rate(cut, points, weights, labels, self.SignMode.POS_NEG)
            neg_pos_error_rate = self._get_cut_error_rate(cut, points, weights, labels, self.SignMode.NEG_POS)
            candidate_cuts_result.extend(
                [(self.SignMode.POS_NEG, cut, pos_neg_error_rate), (self.SignMode.NEG_POS, cut, neg_pos_error_rate)],
            )

        # sorted by error rate
        best_cut_result = sorted(candidate_cuts_result, key=lambda x: x[2])[0]
        sign_mode, best_cut, best_error_rate = best_cut_result
        return sign_mode, best_cut, best_error_rate

    def _get_cut_error_rate(
        self,
        cut: float,
        points: list[float],
        weights: list[float],
        labels: list[int],
        sign_mode: SignMode,
    ) -> float:
        """Get the error rate of the training dataset."""
        if sign_mode == self.SignMode.POS_NEG:
            error_rate = sum(
                weights[i]
                for i, x in enumerate(points)
                if (x <= cut and labels[i] != 1) or (x > cut and labels[i] != -1)
            )
        else:
            error_rate = sum(
                weights[i]
                for i, x in enumerate(points)
                if (x <= cut and labels[i] != -1) or (x > cut and labels[i] != 1)
            )
        return error_rate


if __name__ == "__main__":
    dataset_demo: list[list[float]] = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    labels_demo: list[int] = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    ada = AdaBoost(weak_learner=OneDimensionClassifier, n_weak_learner=3).fit(dataset_demo, labels_demo)
    print(f"Training dataset prediction labels: {ada.predict_labels_}")
    print(f"Training dataset error rate: {ada.training_error_rate_}")
    test_sample = [1.5]
    print(f"The label of {test_sample} is {ada.predict(test_sample)}")

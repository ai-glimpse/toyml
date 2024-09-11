from __future__ import annotations

import abc
import logging
import math

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class BaseWeakLeaner:
    @abc.abstractmethod
    def fit(self, dataset: list[list[float]], weights: list[float], labels: list[int]) -> BaseWeakLeaner:
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
    """
    The implementation of AdaBoost algorithm.

    Ref:
    1. Li Hang
    2. Zhou
    """

    weak_learner: Type[BaseWeakLeaner]
    n_weak_learner: int = 5
    predict_labels_: Optional[list[int]] = None
    training_error_rate_: Optional[float] = None

    _n: int = -1
    _labels: list[int] = field(default_factory=list)
    _weights: list[float] = field(default_factory=list)
    _base_clf_labels: list[list[int]] = field(default_factory=list)
    _weak_learner_predicts: List[Callable[..., Any]] = field(default_factory=list)
    _alphas: list[float] = field(default_factory=list)

    def fit(
        self,
        dataset: list[list[float]],
        labels: list[int],
    ) -> AdaBoost:
        self._labels = labels
        # for model training(Gm)
        self._n = len(labels)
        self._weights: Any = [1.0 / self._n] * self._n
        # we use -2 to initialize the class which can handle cases
        # such as multi-classes(0, 1, 2, ...) and binary classes(-1, 1)
        self._base_clf_labels = [[-2] * self._n for _ in range(self.n_weak_learner)]
        # base clf models
        self._weak_learner_predicts: List[Callable[..., int]] = []
        self._alphas = [0.0] * self.n_weak_learner

        for m in range(self.n_weak_learner):
            model = self.weak_learner().fit(dataset, self._weights, self._labels)
            self._base_clf_labels[m] = model.get_predict_labels()
            self._weak_learner_predicts.append(model.predict)
            error_rate = model.get_error_rate()
            # Warning when the error rate is too large
            if error_rate > 0.5:
                logger.warning(f"Weak learner error rate = {error_rate} < 0.5")
            alpha = 0.5 * math.log((1 - error_rate) / error_rate)
            self._alphas[m] = alpha
            # update the weights
            weights = [0.0] * self._n
            for i in range(self._n):
                weights[i] = self._weights[i] * math.exp(-alpha * self._labels[i] * self._base_clf_labels[m][i])
            self._weights = [weight / sum(weights) for weight in weights]
        # collect training dataset result
        self.predict_labels_ = [self.predict(x) for x in dataset]
        self.training_error_rate_ = sum(self.predict_labels_[i] != self._labels[i] for i in range(self._n)) / self._n
        return self

    def predict(self, x: list[float]) -> int:
        ensemble_predict = 0
        for m in range(self.n_weak_learner):
            model_predict = self._weak_learner_predicts[m]
            ensemble_predict += self._alphas[m] * model_predict(x)
        if ensemble_predict >= 0:
            return 1
        else:
            return -1

    def _get_(self) -> list[int]:
        predictions = [-2] * self._n
        for i in range(self._n):
            result = 0.0
            for m in range(self.n_weak_learner):
                result += self._alphas[m] * self._base_clf_labels[m][i]
            if result >= 0:
                predictions[i] = 1
            else:
                predictions[i] = -1
        return predictions


@dataclass
class OneDimensionClassifier(BaseWeakLeaner):
    """
    Binary classifier with one dimension feature.

    Ref: Li Hang, 1 ed, E8.1.3
    """

    class SignMode(str, Enum):
        POS_NEG = "POS_NEG"
        NEG_POS = "NEG_POS"

    _sign_mode: SignMode = SignMode.POS_NEG
    _best_cut: float = math.inf
    error_rate_: float = math.inf
    predict_labels_: Optional[list[int]] = None

    def fit(
        self,
        dataset: list[list[float]],
        weights: list[float],
        labels: list[int],
    ) -> OneDimensionClassifier:
        # search for the best cut point
        sign_mode, best_cut, best_error_rate = self.get_best_cut(dataset, weights)
        self.error_rate_ = best_error_rate
        self._best_cut = best_cut
        self._sign_mode = sign_mode
        # get labels
        self.predict_labels_ = [0] * len(labels)
        for i, x in enumerate(dataset):
            self.predict_labels_[i] = self.predict(x)
        return self

    def predict(self, x: list[float]) -> int:
        if self._best_cut is None:
            raise ValueError("The model is not fitted yet!")
        if self._sign_mode == "POS_NEG":
            if x[0] <= self._best_cut:
                return 1
            else:
                return -1
        if x[0] <= self._best_cut:
            return -1
        else:
            return 1

    def get_error_rate(self) -> float:
        if self.error_rate_ is None:
            raise ValueError("The model is not fitted yet!")
        return self.error_rate_

    def get_predict_labels(self) -> list[int]:
        if self.predict_labels_ is None:
            raise ValueError("The model is not fitted yet!")
        return self.predict_labels_

    @staticmethod
    def _get_candidate_cuts(points: list[float]) -> list[float]:
        min_x_int = math.floor(min(points))
        max_x_int = math.ceil(max(points))
        return [i + 0.5 for i in range(min_x_int, max_x_int)]

    def get_best_cut(self, dataset: list[list[float]], weights: list[float]) -> tuple[SignMode, float, float]:
        points = [x[0] for x in dataset]
        candidate_cuts = self._get_candidate_cuts(points)
        # (func_mode, cut, error_rate)
        candidate_cuts_result = []
        for cut in candidate_cuts:
            pos_neg_error_rate = self._get_cut_error_rate(cut, points, weights, self.SignMode.POS_NEG)
            neg_pos_error_rate = self._get_cut_error_rate(cut, points, weights, self.SignMode.NEG_POS)
            candidate_cuts_result.extend(
                [(self.SignMode.POS_NEG, cut, pos_neg_error_rate), (self.SignMode.NEG_POS, cut, neg_pos_error_rate)]
            )

        # sorted by error rate
        best_cut_result = sorted(candidate_cuts_result, key=lambda x: x[2])[0]
        sign_mode, best_cut, best_error_rate = best_cut_result
        return sign_mode, best_cut, best_error_rate

    def _get_cut_error_rate(self, cut: float, points: list[float], weights: list[float], sign_mode: SignMode) -> float:
        error_rate = 0.0
        for i, x in enumerate(points):
            if sign_mode == self.SignMode.POS_NEG:
                if x <= cut and labels[i] != 1 or x > cut and labels[i] != -1:
                    error_rate += weights[i]
            else:
                if x <= cut and labels[i] != -1 or x > cut and labels[i] != 1:
                    error_rate += weights[i]
        return error_rate


if __name__ == "__main__":
    dataset: list[list[float]] = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    labels: list[int] = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    ada = AdaBoost(weak_learner=OneDimensionClassifier, n_weak_learner=3).fit(dataset, labels)
    print(f"Training dataset prediction labels: {ada.predict_labels_}")
    print(f"Training dataset error rate: {ada.training_error_rate_}")
    test_sample = [1.5]
    print(f"The label of {test_sample} is {ada.predict(test_sample)}")

import logging
import math

from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional

logger = logging.getLogger(__name__)


class AdaBoost:
    """
    The implementation of AdaBoost algorithm.

    Ref:
    1. Li Hang
    2. Zhou
    """

    def __init__(
        self,
        dataset: list[list[float]],
        labels: list[int],
        base_clf: Callable[..., Any],  # TODO: change to clf class type
        clf_num: int = 5,
    ) -> None:
        self._dataset = dataset
        self._labels = labels
        self._base_clf = base_clf
        self._clf_num = clf_num
        # for model training(Gm)
        self._n = len(labels)
        self._weights: Any = [1.0 / self._n] * self._n
        # we use -2 to initialize the class which can handle cases
        # such as multi-classes(0, 1, 2, ...) and binary classes(-1, 1)
        self._base_clf_results = [[-2] * self._n for _ in range(self._clf_num)]
        # base clf models
        self._sub_clf_models: List[Callable[..., Any]] = []
        self._alphas: Any = [0] * self._clf_num

    def fit(self) -> None:
        for m in range(self._clf_num):
            model = self._base_clf()
            model.fit(self._dataset, self._weights, self._labels)
            self._base_clf_results[m] = model.get_predict_labels()
            self._sub_clf_models.append(model.predict)
            error_rate = model.get_error_rate()
            # Warning when the error rate is too large
            if error_rate > 0.5:
                logger.warning(f"Base Clf error rate = {error_rate}!")
            alpha = 0.5 * math.log((1 - error_rate) / error_rate)
            self._alphas[m] = alpha
            # update the weights
            weights = [0] * self._n
            for i in range(self._n):
                weights[i] = self._weights[i] * math.exp(-alpha * self._labels[i] * self._base_clf_results[m][i])
            self._weights = [weight / sum(weights) for weight in weights]

    def get_training_result(self) -> list[int]:
        predictions = [-2] * self._n
        for i in range(self._n):
            result = 0
            for m in range(self._clf_num):
                result += self._alphas[m] * self._base_clf_results[m][i]
            if result >= 0:
                predictions[i] = 1
            else:
                predictions[i] = -1
        training_error = sum(predictions[i] != self._labels[i] for i in range(self._n)) / self._n
        print("Training Error: ", training_error)
        print("Predictions: ", predictions)
        return predictions

    def predict(self, x: float) -> int:
        result = 0
        for m in range(self._clf_num):
            model_predict = self._sub_clf_models[m]
            result += self._alphas[m] * model_predict(x)
        if result >= 0:
            return 1
        else:
            return -1


@dataclass
class OneDimensionClassifier:
    """
    Binary classifier with one dimension feature.

    Ref: Li Hang, 1ed, E8.1.3
    """

    error_rate_: float = math.inf
    best_cut_: float = math.inf
    predict_labels_: Optional[list[int]] = None
    # TODO: make func mode an enum
    _func_mode: Literal["pos-neg", "neg-pos"] = "pos-neg"

    def fit(
        self,
        dataset: list[list[float]],
        weights: list[float],
        labels: list[int],
    ) -> None:
        points = [x[0] for x in dataset]
        # search for the best cut point
        func_mode, best_cut, best_error_rate = self.get_best_cut(points, weights)
        self.error_rate_ = best_error_rate
        self.best_cut_ = best_cut
        self._func_mode = func_mode
        # get labels
        self.predict_labels_ = [0] * len(labels)
        for i, x in enumerate(points):
            self.predict_labels_[i] = self.predict(x)

    @staticmethod
    def _get_candidate_cuts(dataset: list[list[float]]) -> list[float]:
        points = [x[0] for x in dataset]
        min_x_int = math.floor(min(points))
        max_x_int = math.ceil(max(points))
        return [i + 0.5 for i in range(min_x_int, max_x_int)]

    def get_best_cut(
        self, points: list[float], weights: list[float]
    ) -> tuple[Literal["pos-neg", "neg-pos"], float, float]:
        best_error_rate = math.inf
        best_cut = math.inf
        candidate_cuts = self._get_candidate_cuts(dataset)
        # (func_mode, cut, error_rate)
        candidate_cuts_result = []
        for cut in candidate_cuts:
            pos_neg_error_rate = self._get_cut_error_rate(cut, points, weights, "pos-neg")
            neg_pos_error_rate = self._get_cut_error_rate(cut, points, weights, "neg-pos")
            candidate_cuts_result.append(("pos-neg", cut, pos_neg_error_rate))
            candidate_cuts_result.append(("neg-pos", cut, neg_pos_error_rate))

        # sorted by error rate
        best_cut_result = sorted(candidate_cuts_result, key=lambda x: x[2])[-1]
        func_mode, best_cut, best_error_rate = best_cut_result
        return func_mode, best_cut, best_error_rate  # type: ignore

    def _get_cut_error_rate(self, cut: float, points: list[float], weights: list[float], func_mode: str) -> float:
        error_rate = 0.0
        for j, x in enumerate(points):
            if func_mode == "pos-neg":
                if x <= cut and labels[j] != 1 or x > cut and labels[j] != -1:
                    error_rate += weights[j]
            else:
                if x <= cut and labels[j] != -1 or x > cut and labels[j] != 1:
                    error_rate += weights[j]
        return error_rate

    def get_error_rate(self) -> float:
        if self.error_rate_ is None:
            raise ValueError("The model is not fitted yet!")
        return self.error_rate_

    def get_predict_labels(self) -> list[int]:
        if self.predict_labels_ is None:
            raise ValueError("The model is not fitted yet!")
        return self.predict_labels_

    def predict(self, x: float) -> int:
        if self.best_cut_ is None:
            raise ValueError("The model is not fitted yet!")
        if self._func_mode == "pos-neg":
            if x <= self.best_cut_:
                return 1
            else:
                return -1
        if x <= self.best_cut_:
            return -1
        else:
            return 1


if __name__ == "__main__":
    dataset: list[list[float]] = [[0.0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    labels: list[int] = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    M: int = 3
    ada = AdaBoost(dataset, labels, OneDimensionClassifier, M)
    ada.fit()
    ada.get_training_result()
    TEST_X = 1.5
    print(f"The label of {TEST_X} is {ada.predict(TEST_X)}")

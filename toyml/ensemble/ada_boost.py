import math

from typing import Any

from toyml.utils.types import DataSet, Label, Labels, Weights


class AdaBoost:
    """
    The implementation of AdaBoost algorithm.

    Ref:
    1. Li Hang
    2. Zhou
    """

    def __init__(
        self, dataset: DataSet, labels: Labels, base_clf, clf_num: int = 5
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
        self._sub_clf_models = []
        self._alphas: Any = [0] * self._clf_num

    def fit(self) -> None:
        for m in range(self._clf_num):
            model = self._base_clf(self._dataset, self._weights, self._labels)
            model.fit()
            self._base_clf_results[m] = model.get_training_results()
            self._sub_clf_models.append(model.predict)
            error_rate = model.get_error_rate()
            # Warning when the error rate is too large
            if error_rate > 0.5:
                Warning(f"Base Clf error rate = {error_rate}!")
            alpha = 0.5 * math.log((1 - error_rate) / error_rate)
            self._alphas[m] = alpha
            # update the weights
            weights = [0] * self._n
            for i in range(self._n):
                weights[i] = self._weights[i] * math.exp(
                    -alpha * self._labels[i] * self._base_clf_results[m][i]
                )
            self._weights = [weight / sum(weights) for weight in weights]

    def get_training_result(self) -> Labels:
        predictions = [-2] * self._n
        for i in range(self._n):
            result = 0
            for m in range(self._clf_num):
                result += self._alphas[m] * self._base_clf_results[m][i]
            if result >= 0:
                predictions[i] = 1
            else:
                predictions[i] = -1
        training_error = (
            sum(predictions[i] != self._labels[i] for i in range(self._n)) / self._n
        )
        print("Training Error: ", training_error)
        print("Predictions: ", predictions)
        return predictions

    def predict(self, x: float) -> Label:
        result = 0
        for m in range(self._clf_num):
            model_predict = self._sub_clf_models[m]
            result += self._alphas[m] * model_predict(x)
        if result >= 0:
            return 1
        else:
            return -1


class OneDimClf:
    """
    Binary clf with only one feature.

    Ref: Li Hang, 1ed, E8.1.3
    """

    def __init__(self, dataset: DataSet, weights: Weights, labels: Labels) -> None:
        self._dataset = dataset
        self._xs = [x[0] for x in self._dataset]
        self._weights = weights
        self._labels = labels
        self._n = len(self._labels)
        # for training and prediction
        self._error_rate = math.inf
        self._best_cut = math.inf
        self._func_mode = "pos-neg"

    def fit(self) -> None:
        min_x_int = math.floor(min(self._xs))
        max_x_int = math.ceil(max(self._xs))
        # search for the best cut point
        best_cut = min_x_int
        best_error_rate = math.inf
        # pos-neg: 1, -1(default)
        for i in range(min_x_int, max_x_int):
            cut = i + 0.5
            error_rate = 0
            for j, x in enumerate(self._xs):
                if x <= cut and self._labels[j] != 1:
                    error_rate += self._weights[j]
                if x > cut and self._labels[j] != -1:
                    error_rate += self._weights[j]
            if error_rate < best_error_rate:
                best_error_rate = error_rate
                best_cut = cut
        # neg-pos: -1, 1
        for i in range(min_x_int, max_x_int):
            cut = i + 0.5
            error_rate = 0
            for j, x in enumerate(self._xs):
                if x <= cut and self._labels[j] != -1:
                    error_rate += self._weights[j]
                if x > cut and self._labels[j] != 1:
                    error_rate += self._weights[j]
            if error_rate < best_error_rate:
                self._func_mode = "neg-pos"
                best_error_rate = error_rate
                best_cut = cut
        self._error_rate = best_error_rate
        self._best_cut = best_cut

    def get_error_rate(self) -> float:
        return self._error_rate

    def get_training_results(self) -> Labels:
        results = [-2] * self._n
        for i, x in enumerate(self._xs):
            results[i] = self.predict(x)
        return results

    def predict(self, x):
        if self._func_mode == "pos-neg":
            if x <= self._best_cut:
                return 1
            else:
                return -1
        if x <= self._best_cut:
            return -1
        else:
            return 1


if __name__ == "__main__":
    dataset = [[0.0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    labels = [1, 1, 1, -1, -1, -1, 1, 1, 1, -1]
    M = 3
    ada = AdaBoost(dataset, labels, OneDimClf, M)
    ada.fit()
    ada.get_training_result()
    TEST_X = 1.5
    print(f"The label of {TEST_X} is {ada.predict(TEST_X)}")

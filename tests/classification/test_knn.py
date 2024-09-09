import numpy as np
import pytest

from hypothesis import given
from hypothesis import strategies as st

from toyml.classification.knn import KNN, Standardizationer


class TestKNN:
    @given(
        dataset=st.lists(
            st.lists(
                st.floats(allow_nan=False, allow_infinity=False, max_value=100, min_value=-100), min_size=4, max_size=4
            ),
            min_size=2,
            max_size=10,
        ),
        labels=st.lists(st.integers(min_value=0, max_value=9), min_size=2, max_size=10),
        k=st.integers(min_value=1, max_value=10),
        std_transform=st.booleans(),
    )
    def test_knn_fit_predict(self, dataset: list[list[float]], labels: list[int], k: int, std_transform: bool) -> None:
        # Ensure dataset and labels have the same length
        dataset = dataset[: len(labels)]
        labels = labels[: len(dataset)]

        knn = KNN(k=k, std_transform=std_transform)
        knn.fit(dataset, labels)

        # Test prediction on a random point from the dataset
        random_index = np.random.randint(0, len(dataset))
        prediction = knn.predict(dataset[random_index])
        assert prediction in labels

    @pytest.mark.parametrize(
        "dataset, labels, k, test_point, expected_label",
        [
            ([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1], 3, [2.5], 0),
            ([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], ["A", "A", "B", "B"], 2, [3.5, 3.5], "B"),
        ],
    )
    def test_knn_specific_cases(
        self, dataset: list[list[float]], labels: list[int], k: int, test_point: list[float], expected_label: int
    ) -> None:
        knn = KNN(k=k, std_transform=False)
        knn.fit(dataset, labels)
        prediction = knn.predict(test_point)
        assert prediction == expected_label

    def test_knn_not_fitted(self) -> None:
        knn = KNN(k=3)
        with pytest.raises(ValueError, match="The model is not fitted yet!"):
            knn.predict([1.0, 2.0])


class TestStandardizationer:
    @pytest.mark.parametrize(
        "dataset, expected_means, expected_stds",
        [
            ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [3.0, 4.0], [2.0, 2.0]),
            ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [1.0, 1.0], [1.0, 1.0]),
            ([[1.0], [2.0], [3.0]], [2.0], [1.0]),
        ],
    )
    def test_fit(self, dataset: list[list[float]], expected_means: list[float], expected_stds: list[float]) -> None:
        standardizationer = Standardizationer()
        standardizationer.fit(dataset)

        assert standardizationer._dimension == len(dataset[0])
        assert pytest.approx(standardizationer._means) == expected_means
        assert pytest.approx(standardizationer._stds) == expected_stds

    @pytest.mark.parametrize(
        "dataset, input_data, expected_output",
        [
            ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[2.0, 3.0]], [[-0.5, -0.5]]),
            ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [[1.5, 1.5]], [[0.5, 0.5]]),
            ([[1.0], [2.0], [3.0]], [[2.5]], [[0.5]]),
        ],
    )
    def test_transform(
        self, dataset: list[list[float]], input_data: list[list[float]], expected_output: list[list[float]]
    ) -> None:
        standardizationer = Standardizationer()
        standardizationer.fit(dataset)

        transformed = standardizationer.transform(input_data)
        np.testing.assert_array_almost_equal(transformed, expected_output)

    @pytest.mark.parametrize(
        "dataset, expected_output",
        [
            ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
            ([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
            ([[1.0], [2.0], [3.0]], [[-1.0], [0.0], [1.0]]),
        ],
    )
    def test_fit_transform(self, dataset: list[list[float]], expected_output: list[list[float]]) -> None:
        standardizationer = Standardizationer()
        transformed = standardizationer.fit_transform(dataset)
        np.testing.assert_array_almost_equal(transformed, expected_output)

    def test_not_fitted(self) -> None:
        standardizationer = Standardizationer()
        with pytest.raises(ValueError, match="The model is not fitted yet!"):
            standardizationer.transform([[1.0, 2.0]])

    def test_inconsistent_dimensions(self) -> None:
        standardizationer = Standardizationer()
        with pytest.raises(ValueError, match="All rows must have the same number of columns"):
            standardizationer.fit([[1.0, 2.0], [3.0, 4.0, 5.0]])

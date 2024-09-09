import numpy as np
import pytest

from toyml.classification.knn import Standardizationer


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

"""KNN classification and feature standardization contracts."""

import math
from dataclasses import asdict

import pytest
from hypothesis import given
from hypothesis import strategies as st

from toyml.classification import KNN, Standardizer


class TestConfiguration:
    def test_default_configuration(self) -> None:
        """Use standardized Euclidean distances when only the neighbor count is supplied."""
        model = KNN(3)
        assert (model.k, model.std_transform, model.distance_metric) == (3, True, "euclidean")

    def test_dataclass_exports_public_configuration_and_fitted_state(self) -> None:
        """Expose the settings and fitted data as top-level dataclass fields."""
        model = KNN(1, False).fit([[2.0]], ["A"])
        exported = asdict(model)
        assert exported["k"] == 1
        assert exported["std_transform"] is False
        assert exported["distance_metric"] == "euclidean"
        assert exported["dataset_"] == [[2.0]]
        assert exported["labels_"] == ["A"]
        assert exported["standardizer_"] is None


class TestFit:
    @pytest.mark.parametrize("k", [0, True, 1.5, 2])
    def test_invalid_k_is_rejected_at_fit(self, k: object) -> None:
        """Require an integer from one to the sample count, excluding booleans."""
        model = KNN(k=k)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match="k"):
            model.fit([[0.0]], ["A"])

    def test_standardization_requires_a_boolean_at_fit(self) -> None:
        """Reject the integer 1 even though it compares equal to True."""
        model = KNN(1, std_transform=1)  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match="std_transform"):
            model.fit([[0.0]], ["A"])

    def test_refit_validates_changed_distance_metric(self) -> None:
        """Check the current metric even when an earlier fit succeeded."""
        model = KNN(1).fit([[0.0]], ["A"])
        model.distance_metric = "manhattan"  # ty: ignore[invalid-assignment]
        with pytest.raises(ValueError, match="distance metric"):
            model.fit([[0.0]], ["A"])

    @pytest.mark.parametrize("labels", [[], [1, 2], [[]]])
    def test_requires_one_hashable_label_per_row(self, labels: list[object]) -> None:
        """Each training row needs one label that can be counted in a vote."""
        with pytest.raises(ValueError, match="label"):
            KNN(1).fit([[0.0]], labels)

    @pytest.mark.parametrize("samples", [[[]], [[0.0], [1.0, 2.0]], [[math.nan]]])
    @pytest.mark.parametrize("standardize", [True, False])
    def test_requires_rectangular_finite_samples(self, samples: list[list[float]], standardize: bool) -> None:
        """Both preprocessing paths require finite rows of the same positive width."""
        with pytest.raises(ValueError, match="sample|feature|columns|numeric"):
            KNN(1, standardize).fit(samples, ["A"] * len(samples))

    @pytest.mark.parametrize("standardize", [True, False])
    def test_training_rows_and_labels_are_owned_by_the_model(self, standardize: bool) -> None:
        """Caller edits after fitting must leave the original neighbor and its label available."""
        samples, labels = [[10.0], [20.0]], ["A", "B"]
        model = KNN(1, standardize).fit(samples, labels)
        assert samples == [[10.0], [20.0]]
        samples[0][0] = 100.0
        labels[0] = "changed"
        query = [10.0]
        assert model.predict(query) == "A"
        assert query == [10.0]

    @pytest.mark.parametrize("standardize", [True, False])
    def test_refit_replaces_training_data(self, standardize: bool) -> None:
        """The old point at zero must take a new label after its training row is replaced."""
        model = KNN(1, standardize).fit([[0.0]], ["old"])
        assert model.fit([[100.0], [200.0]], ["near", "far"]) is model
        assert model.predict([0.0]) == "near"
        assert model.predict([200.0]) == "far"

    @pytest.mark.parametrize("invalid", [[], [[-1.7e308], [1.7e308]]], ids=["empty", "scaling-overflow"])
    def test_failed_refit_preserves_predictions(self, invalid: list[list[float]]) -> None:
        """Keep both old predictions after either input validation or standardization fails."""
        model = KNN(1).fit([[10.0], [20.0]], ["A", "B"])
        with pytest.raises(ValueError):  # noqa: PT011
            model.fit(invalid, ["new"] * len(invalid))
        assert model.predict([10.0]) == "A"
        assert model.predict([20.0]) == "B"


class TestPredict:
    def test_requires_fit(self) -> None:
        """Report an unfitted model even for a valid query."""
        with pytest.raises(ValueError, match="not fitted"):
            KNN(1).predict([0.0])

    @pytest.mark.parametrize(
        ("standardize", "query"),
        [(True, []), (False, [0.0, 1.0]), (True, ["bad"]), (False, [math.inf]), (False, [10**400])],
    )
    def test_query_must_match_training_features(self, standardize: bool, query: list[float]) -> None:
        """Require the training width and finite numeric values for each new query."""
        model = KNN(1, standardize).fit([[0.0]], ["A"])
        with pytest.raises(ValueError, match="sample|feature|dimension|numeric"):
            model.predict(query)

    def test_refit_uses_the_new_neighbor_count_for_voting(self) -> None:
        """In distance order the labels are A, B, B, A, A, so the majority changes twice."""
        samples = [[4.0], [2.0], [0.0], [3.0], [1.0]]
        labels = ["A", "B", "A", "A", "B"]
        model = KNN(1, False).fit(samples, labels)
        assert model.predict([0.1]) == "A"
        model.k = 3
        model.fit(samples, labels)
        assert model.predict([0.1]) == "B"
        model.k = len(samples)
        model.fit(samples, labels)
        assert model.predict([0.1]) == "A"

    @pytest.mark.parametrize("first_point", [[-1.0, 0.0], [1.0, 2**-27]], ids=["exact-tie", "rounded-tie"])
    def test_equal_distances_at_cutoff_keep_training_order(self, first_point: list[float]) -> None:
        """Both computed distances are 1.0, even when the first point's squared distance is 1 + 2⁻⁵⁴."""
        model = KNN(1, False).fit([first_point, [1.0, 0.0]], ["Z", "A"])
        assert model.predict([0.0, 0.0]) == "Z"

    def test_vote_tie_favors_first_label_in_distance_order(self) -> None:
        """Each label gets one vote. Z is nearer even though A was fitted first."""
        model = KNN(2, False).fit([[2.0], [1.0]], ["A", "Z"])
        assert model.predict([0.0]) == "Z"

    @pytest.mark.parametrize("scale", [1e-250, 1.0, 1e250])
    def test_euclidean_metric_ranks_by_l2_distance_across_scales(self, scale: float) -> None:
        """The diagonal wins under L2 at every scale. L1 would choose the axis point."""
        # A spurious tie from squaring at an extreme scale would select the first (axis) point.
        samples = [[0.0, 5 * scale], [3 * scale, 3 * scale]]
        model = KNN(1, False, distance_metric="euclidean").fit(samples, ["axis", "diagonal"])
        assert model.predict([0.0, 0.0]) == "diagonal"

    @pytest.mark.parametrize("limit", [1e308, 10**308])
    def test_finite_distance_ranks_before_overflow(self, limit: float) -> None:
        """The distance to the origin is finite even when the distance to the first sample overflows."""
        model = KNN(1, False).fit([[-limit], [0.0]], ["far", "near"])
        assert model.predict([limit]) == "near"

    def test_refit_without_standardization_uses_original_feature_units(self) -> None:
        """Scaling favors B's matching second coordinate. In original units A is closer."""
        samples = [[100.0, 0.0], [110.0, 1.0], [200.0, 0.0]]
        labels = ["A", "B", "C"]
        model = KNN(1).fit(samples, labels)
        assert model.predict([100.0, 1.0]) == "B"
        model.std_transform = False
        model.fit(samples, labels)
        assert model.predict([100.0, 1.0]) == "A"

    def test_rejects_unrepresentable_standardized_query(self) -> None:
        """A finite query can overflow when divided by a tiny training scale."""
        model = KNN(1).fit([[0.0], [1e-250]], ["A", "B"])
        with pytest.raises(ValueError, match="floating-point range"):
            model.predict([1e308])


class TestStandardizer:
    @pytest.mark.parametrize("magnitude", [1.6e308, 16 * 10**307])
    def test_recovers_finite_scores_when_centering_overflows(self, magnitude: float) -> None:
        """The last deviation is -1.5 times the magnitude, but its z-score is just -1.5."""
        rows = [[magnitude], [magnitude], [magnitude], [-magnitude]]
        scores = Standardizer().fit_transform(rows)
        assert [row[0] for row in scores] == pytest.approx([0.5, 0.5, 0.5, -1.5])

    def test_rejects_an_unrepresentable_score_for_integer_features(self) -> None:
        """With unit scale the final score is still too large for a float."""
        scaler = Standardizer().fit([[10**308]])
        with pytest.raises(ValueError, match="floating-point range"):
            scaler.transform([[-(10**308)]])

    @pytest.mark.parametrize(("row", "reason"), [([1.0, 2.0], "dimension"), ([10**400], "finite numeric")])
    def test_transform_validates_new_rows(self, row: list[float], reason: str) -> None:
        """Check the fitted width and numeric range on new data passed directly to the scaler."""
        scaler = Standardizer().fit([[0.0]])
        with pytest.raises(ValueError, match=reason):
            scaler.transform([row])

    def test_uses_training_mean_and_sample_standard_deviation(self) -> None:
        """Reuse the two training scales, 2 and 10, when transforming a new row."""
        scaler = Standardizer()
        rows = [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]
        assert scaler.fit_transform(rows) == [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]
        assert rows == [[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]
        assert scaler.transform([[4.0, 15.0]]) == [[0.5, -0.5]]

    def test_training_rows_and_outputs_are_independent_of_fitted_statistics(self) -> None:
        """Editing a training row and a returned score must leave future scores unchanged."""
        rows = [[0.0], [1.0], [2.0]]
        scaler = Standardizer().fit(rows)
        result = scaler.transform(rows)
        rows[0][0] = 100.0
        result[0][0] = 200.0
        assert scaler.transform([[0.0]]) == [[-1.0]]

    @pytest.mark.parametrize("samples", [[[2.0]], [[2.0], [2.0]]])
    def test_single_sample_and_constant_feature_use_unit_scale(self, samples: list[list[float]]) -> None:
        """With no observed variation, a new value one unit above the mean scores one."""
        scaler = Standardizer().fit(samples)
        assert scaler.transform([[2.0], [3.0]]) == [[0.0], [1.0]]

    @given(exponent=st.integers(-250, 250))
    def test_standardization_removes_positive_rescaling(self, exponent: int) -> None:
        """Changing feature units scales the mean and standard deviation by the same factor."""
        scale = 10.0**exponent
        result = Standardizer().fit_transform([[0.0], [scale], [2 * scale]])
        assert [row[0] for row in result] == pytest.approx([-1.0, 0.0, 1.0])

    def test_requires_fit(self) -> None:
        """Require training statistics before transforming even a single value."""
        with pytest.raises(ValueError, match="not fitted"):
            Standardizer().transform([[0.0]])

    def test_failed_refit_preserves_statistics(self) -> None:
        """A ragged second dataset must leave the first mean and scale available for subsequent transforms."""
        scaler = Standardizer().fit([[0.0], [2.0], [4.0]])
        with pytest.raises(ValueError, match="columns"):
            scaler.fit([[0.0], [0.0, 1.0]])
        assert scaler.transform([[4.0]]) == [[1.0]]

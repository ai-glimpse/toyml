"""KMeans contracts, from input validation through numerical limits."""

import math
import random
import sys
from copy import deepcopy
from fractions import Fraction
from inspect import signature
from typing import Literal
from unittest.mock import patch

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from toyml.clustering import KMeans

type Initialization = Literal["random", "kmeans++"]

PROPERTY = settings(max_examples=50, derandomize=True, database=None, deadline=None)
SAMPLES = st.lists(st.tuples(st.integers(-5, 5), st.integers(-5, 5)), min_size=1, max_size=10).map(
    lambda rows: [[float(x), float(y)] for x, y in rows]
)


def fitted_state(model: KMeans) -> tuple[dict[int, list[float]], dict[int, list[int]], list[int], int]:
    """Snapshot the public fit result so a later mutation cannot change the reference."""
    return deepcopy((model.centroids_, model.clusters_, model.labels_, model.iter_))


class TestInputValidation:
    @pytest.mark.parametrize("k", [0, -1, True, 1.5])
    def test_rejects_invalid_cluster_counts(self, k: float) -> None:
        """Require a positive integer cluster count, excluding booleans."""
        with pytest.raises(ValueError, match="k"):
            KMeans(k=k).fit([[0.0]])  # ty: ignore[invalid-argument-type]

    def test_rejects_more_clusters_than_samples(self) -> None:
        """Reject k just above the sample count."""
        with pytest.raises(ValueError, match="k"):
            KMeans(k=3).fit([[0.0], [1.0]])

    @pytest.mark.parametrize("max_iter", [0, -1, True, 1.5])
    def test_rejects_invalid_iteration_limits(self, max_iter: float) -> None:
        """Require at least one whole iteration."""
        with pytest.raises(ValueError, match="max_iter"):
            KMeans(k=1, max_iter=max_iter).fit([[0.0]])  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize("tol", [-1.0, math.nan, math.inf, 10**400])
    def test_rejects_invalid_tolerances(self, tol: float) -> None:
        """Reject negative or unrepresentable stopping tolerances."""
        with pytest.raises(ValueError, match="tol"):
            KMeans(k=1, tol=tol).fit([[0.0]])

    @pytest.mark.parametrize("samples", [[], [[]], [[0.0], [1.0, 2.0]], [[math.nan]], [[math.inf]], [[10**400]]])
    def test_rejects_invalid_feature_matrices(self, samples: list[list[float]]) -> None:
        """Reject empty, ragged or nonfinite training data before initialization."""
        with pytest.raises(ValueError, match="sample|feature|dimension|numeric"):
            KMeans(k=1).fit(samples)

    @pytest.mark.parametrize(
        ("model", "message"),
        [
            (KMeans(k=1, random_seed=1.5), "random_seed"),  # ty: ignore[invalid-argument-type]
            (KMeans(k=1, centroids_init_method="other"), "initialization"),  # ty: ignore[invalid-argument-type]
            (KMeans(k=1, distance_metric="other"), "Distance metric"),  # ty: ignore[invalid-argument-type]
        ],
    )
    def test_fit_rejects_unsupported_configuration(self, model: KMeans, message: str) -> None:
        """Report which seed, initializer or distance setting is invalid."""
        with pytest.raises(ValueError, match=message):
            model.fit([[0.0]])

    @pytest.mark.parametrize("point", [[], [1.0, 2.0], [math.nan], [math.inf], [10**400]])
    def test_predict_validates_point(self, point: list[float]) -> None:
        """Require a finite query with the fitted feature dimension."""
        model = KMeans(k=1).fit([[0.0]])
        with pytest.raises(ValueError, match="dimension|numeric"):
            model.predict(point)

    def test_predict_rejects_unsupported_distance_metric_after_fit(self) -> None:
        """Recheck the active distance metric when predicting."""
        model = KMeans(k=1).fit([[0.0]])
        model.distance_metric = "other"  # ty: ignore[invalid-assignment]
        with pytest.raises(ValueError, match="Distance metric other not supported"):
            model.predict([0.0])


class TestInitialization:
    @PROPERTY
    @given(seed=st.integers(-(2**32), 2**32 - 1), k=st.integers(1, 8))
    def test_random_initialization_samples_without_replacement(self, k: int, seed: int) -> None:
        """Draw k distinct observations from a dataset with unique rows."""
        samples = [[float(i)] for i in range(8)]
        centroids = KMeans(k=k)._initial_centroids(samples, random.Random(seed))
        assert set(centroids) == set(range(k))
        assert len({tuple(point) for point in centroids.values()}) == k
        assert all(point in samples for point in centroids.values())

    @pytest.mark.parametrize("exponent", [-250, 0, 250])
    def test_kmeans_plus_plus_weights_follow_nearest_centroid(self, exponent: int) -> None:
        """Each draw uses squared distances to the nearest center selected so far."""
        scale = 10.0**exponent
        samples = [[x * scale] for x in (0, 2, 5, 9)]
        model = KMeans(k=3, centroids_init_method="kmeans++")
        rng = random.Random(0)
        draws = iter([samples[3], samples[2]])
        probabilities: list[list[float]] = []

        def choose(population: list[list[float]], weights: list[float]) -> list[list[float]]:
            total = math.fsum(weights)
            by_coordinate = {point[0]: weight / total for point, weight in zip(population, weights, strict=True)}
            probabilities.append([by_coordinate.get(point[0], 0.0) for point in samples])
            return [next(draws)]

        with patch.object(rng, "choice", return_value=samples[0]), patch.object(rng, "choices", choose):
            centers = model._initial_centroids(samples, rng)

        assert sorted(centers.values()) == [samples[0], samples[2], samples[3]]
        first, second = probabilities
        assert first == pytest.approx([0, 4 / 110, 25 / 110, 81 / 110], rel=1e-14, abs=0.0)
        assert second == pytest.approx([0, 4 / 20, 16 / 20, 0], rel=1e-14, abs=0.0)


class TestRandomness:
    @pytest.mark.parametrize("initialization", ["random", "kmeans++"])
    def test_same_seed_reproduces_fit(self, initialization: Initialization) -> None:
        """Refitting and fitting a fresh instance with the same seed reproduce the public result."""
        samples = [[0.0], [4.0], [5.0], [9.0]]
        model = KMeans(k=2, random_seed=42, centroids_init_method=initialization)
        expected = fitted_state(model.fit(samples))
        fresh = KMeans(k=2, random_seed=42, centroids_init_method=initialization)
        assert fitted_state(model.fit(samples)) == expected
        assert fitted_state(fresh.fit(samples)) == expected

    @pytest.mark.parametrize("initialization", ["random", "kmeans++"])
    def test_seed_can_change_partition(self, initialization: Initialization) -> None:
        """Different seeds can yield different partitions, even after ignoring label numbers."""
        samples = [[0.0], [4.0], [5.0], [9.0]]
        partitions = set()
        for seed in range(32):
            model = KMeans(k=2, random_seed=seed, centroids_init_method=initialization).fit(samples)
            partitions.add(frozenset(frozenset(members) for members in model.clusters_.values()))
            if len(partitions) > 1:
                break
        assert len(partitions) > 1

    @pytest.mark.parametrize("initialization", ["random", "kmeans++"])
    @pytest.mark.parametrize("seed", [None, -42])
    def test_fit_preserves_global_random_state(self, initialization: Initialization, seed: int | None) -> None:
        """Seeded and unseeded fits must leave the caller's global random stream untouched."""
        before = random.getstate()
        try:
            KMeans(k=2, centroids_init_method=initialization, random_seed=seed).fit([[0.0], [1.0], [4.0]])
            assert random.getstate() == before
        finally:
            random.setstate(before)

    def test_two_kmeans_instances_fit_independently(self) -> None:
        """Fitting or refitting one model leaves the other model's fitted result intact."""
        first = KMeans(k=1, random_seed=42).fit([[1.0], [3.0]])
        first_result = fitted_state(first)
        second = KMeans(k=2).fit([[10.0], [20.0]])
        assert fitted_state(first) == first_result
        second_result = fitted_state(second)
        first.fit([[-5.0]])
        assert fitted_state(second) == second_result


class TestLloydIteration:
    @pytest.mark.parametrize(("tol", "iterations"), [(1.0, 1), (0.99, 2)])
    def test_tolerance_applies_to_each_coordinate(self, tol: float, iterations: int) -> None:
        """Stop on equality with tol, comparing coordinates independently."""
        samples = [[0.0, 0.0], [2.0, 2.0]]
        model = KMeans(k=1, tol=tol)
        centers, actual = model._lloyd(samples, {0: samples[0]})
        assert centers == {0: [1.0, 1.0]}
        assert actual == iterations

    def test_lloyd_continues_while_any_centroid_coordinate_moves(self) -> None:
        """Only the second center's y coordinate moves at first, changing the next assignments."""
        samples = [[0.0, y] for y in (0.0, 2.0, 3.0, 8.0, 12.0)]
        centers, _ = KMeans(k=2, tol=0.0)._lloyd(samples, {0: samples[0], 1: samples[1]})
        assert centers == {0: [0.0, 5 / 3], 1: [0.0, 10.0]}

    @PROPERTY
    @given(values=st.lists(st.integers(-30, 30), min_size=1, max_size=15), requested_k=st.integers(1, 8))
    def test_one_iteration_returns_means_of_nearest_center_groups(self, values: list[int], requested_k: int) -> None:
        """One update must produce the independently computed means of the assigned samples."""
        # Each distinct starting center owns at least one sample, so relocation is unnecessary.
        initial = list(dict.fromkeys(values))[:requested_k]
        groups: list[list[int]] = [[] for _ in initial]
        for value in values:
            label = min(range(len(initial)), key=lambda i: (value - initial[i]) ** 2)
            groups[label].append(value)

        model = KMeans(k=len(initial), max_iter=1)
        centers, _ = model._lloyd([[float(v)] for v in values], {i: [float(v)] for i, v in enumerate(initial)})
        assert centers == {i: [sum(group) / len(group)] for i, group in enumerate(groups)}

    @PROPERTY
    @given(samples=SAMPLES, requested_k=st.integers(1, 10))
    def test_lloyd_step_does_not_increase_squared_error(self, samples: list[list[float]], requested_k: int) -> None:
        """A full assignment and centroid update cannot increase the nearest-centroid squared error."""
        model = KMeans(k=min(requested_k, len(samples)), max_iter=1)
        initial = dict(enumerate(samples[: model.k]))
        before = sum(min(math.dist(sample, c) ** 2 for c in initial.values()) for sample in samples)
        centers, _ = model._lloyd(samples, initial)
        after = sum(min(math.dist(sample, c) ** 2 for c in centers.values()) for sample in samples)
        assert after <= before + 1e-12

    def test_relocation_can_empty_a_singleton_cluster(self) -> None:
        """Move 16 to the empty cluster and give its former cluster the largest cluster's center."""
        samples = [[-1.0], [2.0], [16.0]]
        model = KMeans(k=3, max_iter=1)
        centers, _ = model._lloyd(samples, {0: [0.0], 1: [10.0], 2: [10.0]})
        assert centers == {0: [0.5], 1: [0.5], 2: [16.0]}

    def test_relocation_stops_at_zero_distance(self) -> None:
        """Move 9 and 3 into separate clusters, leaving both zero-distance samples in their original cluster."""
        samples = [[0.0], [0.0], [3.0], [9.0]]
        model = KMeans(k=4)
        centers = dict.fromkeys(range(4), samples[0])
        clusters = {0: [0, 1, 2, 3], 1: [], 2: [], 3: []}
        model._relocate_empty_clusters(samples, centers, clusters)
        assert sorted(clusters[0]) == [0, 1]
        assert sorted(clusters[label] for label in (1, 2, 3)) == [[], [2], [3]]


class TestFitAndPredict:
    @pytest.mark.parametrize("exponent", [-1000, 0, 1000])
    def test_equal_distances_to_distinct_centroids_choose_lowest_label(self, exponent: int) -> None:
        """Both squared distances are 85 times the same scale squared, so label zero wins."""
        samples = [[math.ldexp(x, exponent), math.ldexp(y, exponent)] for x, y in [(2, 9), (6, 7)]]
        model = KMeans(k=2, random_seed=1).fit(samples)
        assert model.predict([0.0, 0.0]) == 0

    def test_one_ulp_difference_still_selects_the_nearer_centroid(self) -> None:
        """The nearer centroid wins even when the distances are adjacent floats."""
        model = KMeans(k=2)
        model.centroids_ = {0: [math.nextafter(1.0, math.inf)], 1: [1.0]}
        assert model.predict([0.0]) == 1

    def test_distances_rounding_to_the_same_float_choose_lowest_label(self) -> None:
        """An exact squared-distance difference of one can disappear in the computed norms."""
        model = KMeans(k=2)
        model.centroids_ = {0: [1e8, 1.0], 1: [1e8, 0.0]}
        assert model.predict([0.0, 0.0]) == 0

    @pytest.mark.parametrize("initialization", ["random", "kmeans++"])
    def test_fit_separates_two_groups(self, initialization: Initialization, simple_dataset: list[list[float]]) -> None:
        """Both initializers must separate the two known groups and classify a nearby query."""
        model = KMeans(2, 10, 1e-5, initialization, random_seed=0).fit(simple_dataset)
        left, right = model.predict(simple_dataset[0]), model.predict(simple_dataset[-1])
        assert model.clusters_ == {left: [0, 1, 2], right: [3, 4, 5]}
        assert model.predict([0.0, 0.0]) == left

    @PROPERTY
    @given(samples=SAMPLES)
    def test_one_cluster_is_the_dataset_mean(self, samples: list[list[float]]) -> None:
        """With k=1, every observation contributes to the only centroid."""
        model = KMeans(k=1, random_seed=0).fit(samples)
        assert model.centroids_[0] == [sum(column) / len(samples) for column in zip(*samples, strict=True)]
        assert model.clusters_ == {0: list(range(len(samples)))}

    @pytest.mark.parametrize("initialization", ["random", "kmeans++"])
    @PROPERTY
    @given(samples=SAMPLES, requested_k=st.integers(1, 10), seed=st.integers(0, 1000))
    def test_duplicate_samples_produce_a_valid_partition(
        self, initialization: Initialization, samples: list[list[float]], requested_k: int, seed: int
    ) -> None:
        """Every index has one consistent label even when several observations coincide."""
        k = min(requested_k, len(samples))
        model = KMeans(k=k, centroids_init_method=initialization, random_seed=seed, max_iter=20).fit(samples)
        assert set(model.centroids_) == set(model.clusters_) == set(range(k))
        assert 1 <= model.iter_ <= model.max_iter
        assert all(len(center) == 2 and all(math.isfinite(x) for x in center) for center in model.centroids_.values())
        assert sorted(i for members in model.clusters_.values() for i in members) == list(range(len(samples)))
        assert model.labels_ == [model.predict(sample) for sample in samples]
        for label, members in model.clusters_.items():
            assert members == [i for i, assigned in enumerate(model.labels_) if assigned == label]

    @pytest.mark.parametrize("initialization", ["random", "kmeans++"])
    @PROPERTY
    @given(value=st.integers(-10, 10), count=st.integers(2, 8))
    def test_identical_samples_allow_duplicate_centers(
        self, initialization: Initialization, value: int, count: int
    ) -> None:
        """Coincident centroids keep all k slots and ties go to label zero."""
        samples = [[float(value)]] * count
        model = KMeans(k=count, centroids_init_method=initialization, random_seed=0).fit(samples)
        assert len(model.centroids_) == count
        assert all(center == [float(value)] for center in model.centroids_.values())
        assert model.labels_ == [0] * count

    @pytest.mark.parametrize("initialization", ["random", "kmeans++"])
    def test_degenerate_fit_reports_fewer_nonempty_clusters(
        self, initialization: Initialization, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Report a warning when fewer than k distinct groups remain populated."""
        model = KMeans(k=3, centroids_init_method=initialization, random_seed=0).fit([[0.0], [0.0], [10.0], [10.0]])
        assert len(model.centroids_) == 3
        assert len(set(model.labels_)) == 2
        assert "2 non-empty clusters" in caplog.text

    def test_iteration_limit_reassigns_sample_to_final_centers(self) -> None:
        """The sample at 3 switches from label 1 to label 0 after the only centroid update."""
        samples = [[0.0], [3.0], [9.0], [2.0]]
        model = KMeans(k=2, max_iter=1)
        with patch.object(model, "_initial_centroids", return_value={0: [2.0], 1: [3.0]}):
            model.fit(samples)
        assert model.iter_ == 1
        assert model.centroids_ == {0: [1.0], 1: [6.0]}
        assert model.labels_[1] == 0  # 3 is closer to 1 than to 6.
        assert 1 in model.clusters_[0]


class TestModelLifecycle:
    def test_constructor_excludes_training_state(self) -> None:
        """Fitted attributes start empty and cannot be supplied to the constructor."""
        model = KMeans(k=2)
        assert {"centroids_", "clusters_", "labels_", "iter_"}.isdisjoint(signature(KMeans).parameters)
        assert model.centroids_ == {}
        assert model.clusters_ == {}
        assert model.labels_ == []
        assert model.iter_ == 0

    def test_predict_before_fit_fails(self) -> None:
        """Prediction needs a fitted model even when the query itself is valid."""
        with pytest.raises(ValueError, match="not.*fitted"):
            KMeans(k=2).predict([0.0])

    @PROPERTY
    @given(first=SAMPLES, second=SAMPLES)
    def test_refit_replaces_state_and_preserves_configuration(
        self, first: list[list[float]], second: list[list[float]]
    ) -> None:
        """A successful refit keeps its configuration and replaces all previous fitted attributes."""
        k = min(2, len(first), len(second))
        model = KMeans(k=k, centroids_init_method="kmeans++", random_seed=42, max_iter=5, tol=0.25)
        fresh = KMeans(k=k, centroids_init_method="kmeans++", random_seed=42, max_iter=5, tol=0.25)
        model.fit(first).fit(second)
        for name in signature(KMeans).parameters:
            assert getattr(model, name) == getattr(fresh, name)
        assert fitted_state(model) == fitted_state(fresh.fit(second))

    @pytest.mark.parametrize("samples", [[[math.nan], [0.0]], [[-1.7e308], [1.7e308]]])
    def test_failed_refit_preserves_previous_result(self, samples: list[list[float]]) -> None:
        """Validation and numerical failures must leave the previous fit usable."""
        model = KMeans(k=2, random_seed=0).fit([[0.0], [10.0]])
        previous = fitted_state(model)
        with pytest.raises(ValueError, match="numeric|floating-point range"):
            model.fit(samples)
        assert fitted_state(model) == previous
        assert model.predict([10.0]) == model.labels_[1]

    @PROPERTY
    @given(samples=SAMPLES)
    @example(samples=[[0.0], [0.1], [4.0], [4.1]])
    def test_fit_owns_its_result(self, samples: list[list[float]]) -> None:
        """Training leaves caller-owned rows intact and later input mutations cannot change the fit."""
        original = [row.copy() for row in samples]
        model = KMeans(k=min(2, len(samples)), random_seed=42).fit(samples)
        assert samples == original
        fitted = fitted_state(model)
        samples[0][0] += 100
        samples.clear()
        assert fitted_state(model) == fitted

    def test_fit_predict_returns_independent_labels(self) -> None:
        """Mutating the returned list must leave training labels intact."""
        model = KMeans(k=1)
        labels = model.fit_predict([[1.0], [2.0]])
        assert labels == [0, 0]
        labels.clear()
        assert model.labels_ == [0, 0]


class TestNumericalStability:
    @pytest.mark.parametrize(
        ("first", "second"),
        [([-1e308], [1e308]), ([-(10**308)], [10**308]), ([0.0, 0.0], [1.5e308, 1.5e308])],
    )
    def test_fit_rejects_unrepresentable_distances(self, first: list[float], second: list[float]) -> None:
        """Report overflow in either a coordinate difference or the final Euclidean norm."""
        with pytest.raises(ValueError, match="floating-point range"):
            KMeans(k=2, random_seed=0).fit([first, second])

    @given(values=st.lists(st.integers(-100, 100), min_size=1, max_size=8), exponent=st.integers(-250, 250))
    def test_distance_preserves_small_and_large_scales(self, values: list[int], exponent: int) -> None:
        """At every scale, the norm is symmetric and agrees with the reference magnitude."""
        point = [value * 10.0**exponent for value in values]
        origin = [0.0] * len(point)
        actual = KMeans._euclidean_distance(origin, point)
        expected = math.dist(origin, point)
        assert actual == pytest.approx(expected, rel=1e-14, abs=0.0)
        assert KMeans._euclidean_distance(point, origin) == actual
        assert KMeans._euclidean_distance(point, point) == 0.0

    @given(pairs=st.lists(st.tuples(st.floats(-1e150, 1e150), st.floats(-1e150, 1e150)), min_size=1, max_size=8))
    def test_distance_matches_reference_for_arbitrary_points(self, pairs: list[tuple[float, float]]) -> None:
        """Allow rounding in subtraction and the norm, using exact squared distance as the reference."""
        first = [x for x, _ in pairs]
        second = [y for _, y in pairs]
        distance = KMeans._euclidean_distance(first, second)
        exact_squared = sum((Fraction(x) - Fraction(y)) ** 2 for x, y in pairs)
        margin = 2 * Fraction(math.ulp(distance))
        rounded = Fraction(distance)
        assert max(0, rounded - margin) ** 2 <= exact_squared <= (rounded + margin) ** 2

    @pytest.mark.parametrize(
        ("column", "expected"),
        [
            ([2**53 + 1, -(2**53)], 0.5),
            ([2**53 + 1, -float(2**53), 0.5], 0.5),
            ([8.5e307] * 3 + [-8.5e307] * 3 + [1e-100], 1e-100 / 7),
        ],
    )
    def test_centroid_mean_preserves_cancellation_remainder(self, column: list[float], expected: float) -> None:
        """The single centroid retains the small remainder after large coordinates cancel."""
        model = KMeans(k=1, random_seed=0).fit([[value] for value in column])
        assert model.centroids_[0] == [expected]

    @pytest.mark.parametrize("value", [1e308, -1e308, sys.float_info.max, -sys.float_info.max, 5e-324, -5e-324])
    def test_duplicate_extreme_coordinates_keep_their_mean(self, value: float) -> None:
        """A constant column must keep its value even near the limits of float."""
        model = KMeans(k=2, centroids_init_method="kmeans++", random_seed=0).fit([[value]] * 3)
        assert all(center == [value] for center in model.centroids_.values())

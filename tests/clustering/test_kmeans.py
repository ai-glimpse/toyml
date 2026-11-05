import math
import sys
from dataclasses import asdict
from typing import Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from toyml.clustering import KMeans


class TestKMeans:
    """Test the k-means algorithm completely."""

    @pytest.mark.parametrize("centroids_init_method", ["random", "kmeans++"])
    def test_fit(
        self,
        centroids_init_method: Literal["random", "kmeans++"],
        simple_dataset: list[list[float]],
    ) -> None:
        k = 2
        max_iter = 10
        dataset = simple_dataset

        kmeans = KMeans(k, max_iter, 1e-5, centroids_init_method)
        assert len(kmeans.clusters_) == 0
        assert len(kmeans.centroids_) == 0
        kmeans.fit(dataset)
        assert len(kmeans.clusters_) == 2
        assert len(kmeans.centroids_) == 2

        # check clusters
        first_sample_cluster, last_sample_cluster = kmeans.predict(dataset[0]), kmeans.predict(dataset[-1])
        assert kmeans.clusters_[first_sample_cluster] == [0, 1, 2]
        assert kmeans.clusters_[last_sample_cluster] == [3, 4, 5]

        # check prediction
        x_sample = [0.0] * len(dataset[0])
        assert kmeans.predict(x_sample) == first_sample_cluster


PROPERTY = settings(max_examples=50, derandomize=True, database=None, deadline=None)
DUPLICATED_DATA = st.lists(st.tuples(st.integers(-5, 5), st.integers(-5, 5)), min_size=1, max_size=10).map(
    lambda rows: [[float(x), float(y)] for x, y in rows]
)


@pytest.mark.parametrize("initialization", ["random", "kmeans++"])
@PROPERTY
@given(samples=DUPLICATED_DATA, requested_k=st.integers(1, 10), seed=st.integers(0, 1000))
def test_duplicate_samples_keep_valid_centers_and_reproducible_labels(
    initialization: Literal["random", "kmeans++"], samples: list[list[float]], requested_k: int, seed: int
) -> None:
    k = min(requested_k, len(samples))
    original = [row.copy() for row in samples]
    model = KMeans(k=k, centroids_init_method=initialization, random_seed=seed, max_iter=20).fit(samples)
    expected = asdict(model)
    assert samples == original
    assert len(model.centroids_) == k
    assert all(len(center) == 2 and all(math.isfinite(x) for x in center) for center in model.centroids_.values())
    assert sorted(i for members in model.clusters_.values() for i in members) == list(range(len(samples)))
    assert model.labels_ == [model.predict(sample) for sample in samples]
    assert len(set(model.labels_)) <= min(k, len(set(map(tuple, samples))))
    model.fit(samples)
    assert asdict(model) == expected


@pytest.mark.parametrize("initialization", ["random", "kmeans++"])
@PROPERTY
@given(value=st.integers(-10, 10), count=st.integers(2, 8))
def test_identical_samples_allow_duplicate_centers(
    initialization: Literal["random", "kmeans++"], value: int, count: int
) -> None:
    samples = [[float(value)]] * count
    model = KMeans(k=count, centroids_init_method=initialization, random_seed=0).fit(samples)
    assert len(model.centroids_) == count
    assert all(center == [float(value)] for center in model.centroids_.values())
    assert len(set(model.labels_)) == 1
    assert model.labels_ == [model.predict(sample) for sample in samples]


@PROPERTY
@given(offset=st.integers(-10, 10), gap=st.integers(1, 10))
def test_empty_cluster_is_relocated_to_farthest_sample(offset: int, gap: int) -> None:
    samples = [[float(offset)], [float(offset)], [float(offset + gap)]]
    model = KMeans(k=2, random_seed=0)
    model.centroids_ = {0: [float(offset)], 1: [float(offset)]}
    model._iter_step(samples)
    assert model.clusters_ == {0: [0, 1], 1: [2]}
    assert model.centroids_ == {0: [float(offset)], 1: [float(offset + gap)]}
    assert model.labels_ == [0, 0, 1]


@PROPERTY
@given(offset=st.integers(-10, 10), count=st.integers(3, 8))
def test_multiple_empty_clusters_do_not_empty_the_donor(offset: int, count: int) -> None:
    samples = [[float(offset + i)] for i in range(count)]
    model = KMeans(k=count, random_seed=0)
    model.centroids_ = {i: [float(offset)] for i in range(count)}
    model._iter_step(samples)
    assert all(model.clusters_.values())
    assert sorted(i for members in model.clusters_.values() for i in members) == list(range(count))
    assert len({tuple(center) for center in model.centroids_.values()}) == count


@PROPERTY
@given(exponent=st.integers(-250, 250), seed=st.integers(0, 1000))
def test_kmeans_plus_plus_sampling_handles_distance_scales(exponent: int, seed: int) -> None:
    scale = 10.0**exponent
    samples = [[0.0], [scale], [2 * scale], [3 * scale]]
    model = KMeans(k=4, centroids_init_method="kmeans++", random_seed=seed).fit(samples)
    assert len(set(model.labels_)) == 4
    assert {tuple(center) for center in model.centroids_.values()} == set(map(tuple, samples))


@PROPERTY
@given(samples=DUPLICATED_DATA, requested_k=st.integers(1, 10))
def test_lloyd_step_does_not_increase_squared_error(samples: list[list[float]], requested_k: int) -> None:
    model = KMeans(k=min(requested_k, len(samples)), random_seed=0)
    model.centroids_ = {i: samples[0].copy() for i in range(model.k)}
    before = sum(min(math.dist(sample, c) ** 2 for c in model.centroids_.values()) for sample in samples)
    model._iter_step(samples)
    after = sum(min(math.dist(sample, c) ** 2 for c in model.centroids_.values()) for sample in samples)
    assert after <= before + 1e-12


@pytest.mark.parametrize("initialization", ["random", "kmeans++"])
@given(count=st.integers(1, 8), extra=st.integers(1, 8))
def test_rejects_more_clusters_than_samples(
    initialization: Literal["random", "kmeans++"], count: int, extra: int
) -> None:
    with pytest.raises(ValueError, match="k"):
        KMeans(k=count + extra, centroids_init_method=initialization).fit([[float(i)] for i in range(count)])


@pytest.mark.parametrize("initialization", ["random", "kmeans++"])
def test_degenerate_fit_reports_fewer_nonempty_clusters(
    initialization: Literal["random", "kmeans++"], caplog: pytest.LogCaptureFixture
) -> None:
    model = KMeans(k=3, centroids_init_method=initialization, random_seed=0).fit([[0.0], [0.0], [10.0], [10.0]])
    assert len(model.centroids_) == 3
    assert len(set(model.labels_)) == 2
    assert "2 non-empty clusters" in caplog.text


@PROPERTY
@given(samples=DUPLICATED_DATA, requested_k=st.integers(1, 10))
def test_iteration_limit_keeps_labels_consistent_with_final_centers(
    samples: list[list[float]], requested_k: int
) -> None:
    model = KMeans(k=min(requested_k, len(samples)), random_seed=0, max_iter=1).fit(samples)
    assert model.labels_ == [model.predict(point) for point in samples]
    for label, members in model.clusters_.items():
        assert members == [i for i, assigned in enumerate(model.labels_) if assigned == label]


@pytest.mark.parametrize("samples", [[], [[]], [[0.0], [1.0, 2.0]], [[math.nan]], [[math.inf]], [[10**400]]])
def test_rejects_invalid_feature_matrices(samples: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="sample|feature|dimension|numeric"):
        KMeans(k=1).fit(samples)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_rejects_invalid_cluster_counts(k: float) -> None:
    with pytest.raises(ValueError, match="k"):
        KMeans(k=k).fit([[0.0]])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("max_iter", [0, -1, True, 1.5])
def test_rejects_invalid_iteration_limits(max_iter: float) -> None:
    with pytest.raises(ValueError, match="max_iter"):
        KMeans(k=1, max_iter=max_iter).fit([[0.0]])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("tol", [-1.0, math.nan, math.inf])
def test_rejects_invalid_tolerances(tol: float) -> None:
    with pytest.raises(ValueError, match="tol"):
        KMeans(k=1, tol=tol).fit([[0.0]])


def test_rejects_unrepresentable_distances_explicitly() -> None:
    with pytest.raises(ValueError, match="floating-point range"):
        KMeans(k=2, centroids_init_method="kmeans++").fit([[-1.7e308], [1.7e308]])


@PROPERTY
@given(exponent=st.integers(-250, 250))
def test_sampling_probabilities_follow_squared_distance(exponent: int) -> None:
    model = KMeans(k=2, random_seed=0)
    samples = [[0.0], [2.0], [4.0]]
    scale = 10.0**exponent

    def choose(population: list[list[float]], weights: list[float]) -> list[list[float]]:
        total = math.fsum(weights)
        assert [weight / total for weight in weights] == pytest.approx([0.0, 0.2, 0.8], abs=0.0)
        return [population[1]]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model.random_state, "choices", choose)
        assert model._sample_next_centroid(samples, [0.0, 2 * scale, 4 * scale]) == samples[1]


@pytest.mark.parametrize("value", [1e308, -1e308, sys.float_info.max, -sys.float_info.max, 5e-324, -5e-324])
def test_duplicate_extreme_coordinates_keep_their_mean(value: float) -> None:
    model = KMeans(k=2, centroids_init_method="kmeans++", random_seed=0).fit([[value]] * 3)
    assert all(center == [value] for center in model.centroids_.values())


@given(values=st.lists(st.integers(-100, 100), min_size=1, max_size=10), exponent=st.integers(-250, 250))
def test_centroid_mean_respects_coordinate_scale(values: list[int], exponent: int) -> None:
    scale = 10.0**exponent
    column = tuple(value * scale for value in values)
    expected = (sum(values) / len(values)) * scale
    assert KMeans._mean(column) == pytest.approx(expected, rel=1e-14, abs=1e-14 * scale)

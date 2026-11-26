import math
import random
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, fields
from random import Random
from typing import Literal

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from toyml.ensemble.isolation_forest import IsolationForest, IsolationTree

TRAINING_DATA = st.lists(st.integers(1, 30), min_size=4, max_size=8, unique=True).map(
    lambda xs: [[float(x), float(x % 5)] for x in xs]
)
PROPERTY = settings(max_examples=30, deadline=None, derandomize=True, database=None)


class TestIsolationTree:
    @pytest.mark.parametrize("max_height", [-1, -2])
    def test_invalid_max_height_raise_error(self, max_height: int) -> None:
        with pytest.raises(ValueError, match="The max height of"):
            _ = IsolationTree(max_height=max_height)

    @pytest.mark.parametrize("max_height", [0, 1, 2])
    def test_single_sample_build_leaf_node(self, max_height: int) -> None:
        sut = IsolationTree(max_height=max_height)
        samples = [[0.0, 1.0]]

        sut.fit(samples)

        assert sut.sample_size_ == 1
        assert sut.feature_num_ == 2
        assert sut.is_external_node() is True

    @pytest.mark.parametrize("max_height", [0, 1, 2])
    def test_two_same_samples_build_leaf_node(self, max_height: int) -> None:
        sut = IsolationTree(max_height=max_height)
        samples = [[0.0, 1.0], [0.0, 1.0]]

        sut.fit(samples)

        assert sut.sample_size_ == 2
        assert sut.feature_num_ == 2
        assert sut.is_external_node() is True

    @pytest.mark.parametrize("max_height", [1, 2, 3])
    def test_two_different_samples_build_three_node_itree(self, max_height: int) -> None:
        sut = IsolationTree(max_height=max_height)
        samples = [[0.0, 1.0], [0.0, 2.0]]

        sut.fit(samples)

        assert sut.sample_size_ == 2
        assert sut.feature_num_ == 2
        assert sut.is_external_node() is False
        assert sut.left_ is not None
        assert sut.left_.is_external_node() is True
        assert sut.right_ is not None
        assert sut.right_.is_external_node() is True

    @pytest.mark.parametrize("max_height", [0, 1, 2])
    def test_leaf_node_path_length(self, max_height: int) -> None:
        sut = IsolationTree(max_height=max_height)
        samples = [[0.0, 1.0]]

        sut.fit(samples)

        assert sut.get_sample_path_length(samples[0]) == 0

    @pytest.mark.parametrize("max_height", [1, 2, 3])
    def test_itree_node_path_length(self, max_height: int) -> None:
        sut = IsolationTree(max_height=max_height)
        samples = [[0.0, 1.0], [0.0, 2.0]]

        sut.fit(samples)

        assert sut.get_sample_path_length(samples[0]) == 1
        assert sut.get_sample_path_length(samples[1]) == 1


class TestIsolationForest:
    def test_full_sample_trees_have_different_splits(self) -> None:
        samples = [[float(i), float(i % 3)] for i in range(12)]
        model = IsolationForest(n_itree=5, max_samples=len(samples), random_seed=0).fit(samples)
        splits = {(tree.split_at_, tree.split_value_) for tree in model.itrees_}
        assert len(splits) > 1

    @pytest.fixture
    def simple_dataset(self) -> list[list[float]]:
        return [[-1.1], [0.3], [0.5], [100.0]]

    @pytest.mark.parametrize(
        ("n_itree", "max_samples"),
        [
            (5, 3),
            (8, 4),
            (10, 6),
        ],
    )
    def test_itree_build(
        self,
        simple_dataset: list[list[float]],
        n_itree: int,
        max_samples: int,
    ) -> None:
        sut = IsolationForest(n_itree=n_itree, max_samples=max_samples)

        sut.fit(simple_dataset)

        assert len(sut.itrees_) == n_itree

    @pytest.mark.parametrize(
        "n_itree",
        [
            5,
            8,
            10,
        ],
    )
    def test_anomaly_predict(
        self,
        simple_dataset: list[list[float]],
        n_itree: int,
    ) -> None:
        sut = IsolationForest(n_itree=n_itree)

        labels = sut.fit_predict(simple_dataset)

        assert all(label in (1, -1) for label in labels) is True
        assert labels[-1] == -1

    @pytest.mark.parametrize(
        ("n_itree", "max_samples"),
        [
            (5, 4),
            (10, 4),
        ],
    )
    def test_anomaly_score_property(
        self,
        simple_dataset: list[list[float]],
        n_itree: int,
        max_samples: int,
    ) -> None:
        sut = IsolationForest(n_itree=n_itree, max_samples=max_samples)

        sut.fit(simple_dataset)
        scores = [sut.score(sample) for sample in simple_dataset]

        assert math.isclose(max(scores), scores[-1])
        assert all(0 <= score <= 1 for score in scores) is True


@pytest.mark.parametrize("max_samples", [None, 6])
@given(first_count=st.integers(2, 5), second_count=st.integers(6, 10))
def test_refit_recomputes_sample_count_without_changing_configuration(
    max_samples: int | None, first_count: int, second_count: int
) -> None:
    model = IsolationForest(n_itree=3, max_samples=max_samples, random_seed=0)
    model.fit([[float(i)] for i in range(first_count)])
    model.fit([[float(i)] for i in range(second_count)])
    expected = second_count if max_samples is None else max_samples
    assert model.max_samples == max_samples
    assert model.max_samples_ == expected
    assert all(tree.sample_size_ == expected for tree in model.itrees_)


@given(value=st.integers(-20, 20), count=st.integers(1, 5))
def test_tree_refit_to_leaf_discards_previous_splits(value: int, count: int) -> None:
    model = IsolationTree(max_height=4, random_seed=0).fit([[0.0], [1.0], [2.0], [3.0]])
    model.fit([[float(value)]] * count)
    assert model.is_external_node()
    assert model.left_ is None
    assert model.right_ is None
    assert model.split_at_ is None
    assert model.split_value_ is None


Model = IsolationForest | IsolationTree
ModelKind = Literal["isolation_forest", "isolation_tree"]
MODEL_KINDS = ("isolation_forest", "isolation_tree")
SEEDS = st.integers(0, 2**32 - 1)
DATASETS = st.lists(st.tuples(st.integers(-20, 20), st.integers(-20, 20)), min_size=4, max_size=8, unique=True).map(
    lambda rows: [[float(x), float(y)] for x, y in rows]
)
PROPERTY_SETTINGS = settings(max_examples=50, derandomize=True, database=None, deadline=None)


@contextmanager
def random_environment(entropy_seed: int) -> Generator[None]:
    """Make implicit seeding replayable without changing explicit seeds."""
    # Hypothesis must replay a failing case while shrinking it; real OS entropy would make that flaky.
    entropy = Random(entropy_seed)
    global_state = random.getstate()
    original_seed = random.seed

    def make_random(seed: int | None = None) -> Random:
        return Random(entropy.getrandbits(64) if seed is None else seed)

    def seed_global(seed: int | None = None) -> None:
        original_seed(entropy.getrandbits(64) if seed is None else seed)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(random, "Random", make_random)
        patch.setattr(random, "seed", seed_global)
        try:
            yield
        finally:
            random.setstate(global_state)


def make_model(kind: ModelKind, seed: int | None, *, max_samples: int | None = None) -> Model:
    if kind == "isolation_forest":
        return IsolationForest(n_itree=3, max_samples=max_samples, random_seed=seed)
    return IsolationTree(max_height=4, random_seed=seed)


def fitted_result(model: Model, samples: list[list[float]]) -> object:
    model.fit(samples)
    # Dataclass fields include fitted structures, not the identity of the RNG object.
    result = asdict(model)
    if isinstance(model, IsolationForest):
        result["scores"] = [model.score(sample) for sample in samples]
        result["predictions"] = [model.predict(sample) for sample in samples]
    else:
        result["path_lengths"] = [model.get_sample_path_length(sample) for sample in samples]
    return result


@pytest.mark.parametrize("kind", MODEL_KINDS)
@PROPERTY_SETTINGS
@given(samples=DATASETS, model_seed=SEEDS, entropy_seed=SEEDS)
def test_fixed_seed_reproduces_fresh_and_repeated_fits(
    kind: ModelKind, samples: list[list[float]], model_seed: int, entropy_seed: int
) -> None:
    with random_environment(entropy_seed):
        first = make_model(kind, model_seed)
        expected = fitted_result(first, samples)
    with random_environment(entropy_seed + 1):
        note("Comparing fresh instances with the same explicit seed and different ambient entropy")
        assert fitted_result(make_model(kind, model_seed), samples) == expected
        note("Comparing a repeated fit on the original instance")
        assert fitted_result(first, samples) == expected


@pytest.mark.parametrize("kind", MODEL_KINDS)
@pytest.mark.parametrize("same_seed", [True, False])
@PROPERTY_SETTINGS
@given(
    samples=DATASETS,
    other_samples=DATASETS,
    model_seed=st.one_of(st.none(), SEEDS),
    other_kind=st.sampled_from(MODEL_KINDS),
    entropy_seed=SEEDS,
    other_fits=st.integers(1, 3),
)
def test_model_instances_do_not_affect_each_other(
    kind: ModelKind,
    same_seed: bool,
    samples: list[list[float]],
    other_samples: list[list[float]],
    model_seed: int | None,
    other_kind: ModelKind,
    entropy_seed: int,
    other_fits: int,
) -> None:
    with random_environment(entropy_seed):
        expected = fitted_result(make_model(kind, model_seed), samples)
    with random_environment(entropy_seed):
        target = make_model(kind, model_seed)
        other_seed = model_seed if same_seed else (0 if model_seed is None else model_seed + 1)
        # Keep the target's entropy identical even when it draws a new seed at fit time.
        with random_environment(entropy_seed + 1):
            for _ in range(other_fits):
                fitted_result(make_model(other_kind, other_seed), other_samples)
        assert fitted_result(target, samples) == expected


@pytest.mark.parametrize("kind", MODEL_KINDS)
@PROPERTY_SETTINGS
@given(samples=DATASETS, model_seed=st.one_of(st.none(), SEEDS), entropy_seed=SEEDS)
def test_model_does_not_change_global_random_state(
    kind: ModelKind, samples: list[list[float]], model_seed: int | None, entropy_seed: int
) -> None:
    with random_environment(entropy_seed):
        before = random.getstate()
        fitted_result(make_model(kind, model_seed), samples)
        assert random.getstate() == before


@pytest.mark.parametrize("kind", MODEL_KINDS)
@PROPERTY
@given(first=TRAINING_DATA, second=TRAINING_DATA)
def test_isolation_model_refit_preserves_configuration_and_replaces_state(
    kind: ModelKind, first: list[list[float]], second: list[list[float]]
) -> None:
    model = make_model(kind, seed=0, max_samples=6)
    fresh = make_model(kind, seed=0, max_samples=6)
    model.fit(first)
    model.fit(second)
    for field in fields(model):
        if field.init:
            assert getattr(model, field.name) == getattr(fresh, field.name)
    fresh.fit(second)
    assert asdict(model) == asdict(fresh)


@pytest.mark.parametrize("kind", MODEL_KINDS)
@PROPERTY
@given(data=TRAINING_DATA)
def test_isolation_model_fit_owns_its_data(kind: ModelKind, data: list[list[float]]) -> None:
    original = [row.copy() for row in data]
    model = make_model(kind, seed=0, max_samples=6)
    model.fit(data)
    assert data == original
    trained = asdict(model)
    data[0][0] += 100
    data.clear()
    assert asdict(model) == trained


@pytest.mark.parametrize(
    "model", [IsolationForest(n_itree=3, max_samples=6, random_seed=0), IsolationTree(max_height=4, random_seed=0)]
)
def test_constructor_excludes_training_state(model: IsolationForest | IsolationTree) -> None:
    assert all(not f.init for f in fields(model) if f.name.endswith("_") or f.name.startswith("_"))


@given(value=st.integers(-10, 10))
def test_unfitted_forest_inference_fails(value: int) -> None:
    with pytest.raises(ValueError, match="not.*fitted"):
        IsolationForest().predict([float(value)])


@given(value=st.integers(-10, 10))
def test_unfitted_tree_inference_fails(value: int) -> None:
    with pytest.raises(ValueError, match="not.*fitted"):
        IsolationTree(max_height=2).get_sample_path_length([float(value)])


def test_fit_preserves_input() -> None:
    samples = [[0.0], [0.1], [4.0], [4.1]]
    original = [row.copy() for row in samples]
    IsolationForest(n_itree=5, random_seed=42).fit(samples)
    assert samples == original


@pytest.mark.xfail(
    strict=True,
    raises=ZeroDivisionError,
    reason="IsolationForest accepts one sample and divides by zero during scoring",
)
def test_isolation_forest_rejects_single_sample() -> None:
    with pytest.raises(ValueError, match="at least two samples"):
        IsolationForest(random_seed=42).fit_predict([[1.0]])

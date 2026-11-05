"""Fixed seeds reproduce fits, and model instances own independent random streams."""

import random
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict
from random import Random
from typing import Literal

import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from toyml.clustering import KMeans
from toyml.ensemble.isolation_forest import IsolationForest, IsolationTree

Model = KMeans | IsolationForest | IsolationTree
ModelKind = Literal["kmeans", "kmeans++", "isolation_forest", "isolation_tree"]
MODEL_KINDS = ("kmeans", "kmeans++", "isolation_forest", "isolation_tree")
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


def make_model(kind: ModelKind, seed: int | None) -> Model:
    if kind == "kmeans":
        return KMeans(k=2, random_seed=seed)
    if kind == "kmeans++":
        return KMeans(k=2, centroids_init_method="kmeans++", random_seed=seed)
    if kind == "isolation_forest":
        return IsolationForest(n_itree=3, random_seed=seed)
    return IsolationTree(max_height=4, random_seed=seed)


def fitted_result(model: Model, samples: list[list[float]]) -> object:
    model.fit(samples)
    # Dataclass fields include fitted structures, not the identity of the RNG object.
    result = asdict(model)
    if isinstance(model, IsolationForest):
        result["scores"] = [model.score(sample) for sample in samples]
        result["predictions"] = [model.predict(sample) for sample in samples]
    elif isinstance(model, IsolationTree):
        result["path_lengths"] = [model.get_sample_path_length(sample) for sample in samples]
    else:
        result["predictions"] = [model.predict(sample) for sample in samples]
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

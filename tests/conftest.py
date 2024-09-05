from __future__ import annotations

import pytest


@pytest.fixture
def simple_dataset() -> list[list[float]]:
    dataset: list[list[float]] = [[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [10.0, 1.0], [10.0, 2.0], [11.0, 1.0]]
    return dataset

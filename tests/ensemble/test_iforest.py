import pytest

from toyml.ensemble.iforest import IsolationTree


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

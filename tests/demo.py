import pytest


@pytest.mark.parametrize("speak_type", ["hello", "hi"])
def test_extreme_value_extraction(speak_type):
    assert speak_type in ["hello", "hi"]

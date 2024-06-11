import pytest

from toyml.classification.knn import KNeighborsClassifier


class TestKNeighborsClassifier:
    def test_errors(self):
        # Test bad distribution type
        with pytest.raises(TypeError, match=r"invalid type.*'dataset' argument"):
            KNeighborsClassifier(dataset=1.0, labels=[0, 0, 1, 1], k=3)  # type: ignore[arg-type]

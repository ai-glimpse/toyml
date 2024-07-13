from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Vector:
    """
    Represents a vector.

    Examples:
        >>> v = Vector([1, 2, 3])
        >>> print(v)
        Vector[1, 2, 3]
        >>> for i in v:
        ...     print(i)
        1
        2
        3
    """

    _data: list[float]

    # def mean(self) -> float:
    #     pass
    #
    # def std(self) -> float:
    #     pass

    def __repr__(self) -> str:
        return f"Vector{self._data}"

    def __len__(self) -> int:
        return len(self._data)

    def __add__(self, other: Vector) -> Vector:
        self.__check_length(other)
        return Vector([self._data[i] + other._data[i] for i in range(len(self._data))])

    def __sub__(self, other: Vector) -> Vector:
        self.__check_length(other)
        return Vector([self._data[i] - other._data[i] for i in range(len(self._data))])

    def __getitem__(self, index: int) -> float:
        return self._data[index]

    def __iter__(self):
        return iter(self._data)

    def __check_length(self, other: Vector) -> None:
        if len(self._data) != len(other._data):
            raise ValueError(f"Length of vectors does not match: one is {len(self._data)}, other is {len(other._data)}")

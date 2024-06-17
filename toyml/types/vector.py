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
    """

    data: list[float]

    # def mean(self) -> float:
    #     pass
    #
    # def std(self) -> float:
    #     pass

    def __repr__(self) -> str:
        return f"Vector{self.data}"

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other: Vector) -> Vector:
        self.__check_length(other)
        return Vector([self.data[i] + other.data[i] for i in range(len(self.data))])

    def __sub__(self, other: Vector) -> Vector:
        self.__check_length(other)
        return Vector([self.data[i] - other.data[i] for i in range(len(self.data))])

    def __getitem__(self, index: int) -> float:
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __check_length(self, other: Vector) -> bool:
        return len(self.data) == len(other.data)

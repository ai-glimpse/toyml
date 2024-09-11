from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Type, TypeVar


class BaseB(ABC):
    @abstractmethod
    def method(self) -> str:
        pass


class B1(BaseB):
    def method(self) -> str:
        return "B1 method"


class B2(BaseB):
    def method(self) -> str:
        return "B2 method"


B = TypeVar("B", bound=BaseB)


@dataclass
class A(Generic[B]):
    b: Type[B]

    def do_something(self) -> str:
        instance = self.b()
        return instance.method()


# Usage
a1 = A[B1](B1)
print(a1.do_something())  # Output: B1 method

a2 = A[B2](B2)
print(a2.do_something())  # Output: B2 method

# This will raise a TypeError
# class NotB:
#     pass
# a3 = A[NotB](NotB)  # TypeError: Type argument 'NotB' of 'A' must be a subtype of 'BaseB'

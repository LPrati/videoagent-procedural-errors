"""Frame-selection helpers shared by adapters."""

from typing import Sequence, TypeVar

T = TypeVar("T")


def evenly_spaced(items: Sequence[T], count: int) -> list[T]:
    """Pick ``count`` items uniformly spanning the sequence, first included.

    Returns the whole sequence when it is shorter than ``count``.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if len(items) <= count:
        return list(items)
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]

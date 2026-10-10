import pytest

from video_agent.adapters.frames import evenly_spaced


def test_returns_all_items_when_short() -> None:
    assert evenly_spaced([1, 2, 3], 5) == [1, 2, 3]


def test_picks_uniform_subset_starting_at_first() -> None:
    assert evenly_spaced(list(range(10)), 4) == [0, 2, 5, 7]
    assert evenly_spaced(list(range(10)), 10) == list(range(10))


def test_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError):
        evenly_spaced([1], 0)

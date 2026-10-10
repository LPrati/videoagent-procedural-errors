"""Uncertainty around a metric, and whether two configurations really differ.

Instances of the same recording are not independent — one confused agent
produces a whole recording of wrong verdicts — so the bootstrap resamples
*recordings* and recomputes the metric on each resample. The statistic is passed in as a
function of a list of recording ids, which is what makes this module a pure
function of the resample: the caller owns the matched instances, this module
owns the draw.

Comparisons between configurations are paired (the same videos) and are read
against the run-to-run noise floor: an interval that excludes zero is necessary,
not sufficient, for a difference to be reported as detected.
"""

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

Statistic = Callable[[Sequence[str]], float | None]
"""A metric recomputed from a (possibly repeated) list of recording ids."""


@dataclass(frozen=True)
class CI:
    """A point estimate and its percentile interval.

    ``n_resamples`` is the number of resamples the statistic was *defined*
    on, not the number requested: a resample that draws no positive instance
    has no recall, and averaging a zero in its place would bias the interval
    down. A reader needs to know how many of the 2 000 draws actually spoke.
    """

    point: float | None
    low: float | None
    high: float | None
    n_resamples: int

    def to_dict(self) -> dict:
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "n_resamples": self.n_resamples,
        }


def _check(n: int, level: float) -> None:
    if n < 0:
        raise ValueError(f"n_resamples must be >= 0, got {n}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")


def _percentiles(values: list[float], level: float) -> tuple[float, float]:
    """Percentile bounds of a resample distribution at confidence ``level``."""
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(np.asarray(values, dtype=float), [tail, 100.0 - tail])
    return float(low), float(high)


def _draws(recording_ids: Sequence[str], n: int, seed: int) -> list[list[str]]:
    """The ``n`` resamples of ``recording_ids``, drawn with replacement.

    Materialized so a paired comparison can hand the *same* draws to both
    configurations: the difference between two runs on different resamples
    would carry the resampling noise twice.
    """
    ids = list(recording_ids)
    if not ids or n == 0:
        return []
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(ids), size=(n, len(ids)))
    return [[ids[index] for index in row] for row in indices]


def bootstrap_by_recording(
    stat: Statistic,
    recording_ids: Sequence[str],
    *,
    n: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> CI:
    """Percentile confidence interval of ``stat``, resampling recordings."""
    _check(n, level)
    ids = list(recording_ids)
    point = stat(ids) if ids else None
    values = [
        value for draw in _draws(ids, n, seed) if (value := stat(draw)) is not None
    ]
    if not values:
        return CI(point=point, low=None, high=None, n_resamples=0)
    low, high = _percentiles(values, level)
    return CI(point=point, low=low, high=high, n_resamples=len(values))


def paired_difference_ci(
    stat_a: Statistic,
    stat_b: Statistic,
    recording_ids: Sequence[str],
    *,
    n: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> CI:
    """Interval of ``stat_a − stat_b`` over the same resampled recordings.

    Both configurations saw the same videos, so the comparison is paired and
    the draw is shared; a resample where either statistic is undefined is
    dropped, since their difference is undefined too.
    """
    _check(n, level)
    ids = list(recording_ids)
    value_a = stat_a(ids) if ids else None
    value_b = stat_b(ids) if ids else None
    point = None if value_a is None or value_b is None else value_a - value_b
    values: list[float] = []
    for draw in _draws(ids, n, seed):
        first = stat_a(draw)
        second = stat_b(draw)
        if first is not None and second is not None:
            values.append(first - second)
    if not values:
        return CI(point=point, low=None, high=None, n_resamples=0)
    low, high = _percentiles(values, level)
    return CI(point=point, low=low, high=high, n_resamples=len(values))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts.

    Only the pairs the two configurations disagree on carry information, and
    under the null each of them is a fair coin, so the p-value is the
    two-sided binomial tail — computed exactly with ``math.comb`` rather than
    through the chi-square approximation, which is unusable at the handful of
    discordant instances a split of a few dozen recordings produces. No disagreement
    at all is no evidence of a difference: p = 1.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be >= 0, got b={b}, c={c}")
    total = b + c
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail / float(2**total))


def discordant_counts(
    pred_a: Sequence[bool], pred_b: Sequence[bool], truth: Sequence[bool]
) -> tuple[int, int]:
    """Count the instances where exactly one configuration was right.

    Returns ``(b, c)``: ``b`` instances A got right and B got wrong, ``c``
    the other way round — the two cells McNemar's test reads.
    """
    if not len(pred_a) == len(pred_b) == len(truth):
        raise ValueError(
            f"predictions and truth must align: {len(pred_a)}, {len(pred_b)}, "
            f"{len(truth)}"
        )
    b = 0
    c = 0
    for first, second, label in zip(pred_a, pred_b, truth):
        if first == label and second != label:
            b += 1
        elif first != label and second == label:
            c += 1
    return b, c

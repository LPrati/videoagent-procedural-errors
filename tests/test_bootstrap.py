"""Confidence intervals by recording and the paired comparison tests."""

from typing import Sequence

import pytest

from video_agent.eval.bootstrap import (
    bootstrap_by_recording,
    discordant_counts,
    mcnemar,
    paired_difference_ci,
)

IDS = [f"1_{index}" for index in range(20)]


def constant(value: float):
    def statistic(sample: Sequence[str]) -> float | None:
        return value

    return statistic


def mean_index(sample: Sequence[str]) -> float | None:
    """Mean of the number in each id: varies with the draw, so the CI is wide."""
    if not sample:
        return None
    return sum(int(recording_id.split("_")[1]) for recording_id in sample) / len(sample)


class TestBootstrap:
    def test_a_constant_statistic_has_an_interval_of_zero_width(self) -> None:
        interval = bootstrap_by_recording(constant(0.5), IDS, n=100, seed=0)

        assert interval.point == 0.5
        assert (interval.low, interval.high) == (0.5, 0.5)
        assert interval.n_resamples == 100

    def test_the_interval_brackets_the_point_estimate(self) -> None:
        interval = bootstrap_by_recording(mean_index, IDS, n=500, seed=0)

        assert interval.point == pytest.approx(9.5)
        assert interval.low < interval.point < interval.high

    def test_the_same_seed_draws_the_same_resamples(self) -> None:
        first = bootstrap_by_recording(mean_index, IDS, n=200, seed=7)
        again = bootstrap_by_recording(mean_index, IDS, n=200, seed=7)
        other = bootstrap_by_recording(mean_index, IDS, n=200, seed=8)

        assert (first.low, first.high) == (again.low, again.high)
        assert (first.low, first.high) != (other.low, other.high)

    def test_a_wider_level_gives_a_wider_interval(self) -> None:
        narrow = bootstrap_by_recording(mean_index, IDS, n=500, seed=0, level=0.5)
        wide = bootstrap_by_recording(mean_index, IDS, n=500, seed=0, level=0.99)

        assert wide.low <= narrow.low and narrow.high <= wide.high

    def test_resamples_where_the_metric_is_undefined_are_dropped(self) -> None:
        def only_with_b(sample: Sequence[str]) -> float | None:
            return 1.0 if "b" in sample else None

        interval = bootstrap_by_recording(only_with_b, ["a", "b"], n=200, seed=0)

        assert interval.point == 1.0
        assert 0 < interval.n_resamples < 200

    def test_an_always_undefined_metric_has_no_interval(self) -> None:
        interval = bootstrap_by_recording(lambda sample: None, IDS, n=50, seed=0)

        assert (interval.point, interval.low, interval.high) == (None, None, None)
        assert interval.n_resamples == 0

    def test_no_recordings_is_no_estimate(self) -> None:
        interval = bootstrap_by_recording(constant(1.0), [], n=50, seed=0)

        assert interval.point is None and interval.n_resamples == 0

    def test_a_level_outside_zero_and_one_is_a_contract_violation(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_by_recording(constant(1.0), IDS, level=1.5)
        with pytest.raises(ValueError):
            bootstrap_by_recording(constant(1.0), IDS, n=-1)


class TestPairedDifference:
    def test_two_identical_statistics_differ_by_exactly_zero(self) -> None:
        interval = paired_difference_ci(mean_index, mean_index, IDS, n=200, seed=0)

        assert (interval.point, interval.low, interval.high) == (0.0, 0.0, 0.0)
        assert interval.n_resamples == 200

    def test_a_constant_gap_survives_every_resample(self) -> None:
        interval = paired_difference_ci(constant(0.7), constant(0.4), IDS, n=100, seed=1)

        assert interval.point == pytest.approx(0.3)
        assert (interval.low, interval.high) == pytest.approx((0.3, 0.3))

    def test_a_resample_undefined_on_either_side_is_dropped(self) -> None:
        interval = paired_difference_ci(
            constant(1.0), lambda sample: None, IDS, n=50, seed=0
        )

        assert interval.point is None and interval.n_resamples == 0


class TestMcNemar:
    def test_no_disagreement_is_no_evidence(self) -> None:
        assert mcnemar(0, 0) == 1.0

    def test_ten_wins_to_nothing_is_two_coin_flips_of_ten(self) -> None:
        assert mcnemar(10, 0) == pytest.approx(2 * 0.5**10)
        assert mcnemar(0, 10) == pytest.approx(2 * 0.5**10)

    def test_a_split_of_three_to_seven_is_the_binomial_tail(self) -> None:
        # 2 * (C(10,0) + C(10,1) + C(10,2) + C(10,3)) / 2^10 = 352 / 1024.
        assert mcnemar(3, 7) == pytest.approx(0.34375)

    def test_an_even_split_cannot_exceed_certainty(self) -> None:
        assert mcnemar(5, 5) == 1.0

    def test_negative_counts_are_a_contract_violation(self) -> None:
        with pytest.raises(ValueError):
            mcnemar(-1, 3)


class TestDiscordantCounts:
    def test_counts_only_the_instances_the_two_runs_disagree_on(self) -> None:
        truth = [True, True, False, False, True]
        run_a = [True, False, False, True, True]
        run_b = [False, False, True, True, True]

        # Instance 0: A right, B wrong. Instance 2: A right, B wrong.
        # Instances 1, 3 and 4: both the same.
        assert discordant_counts(run_a, run_b, truth) == (2, 0)

    def test_a_mismatched_length_is_a_contract_violation(self) -> None:
        with pytest.raises(ValueError):
            discordant_counts([True], [True, False], [True])

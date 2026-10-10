"""The evaluation metrics, checked against counts worked out by hand."""

import pytest

from video_agent.eval.matching import MatchedInstance
from video_agent.eval.metrics import (
    RESPECTS,
    VIOLATES,
    CostStats,
    aggregate_any_error_prf,
    category_prf,
    cost_summary,
    eda,
    eda_all_normal,
    indeterminate_rate,
    interval_iou,
    is_positive_prediction,
    is_positive_truth,
    localization,
    pairwise_order_accuracy,
    per_video_prf,
)
from video_agent.memory.types import Segment
from video_agent.task.annotations import (
    MISSING_TAG,
    ORDER_TAG,
    Recording,
    StepInstance,
    TaskGraph,
)
from video_agent.task.verdicts import (
    Anchor,
    Execution,
    Order,
    RecordingVerdicts,
    StepVerdict,
)

ORDER = frozenset({ORDER_TAG})
MISSING = frozenset({MISSING_TAG})
TECHNIQUE = frozenset({"Technique"})
NONE: frozenset[str] = frozenset()


def instance(
    step_id: int,
    tags: frozenset[str] = NONE,
    *,
    start: float | None = None,
    end: float | None = None,
    occurrence: int = 0,
) -> StepInstance:
    return StepInstance(
        recording_id="1_7",
        step_id=step_id,
        occurrence=occurrence,
        description=f"step {step_id}",
        start_s=start,
        end_s=end,
        tags=tags,
    )


def verdict(
    step_id: int,
    execution: Execution = Execution.EXECUTED,
    order: Order = Order.OK,
    *,
    anchor: tuple[float, float] | None = None,
    prerequisite: int | None = None,
) -> StepVerdict:
    return StepVerdict(
        step_id=step_id,
        execution=execution,
        order=order,
        anchor=(
            None
            if anchor is None
            else Anchor(start_s=anchor[0], end_s=anchor[1], segments=(0, 0))
        ),
        prerequisite_step_id=prerequisite,
        prerequisite_anchor=None,
        raw=None,
    )


def matched(*pairs: tuple[StepInstance, StepVerdict | None]) -> list[MatchedInstance]:
    return [MatchedInstance(instance=i, verdict=v) for i, v in pairs]


def recording(*instances: StepInstance) -> Recording:
    return Recording(
        recording_id="1_7", activity_id=1, is_error=True, steps=tuple(instances)
    )


def verdicts(*steps: StepVerdict) -> RecordingVerdicts:
    return RecordingVerdicts(recording_id="1_7", steps=tuple(steps), status="ok")


# Eight instances covering every cell of both categories at once. Counted by
# hand: order tp=1 fp=1 fn=2 tn=4; omission tp=1 fp=1 fn=1 tn=5.
EIGHT = matched(
    (instance(1, ORDER), verdict(1, order=Order.VIOLATION)),  # order tp
    (instance(2, ORDER), verdict(2)),  # order fn
    (instance(3), verdict(3, order=Order.VIOLATION)),  # order fp
    (instance(4), verdict(4)),  # both tn
    (
        instance(5, MISSING),
        verdict(5, Execution.NOT_EXECUTED, Order.NOT_APPLICABLE),
    ),  # omission tp
    (
        instance(6, MISSING),
        verdict(6, Execution.INDETERMINATE, Order.INDETERMINATE),
    ),  # omission fn
    (
        instance(7),
        verdict(7, Execution.NOT_EXECUTED, Order.NOT_APPLICABLE),
    ),  # omission fp
    (instance(8, ORDER), None),  # order fn: no verdict is a negative
)


class TestTruthAndPrediction:
    def test_only_the_categorys_own_tag_is_a_positive(self) -> None:
        assert is_positive_truth(instance(1, ORDER), "order")
        assert not is_positive_truth(instance(1, MISSING), "order")
        assert not is_positive_truth(instance(1, TECHNIQUE), "omission")

    def test_indeterminate_and_missing_verdicts_predict_nothing(self) -> None:
        undecided = verdict(1, Execution.INDETERMINATE, Order.INDETERMINATE)

        assert not is_positive_prediction(undecided, "order")
        assert not is_positive_prediction(None, "omission")

    def test_an_unknown_category_is_a_contract_violation(self) -> None:
        with pytest.raises(ValueError):
            is_positive_truth(instance(1), "technique")


class TestCategoryPrf:
    def test_counts_every_cell_of_the_order_category(self) -> None:
        prf = category_prf(EIGHT, "order")

        assert (prf.tp, prf.fp, prf.fn, prf.tn) == (1, 1, 2, 4)
        assert prf.precision == 0.5
        assert prf.recall == pytest.approx(1 / 3)
        assert prf.f1 == pytest.approx(0.4)

    def test_counts_every_cell_of_the_omission_category(self) -> None:
        prf = category_prf(EIGHT, "omission")

        assert (prf.tp, prf.fp, prf.fn, prf.tn) == (1, 1, 1, 5)
        assert (prf.precision, prf.recall, prf.f1) == (0.5, 0.5, 0.5)

    def test_a_system_that_predicts_nothing_has_no_precision(self) -> None:
        prf = category_prf(matched((instance(1, ORDER), verdict(1))), "order")

        assert (prf.tp, prf.fp, prf.fn, prf.tn) == (0, 0, 1, 0)
        assert prf.precision is None
        assert prf.recall == 0.0
        assert prf.f1 is None

    def test_a_category_with_no_positives_at_all_has_no_recall(self) -> None:
        prf = category_prf(matched((instance(1), verdict(1))), "order")

        assert (prf.precision, prf.recall, prf.f1) == (None, None, None)
        assert prf.prevalence == 0.0

    def test_an_empty_set_of_instances_is_undefined_everywhere(self) -> None:
        prf = category_prf([], "omission")

        assert (prf.tp, prf.fp, prf.fn, prf.tn) == (0, 0, 0, 0)
        assert (prf.precision, prf.recall, prf.f1, prf.prevalence) == (
            None,
            None,
            None,
            None,
        )


class TestPerVideo:
    def test_a_recording_counts_once_however_many_instances_it_has(self) -> None:
        prf = per_video_prf(
            {
                "tp": matched(
                    (instance(1, ORDER), verdict(1, order=Order.VIOLATION)),
                    (instance(2), verdict(2)),
                ),
                "fp": matched((instance(1), verdict(1, order=Order.VIOLATION))),
                "fn": matched((instance(1, ORDER), verdict(1))),
                "tn": matched((instance(1), verdict(1))),
            },
            "order",
        )

        assert (prf.tp, prf.fp, prf.fn, prf.tn) == (1, 1, 1, 1)
        assert (prf.precision, prf.recall) == (0.5, 0.5)


class TestAnyError:
    def test_any_tag_is_the_truth_and_any_error_verdict_the_prediction(self) -> None:
        prf = aggregate_any_error_prf(EIGHT)

        assert (prf.tp, prf.fp, prf.fn, prf.tn) == (2, 2, 3, 1)
        assert prf.precision == 0.5
        assert prf.recall == pytest.approx(0.4)


class TestPairwiseOrder:
    GRAPH = TaskGraph(
        activity_id=1,
        prerequisites={1: frozenset(), 2: frozenset({1}), 3: frozenset({2})},
        edges=frozenset({(1, 2), (2, 3)}),
    )
    IN_ORDER = recording(
        instance(1, start=0.0, end=10.0),
        instance(2, start=20.0, end=30.0),
        instance(3, start=40.0, end=50.0),
    )

    def test_anchors_decide_the_predicted_order(self) -> None:
        stats = pairwise_order_accuracy(
            self.IN_ORDER,
            verdicts(
                verdict(1, anchor=(0.0, 10.0)),
                verdict(2, anchor=(20.0, 30.0)),
                verdict(3, anchor=(40.0, 50.0)),
            ),
            self.GRAPH,
        )

        assert (stats[RESPECTS].n, stats[RESPECTS].correct) == (2, 2)
        assert stats[RESPECTS].accuracy == 1.0
        assert stats[VIOLATES].n == 0
        assert stats[VIOLATES].accuracy is None

    def test_a_swapped_pair_lands_in_the_violates_stratum(self) -> None:
        swapped = recording(
            instance(2, start=0.0, end=10.0),
            instance(1, start=20.0, end=30.0),
            instance(3, start=40.0, end=50.0),
        )

        stats = pairwise_order_accuracy(
            swapped,
            verdicts(
                verdict(1, anchor=(20.0, 30.0)),
                verdict(2, anchor=(0.0, 10.0), order=Order.VIOLATION, prerequisite=1),
                verdict(3, anchor=(40.0, 50.0)),
            ),
            self.GRAPH,
        )

        assert (stats[VIOLATES].n, stats[VIOLATES].correct) == (1, 1)
        assert (stats[RESPECTS].n, stats[RESPECTS].correct) == (1, 1)

    def test_a_cited_prerequisite_answers_the_pair_without_anchors(self) -> None:
        stats = pairwise_order_accuracy(
            self.IN_ORDER,
            verdicts(
                verdict(1),
                verdict(2),
                verdict(3, order=Order.VIOLATION, prerequisite=2),
            ),
            self.GRAPH,
        )

        # (1, 2) reads "ordem ok" as respects; (2, 3) cites 2, hence violates —
        # and the annotation says the pair respects the order, so it is wrong.
        assert stats[RESPECTS].determinate == 2
        assert stats[RESPECTS].correct == 1

    def test_an_undecided_verdict_is_counted_but_not_scored(self) -> None:
        stats = pairwise_order_accuracy(
            self.IN_ORDER,
            verdicts(
                verdict(1),
                verdict(2),
                verdict(3, Execution.INDETERMINATE, Order.INDETERMINATE),
            ),
            self.GRAPH,
        )

        assert stats[RESPECTS].n == 2
        assert stats[RESPECTS].determinate == 1
        assert stats[RESPECTS].accuracy == 1.0

    def test_a_pair_with_an_untimed_step_is_not_a_pair(self) -> None:
        stats = pairwise_order_accuracy(
            recording(
                instance(1, start=0.0, end=10.0),
                instance(2),
                instance(3, start=40.0, end=50.0),
            ),
            verdicts(verdict(1), verdict(2), verdict(3)),
            self.GRAPH,
        )

        assert stats[RESPECTS].n == 0 and stats[VIOLATES].n == 0

    def test_a_recording_that_was_never_run_decides_nothing(self) -> None:
        stats = pairwise_order_accuracy(self.IN_ORDER, None, self.GRAPH)

        assert stats[RESPECTS].n == 2
        assert stats[RESPECTS].determinate == 0
        assert stats[RESPECTS].accuracy is None

    def test_the_transitive_closure_adds_the_indirect_pairs(self) -> None:
        stats = pairwise_order_accuracy(
            self.IN_ORDER, None, self.GRAPH, edges="transitive"
        )

        assert stats[RESPECTS].n == 3  # (1,2), (2,3) and (1,3)

    def test_an_unknown_edge_set_is_a_contract_violation(self) -> None:
        with pytest.raises(ValueError):
            pairwise_order_accuracy(self.IN_ORDER, None, self.GRAPH, edges="all")


class TestLocalization:
    def test_iou_of_two_intervals(self) -> None:
        assert interval_iou((0.0, 10.0), (5.0, 15.0)) == pytest.approx(5 / 15)
        assert interval_iou((0.0, 10.0), (0.0, 10.0)) == 1.0
        assert interval_iou((0.0, 10.0), (20.0, 30.0)) == 0.0
        assert interval_iou((5.0, 5.0), (5.0, 5.0)) == 0.0

    def test_every_instance_lands_in_the_bin_of_its_duration(self) -> None:
        bins = localization(
            matched(
                (instance(1, start=0.0, end=5.0), verdict(1, anchor=(0.0, 5.0))),
                (instance(2, start=0.0, end=15.0), verdict(2, anchor=(0.0, 15.0))),
                (instance(3, start=0.0, end=25.0), verdict(3, anchor=(0.0, 25.0))),
                (instance(4, start=0.0, end=45.0), verdict(4, anchor=(0.0, 45.0))),
                (instance(5, start=0.0, end=100.0), verdict(5, anchor=(0.0, 100.0))),
            )
        )

        assert [stats.n for stats in bins.values()] == [1, 1, 1, 1, 1]
        assert all(stats.mean_iou == 1.0 for stats in bins.values())
        assert all(stats.hit_at_half == 1.0 for stats in bins.values())

    def test_the_mean_iou_is_over_the_anchored_instances_only(self) -> None:
        bins = localization(
            matched(
                (instance(1, start=0.0, end=8.0), verdict(1, anchor=(0.0, 4.0))),
                (instance(2, start=0.0, end=8.0), verdict(2)),
            )
        )
        first = bins[(0, 10)]

        assert (first.n, first.anchored) == (2, 1)
        assert first.mean_iou == pytest.approx(0.5)
        assert first.hit_at_half == 1.0

    def test_only_executed_steps_with_times_are_localized(self) -> None:
        bins = localization(
            matched(
                (instance(1, MISSING), verdict(1, Execution.NOT_EXECUTED)),
                (instance(2, start=0.0, end=5.0), None),
                (
                    instance(3, start=0.0, end=5.0),
                    verdict(3, Execution.INDETERMINATE, Order.INDETERMINATE),
                ),
            )
        )

        assert all(stats.n == 0 for stats in bins.values())
        assert bins[(0, 10)].mean_iou is None

    def test_a_low_iou_is_no_hit(self) -> None:
        bins = localization(
            matched(
                (instance(1, start=0.0, end=100.0), verdict(1, anchor=(0.0, 20.0)))
            )
        )

        assert bins[(80, float("inf"))].mean_iou == pytest.approx(0.2)
        assert bins[(80, float("inf"))].hit_at_half == 0.0


class TestIndeterminateRate:
    def test_omission_counts_missing_and_undecided_executions(self) -> None:
        assert indeterminate_rate(EIGHT, "omission") == pytest.approx(2 / 8)

    def test_order_skips_the_steps_reported_as_not_executed(self) -> None:
        assert indeterminate_rate(EIGHT, "order") == pytest.approx(2 / 6)

    def test_nothing_to_decide_is_not_a_rate(self) -> None:
        assert indeterminate_rate([], "order") is None


class TestEda:
    GRID = [
        Segment(index=i, start_s=2.0 * i, end_s=2.0 * i + 2.0, frame_range=(60 * i, 60 * i + 60))
        for i in range(10)
    ]
    # Step 1 is clean, step 2 carries the error between 8 s and 12 s, so the
    # segments labelled "erro" are exactly 4 and 5.
    RECORDING = recording(
        instance(1, start=0.0, end=4.0), instance(2, ORDER, start=8.0, end=12.0)
    )

    def test_an_anchor_on_the_error_gets_every_segment_right(self) -> None:
        assert eda(
            self.RECORDING,
            verdicts(
                verdict(1),
                verdict(2, order=Order.VIOLATION, anchor=(8.0, 12.0)),
            ),
            self.GRID,
        ) == (10, 10)

    def test_an_anchor_on_the_wrong_step_misses_four_segments(self) -> None:
        assert eda(
            self.RECORDING,
            verdicts(verdict(1, order=Order.VIOLATION, anchor=(0.0, 4.0)), verdict(2)),
            self.GRID,
        ) == (6, 10)

    def test_a_run_that_flagged_nothing_scores_the_normal_baseline(self) -> None:
        assert eda(self.RECORDING, verdicts(verdict(1), verdict(2)), self.GRID) == (
            8,
            10,
        )
        assert eda(self.RECORDING, None, self.GRID) == (8, 10)
        assert eda_all_normal(self.RECORDING, self.GRID) == (8, 10)


class TestCost:
    RECORDS = [
        {
            "wall_s": 10.0,
            "usage": {"llm_calls": 1},
            "usage_total": {
                "llm_calls": 4,
                "tool_calls": 3,
                "tokens_in": 1000,
                "tokens_out": 100,
            },
        },
        {
            "wall_s": 30.0,
            "usage_total": {
                "llm_calls": 6,
                "tool_calls": 5,
                "tokens_in": 3000,
                "tokens_out": 200,
            },
        },
    ]

    def test_averages_the_whole_episode_including_sub_agents(self) -> None:
        cost = cost_summary(self.RECORDS)

        assert cost.recordings == 2
        assert cost.mean_llm_calls == 5.0
        assert cost.mean_tool_calls == 4.0
        assert cost.mean_tokens_in == 2000.0
        assert cost.total_tokens_out == 300
        assert (cost.mean_wall_s, cost.median_wall_s) == (20.0, 20.0)

    def test_no_records_costs_nothing_and_says_so(self) -> None:
        assert cost_summary([]) == CostStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

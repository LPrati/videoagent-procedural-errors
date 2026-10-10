"""The constant baselines every results table carries, and the noise floor."""


import pytest

from video_agent.eval.baselines import all_error, all_normal, noise_floor
from video_agent.eval.matching import count_optional_unperformed, match_recording
from video_agent.eval.metrics import category_prf
from video_agent.task.annotations import (
    MISSING_TAG,
    ORDER_TAG,
    AnnotationSet,
    Recording,
    Sop,
    StepInstance,
)
from video_agent.task.verdicts import Execution, Order

from tests._paths import load_annotations

OPTIONAL_ID = "18_3"
"""A dev recording with the optional step 195, not performed."""

OPTIONAL_STEP = 195

SOP = Sop(
    activity_id=1,
    recipe_name="Toast",
    steps=((10, "Slice the bread"), (11, "Toast the bread"), (12, "Spread the butter")),
)


def instance(step_id: int, tags: tuple[str, ...] = ()) -> StepInstance:
    return StepInstance(
        recording_id="1_7",
        step_id=step_id,
        occurrence=0,
        description=f"step {step_id}",
        start_s=0.0,
        end_s=10.0,
        tags=frozenset(tags),
    )


RECORDING = Recording(
    recording_id="1_7",
    activity_id=1,
    is_error=True,
    steps=(
        instance(10, (ORDER_TAG,)),
        instance(11, (MISSING_TAG,)),
        instance(12),
    ),
)


class TestAllNormal:
    def test_reports_every_sop_step_as_executed_and_in_order(self) -> None:
        verdicts = all_normal("1_7", SOP)

        assert [step.step_id for step in verdicts.steps] == [10, 11, 12]
        assert all(step.execution is Execution.EXECUTED for step in verdicts.steps)
        assert all(step.order is Order.OK for step in verdicts.steps)
        assert all(step.anchor is None for step in verdicts.steps)
        assert verdicts.recording_id == "1_7"

    def test_never_finds_an_error_so_its_recall_is_zero(self) -> None:
        matched = match_recording(RECORDING, all_normal("1_7", SOP))

        for category in ("order", "omission"):
            prf = category_prf(matched, category)
            assert prf.recall == 0.0
            assert prf.precision is None  # it predicted nothing to be right about
            assert prf.f1 is None


class TestAllError:
    def test_flags_every_step_in_both_categories_at_once(self) -> None:
        verdicts = all_error("1_7", SOP)

        assert all(
            step.execution is Execution.NOT_EXECUTED for step in verdicts.steps
        )
        assert all(step.order is Order.VIOLATION for step in verdicts.steps)

    def test_buys_recall_one_at_the_price_of_the_prevalence(self) -> None:
        matched = match_recording(RECORDING, all_error("1_7", SOP))

        for category in ("order", "omission"):
            prf = category_prf(matched, category)
            assert prf.recall == 1.0
            assert prf.precision == pytest.approx(1 / 3)
            assert prf.precision == prf.prevalence


class TestNoiseFloor:
    def test_is_the_absolute_gap_between_two_runs_of_one_configuration(self) -> None:
        floor = noise_floor(
            {"order.recall": 0.5, "omission.recall": 0.2},
            {"order.recall": 0.2, "omission.recall": 0.35},
        )

        assert floor["order.recall"] == pytest.approx(0.3)
        assert floor["omission.recall"] == pytest.approx(0.15)

    def test_a_metric_undefined_in_either_run_has_no_floor(self) -> None:
        floor = noise_floor(
            {"order.precision": None, "order.recall": 0.5},
            {"order.precision": 0.4, "order.recall": 0.5},
        )

        assert floor["order.precision"] is None
        assert floor["order.recall"] == 0.0

    def test_a_metric_only_one_run_reports_is_kept_as_undefined(self) -> None:
        floor = noise_floor({"a": 1.0}, {"b": 1.0})

        assert floor == {"a": None, "b": None}


@pytest.fixture(scope="module")
def annotations() -> AnnotationSet:
    return load_annotations()


class TestTheOptionalStepReachesTheBaselinesToo:
    """Optional steps are excluded where the universe is built, so baselines inherit it.

    The baselines go through ``match_recording`` exactly as a real run does,
    which is the point of producing them as ordinary verdicts: if the
    exclusion were implemented per metric instead, "all error" would carry a
    false positive of a missing step that no system could avoid.
    """

    def test_the_recording_really_holds_the_untimed_untagged_instance(
        self, annotations: AnnotationSet
    ) -> None:
        recording = annotations.recordings[OPTIONAL_ID]
        optional = [
            step for step in recording.steps if step.step_id == OPTIONAL_STEP
        ]

        assert len(optional) == 1
        assert optional[0].start_s is None and optional[0].tags == frozenset()
        assert count_optional_unperformed(recording) == 1

    def test_all_error_does_not_pay_a_false_positive_for_it(
        self, annotations: AnnotationSet
    ) -> None:
        recording = annotations.recordings[OPTIONAL_ID]
        sop, _ = annotations.for_recording(OPTIONAL_ID)

        matched = match_recording(recording, all_error(OPTIONAL_ID, sop))
        prf = category_prf(matched, "omission")

        assert not any(
            entry.instance.step_id == OPTIONAL_STEP
            and entry.instance.start_s is None
            for entry in matched
        )
        assert len(matched) == len(recording.steps) - 1
        assert prf.tp + prf.fp == len(matched)

    def test_all_normal_does_not_score_it_as_a_true_negative(
        self, annotations: AnnotationSet
    ) -> None:
        recording = annotations.recordings[OPTIONAL_ID]
        sop, _ = annotations.for_recording(OPTIONAL_ID)

        matched = match_recording(recording, all_normal(OPTIONAL_ID, sop))
        prf = category_prf(matched, "omission")

        assert prf.tp + prf.fp + prf.fn + prf.tn == len(recording.steps) - 1

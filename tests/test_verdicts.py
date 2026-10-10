"""Turning a model's answer object into the verdicts the evaluation scores."""

import pytest
from tests.fakes import make_temporal_memory

from video_agent.task.annotations import Sop
from video_agent.task.verdicts import (
    Anchor,
    Execution,
    Order,
    RecordingVerdicts,
    StepVerdict,
    normalize,
)

SOP = Sop(
    activity_id=1,
    recipe_name="Toast",
    steps=((10, "Slice the bread"), (11, "Toast the bread"), (12, "Spread the butter")),
)


@pytest.fixture
def segments():
    """The 10-segment, 2-second grid: segment i spans [2i, 2i + 2)."""
    return make_temporal_memory(10).segments


def answer(*entries: dict) -> dict:
    return {"steps": list(entries)}


def by_id(verdicts) -> dict[int, StepVerdict]:
    return {verdict.step_id: verdict for verdict in verdicts}


def test_one_verdict_per_sop_step_in_sop_order(segments) -> None:
    verdicts = normalize(answer({"step_id": 12, "executed": "no"}), SOP, segments)

    assert [verdict.step_id for verdict in verdicts] == [10, 11, 12]


def test_a_missing_answer_makes_every_step_indeterminate(segments) -> None:
    verdicts = normalize(None, SOP, segments)

    for verdict in verdicts:
        assert verdict.execution is Execution.INDETERMINATE
        assert verdict.order is Order.INDETERMINATE
        assert verdict.anchor is None and verdict.raw is None


def test_a_step_the_model_skipped_is_indeterminate(segments) -> None:
    verdicts = by_id(
        normalize(answer({"step_id": 10, "executed": "yes"}), SOP, segments)
    )

    assert verdicts[11].execution is Execution.INDETERMINATE
    assert verdicts[11].order is Order.INDETERMINATE
    assert verdicts[11].raw is None


def test_an_executed_step_gets_its_anchor_in_seconds(segments) -> None:
    entry = {"step_id": 10, "executed": "yes", "segments": [1, 3], "order": "ok"}

    verdict = by_id(normalize(answer(entry), SOP, segments))[10]

    assert verdict.execution is Execution.EXECUTED
    assert verdict.order is Order.OK
    assert verdict.anchor == Anchor(start_s=2.0, end_s=8.0, segments=(1, 3))
    assert verdict.raw == entry


def test_a_step_reported_as_not_executed_has_no_order_verdict(segments) -> None:
    verdict = by_id(
        normalize(
            answer({"step_id": 10, "executed": "no", "order": "violation"}),
            SOP,
            segments,
        )
    )[10]

    assert verdict.execution is Execution.NOT_EXECUTED
    # An order verdict exists only for an execution to order.
    assert verdict.order is Order.NOT_APPLICABLE


def test_an_unsure_step_is_indeterminate_on_both_axes(segments) -> None:
    verdicts = by_id(
        normalize(
            answer(
                {"step_id": 10, "executed": "unsure", "order": "ok"},
                {"step_id": 11, "executed": "probably", "order": "ok"},
                {"step_id": 12, "order": "ok"},
            ),
            SOP,
            segments,
        )
    )

    for step_id in (10, 11, 12):
        assert verdicts[step_id].execution is Execution.INDETERMINATE
        assert verdicts[step_id].order is Order.INDETERMINATE


def test_an_unreadable_order_value_is_indeterminate(segments) -> None:
    verdict = by_id(
        normalize(
            answer({"step_id": 10, "executed": "yes", "order": "probably fine"}),
            SOP,
            segments,
        )
    )[10]

    assert verdict.execution is Execution.EXECUTED
    assert verdict.order is Order.INDETERMINATE


def test_a_violation_cites_a_prerequisite_and_its_anchor(segments) -> None:
    verdict = by_id(
        normalize(
            answer(
                {
                    "step_id": 12,
                    "executed": "yes",
                    "segments": [4, 4],
                    "order": "violation",
                    "violated_prerequisite": {"step_id": 10, "segments": [6, 7]},
                }
            ),
            SOP,
            segments,
        )
    )[12]

    assert verdict.order is Order.VIOLATION
    assert verdict.prerequisite_step_id == 10
    assert verdict.prerequisite_anchor == Anchor(12.0, 16.0, (6, 7))


def test_a_prerequisite_outside_the_sop_is_dropped(segments) -> None:
    verdict = by_id(
        normalize(
            answer(
                {
                    "step_id": 12,
                    "executed": "yes",
                    "order": "violation",
                    "violated_prerequisite": {"step_id": 99, "segments": [1, 2]},
                }
            ),
            SOP,
            segments,
        )
    )[12]

    assert verdict.prerequisite_step_id is None
    assert verdict.prerequisite_anchor is None


def test_a_reversed_range_is_swapped(segments) -> None:
    verdict = by_id(
        normalize(
            answer({"step_id": 10, "executed": "yes", "segments": [5, 2]}),
            SOP,
            segments,
        )
    )[10]

    assert verdict.anchor == Anchor(start_s=4.0, end_s=12.0, segments=(2, 5))


def test_a_range_outside_the_grid_is_clamped(segments) -> None:
    verdict = by_id(
        normalize(
            answer({"step_id": 10, "executed": "yes", "segments": [-4, 99]}),
            SOP,
            segments,
        )
    )[10]

    assert verdict.anchor == Anchor(start_s=0.0, end_s=20.0, segments=(0, 9))


@pytest.mark.parametrize(
    "value",
    [None, [], [3], [1, 2, 3], "1-3", ["1", "3"], [1.5, 3.5], [True, False], {"first": 1}],
)
def test_a_malformed_range_leaves_the_verdict_without_an_anchor(segments, value) -> None:
    verdict = by_id(
        normalize(
            answer({"step_id": 10, "executed": "yes", "segments": value}),
            SOP,
            segments,
        )
    )[10]

    assert verdict.execution is Execution.EXECUTED
    assert verdict.anchor is None


def test_entries_the_sop_does_not_contain_are_ignored(segments) -> None:
    verdicts = normalize(
        answer(
            {"step_id": 99, "executed": "yes", "segments": [0, 1]},
            {"step_id": "10", "executed": "yes"},
            "not an entry",
            {"step_id": 10, "executed": "no"},
        ),
        SOP,
        segments,
    )

    assert [verdict.step_id for verdict in verdicts] == [10, 11, 12]
    assert by_id(verdicts)[10].execution is Execution.NOT_EXECUTED


def test_a_repeated_step_id_keeps_the_first_entry(segments) -> None:
    verdict = by_id(
        normalize(
            answer(
                {"step_id": 10, "executed": "yes", "segments": [0, 0]},
                {"step_id": 10, "executed": "no"},
            ),
            SOP,
            segments,
        )
    )[10]

    assert verdict.execution is Execution.EXECUTED


def test_verdicts_survive_a_json_round_trip(segments) -> None:
    import json

    verdicts = normalize(
        answer(
            {
                "step_id": 10,
                "executed": "yes",
                "segments": [1, 2],
                "order": "violation",
                "violated_prerequisite": {"step_id": 11, "segments": [4, 5]},
            },
            {"step_id": 11, "executed": "no"},
        ),
        SOP,
        segments,
    )
    record = RecordingVerdicts(recording_id="1_7", steps=verdicts, status="ok")

    restored = RecordingVerdicts.from_dict(json.loads(json.dumps(record.to_dict())))

    assert restored == record
    assert restored.steps[0].execution is Execution.EXECUTED
    assert restored.steps[0].order is Order.VIOLATION


def test_the_verdict_labels_are_the_stored_record_values() -> None:
    """The labels are part of the run record format and stay in Portuguese."""
    assert Execution.NOT_EXECUTED.value == "não executado"
    assert Order.VIOLATION.value == "violação de ordem"
    assert Order.NOT_APPLICABLE.value == "n/a"

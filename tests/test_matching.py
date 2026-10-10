"""Which verdict speaks about which occurrence of a step.

Also which occurrences are units of analysis at all: an optional step that
was not performed is taken out of the universe, and this is the module that
builds the universe.
"""

from video_agent.eval.matching import (
    count_optional_unperformed,
    is_optional_unperformed,
    match_recording,
)
from video_agent.eval.metrics import category_prf
from video_agent.task.annotations import (
    MISSING_TAG,
    Recording,
    StepInstance,
)
from video_agent.task.verdicts import (
    Anchor,
    Execution,
    Order,
    RecordingVerdicts,
    StepVerdict,
)


def instance(
    step_id: int,
    occurrence: int = 0,
    start: float | None = None,
    end: float | None = None,
    tags: tuple[str, ...] = (),
) -> StepInstance:
    return StepInstance(
        recording_id="1_7",
        step_id=step_id,
        occurrence=occurrence,
        description=f"step {step_id}",
        start_s=start,
        end_s=end,
        tags=frozenset(tags),
    )


def verdict(
    step_id: int,
    execution: Execution = Execution.EXECUTED,
    order: Order = Order.OK,
    anchor: tuple[float, float] | None = None,
) -> StepVerdict:
    return StepVerdict(
        step_id=step_id,
        execution=execution,
        order=order,
        anchor=(
            None
            if anchor is None
            else Anchor(start_s=anchor[0], end_s=anchor[1], segments=(0, 1))
        ),
        prerequisite_step_id=None,
        prerequisite_anchor=None,
        raw=None,
    )


def recording(*instances: StepInstance) -> Recording:
    return Recording(
        recording_id="1_7", activity_id=1, is_error=False, steps=tuple(instances)
    )


def verdicts(*steps: StepVerdict) -> RecordingVerdicts:
    return RecordingVerdicts(recording_id="1_7", steps=tuple(steps), status="ok")


def test_one_verdict_per_step_reaches_its_only_occurrence() -> None:
    matched = match_recording(
        recording(instance(10, start=0.0, end=5.0), instance(11, start=5.0, end=9.0)),
        verdicts(verdict(10), verdict(11, order=Order.VIOLATION)),
    )

    assert [entry.instance.step_id for entry in matched] == [10, 11]
    assert matched[0].verdict is not None and matched[0].verdict.step_id == 10
    assert matched[1].verdict is not None
    assert matched[1].verdict.order is Order.VIOLATION


def test_a_second_occurrence_of_a_step_answered_once_has_no_verdict() -> None:
    matched = match_recording(
        recording(
            instance(10, occurrence=0, start=0.0, end=5.0),
            instance(10, occurrence=1, start=50.0, end=55.0),
        ),
        verdicts(verdict(10, anchor=(0.0, 5.0))),
    )

    assert matched[0].verdict is not None
    assert matched[1].verdict is None


def test_a_not_executed_verdict_applies_to_every_occurrence() -> None:
    matched = match_recording(
        recording(
            instance(10, occurrence=0, start=0.0, end=5.0),
            instance(10, occurrence=1, start=50.0, end=55.0),
        ),
        verdicts(
            verdict(10, execution=Execution.NOT_EXECUTED, order=Order.NOT_APPLICABLE)
        ),
    )

    assert [entry.verdict.execution for entry in matched] == [
        Execution.NOT_EXECUTED,
        Execution.NOT_EXECUTED,
    ]


def test_verdicts_of_one_step_are_paired_in_the_order_of_their_anchors() -> None:
    matched = match_recording(
        recording(
            instance(10, occurrence=0, start=0.0, end=5.0),
            instance(10, occurrence=1, start=50.0, end=55.0),
        ),
        # Emitted late-first; the anchor decides which occurrence each is about.
        verdicts(
            verdict(10, order=Order.VIOLATION, anchor=(50.0, 55.0)),
            verdict(10, order=Order.OK, anchor=(0.0, 5.0)),
        ),
    )

    assert matched[0].verdict.order is Order.OK
    assert matched[1].verdict.order is Order.VIOLATION


def test_an_unanchored_verdict_sorts_after_the_anchored_ones() -> None:
    matched = match_recording(
        recording(
            instance(10, occurrence=0, start=0.0, end=5.0),
            instance(10, occurrence=1, start=50.0, end=55.0),
        ),
        verdicts(
            verdict(10, order=Order.VIOLATION),
            verdict(10, order=Order.OK, anchor=(0.0, 5.0)),
        ),
    )

    assert matched[0].verdict.order is Order.OK
    assert matched[1].verdict.order is Order.VIOLATION


def test_a_step_the_answer_never_mentions_is_unmatched() -> None:
    matched = match_recording(
        recording(instance(10, start=0.0, end=5.0), instance(11, start=5.0, end=9.0)),
        verdicts(verdict(10)),
    )

    assert matched[1].verdict is None


def test_a_recording_that_was_never_run_has_no_verdicts_at_all() -> None:
    matched = match_recording(
        recording(
            instance(10, start=0.0, end=5.0),
            instance(11, tags=(MISSING_TAG,)),
        ),
        None,
    )

    assert [entry.verdict for entry in matched] == [None, None]
    assert len(matched) == 2


class TestOptionalUnperformed:
    """An optional step nobody did is not a unit of analysis.

    Step 195 "Top with more parmesan if desired" is the only step of the
    dataset annotated without an interval and without ``Missing Step``. It
    was not done and that is not an error, so the correct verdict "não
    executado" (not executed) must not be scored as a false positive of a
    missing step — and the
    instance must not become a true negative either.
    """

    def test_untimed_and_untagged_is_the_optional_case(self) -> None:
        assert is_optional_unperformed(instance(195))

    def test_an_omission_is_untimed_but_tagged_so_it_stays(self) -> None:
        assert not is_optional_unperformed(instance(195, tags=(MISSING_TAG,)))

    def test_a_timed_instance_is_never_the_optional_case(self) -> None:
        assert not is_optional_unperformed(instance(195, start=0.0, end=5.0))
        assert not is_optional_unperformed(
            instance(195, start=0.0, end=5.0, tags=("Technique Error",))
        )

    def test_the_excluded_instances_are_counted_before_they_are_dropped(
        self,
    ) -> None:
        sample = recording(
            instance(10, start=0.0, end=5.0),
            instance(195),
            instance(11, tags=(MISSING_TAG,)),
        )

        assert count_optional_unperformed(sample) == 1

    def test_it_never_reaches_the_matched_pairs(self) -> None:
        matched = match_recording(
            recording(instance(10, start=0.0, end=5.0), instance(195)),
            verdicts(
                verdict(10),
                verdict(
                    195, execution=Execution.NOT_EXECUTED, order=Order.NOT_APPLICABLE
                ),
            ),
        )

        assert [entry.instance.step_id for entry in matched] == [10]

    def test_calling_it_not_executed_costs_no_precision(self) -> None:
        """The point of the caveat: the right answer is not punished."""
        sample = recording(
            instance(10, start=0.0, end=5.0, tags=(MISSING_TAG,)),
            instance(195),
        )
        answered = verdicts(
            verdict(10, execution=Execution.NOT_EXECUTED, order=Order.NOT_APPLICABLE),
            verdict(195, execution=Execution.NOT_EXECUTED, order=Order.NOT_APPLICABLE),
        )

        prf = category_prf(match_recording(sample, answered), "omission")

        assert (prf.tp, prf.fp, prf.fn, prf.tn) == (1, 0, 0, 0)
        assert prf.precision == 1.0

    def test_it_is_not_a_true_negative_either(self) -> None:
        sample = recording(instance(10, start=0.0, end=5.0), instance(195))
        answered = verdicts(verdict(10), verdict(195))

        prf = category_prf(match_recording(sample, answered), "omission")

        assert (prf.tp, prf.fp, prf.fn, prf.tn) == (0, 0, 0, 1)

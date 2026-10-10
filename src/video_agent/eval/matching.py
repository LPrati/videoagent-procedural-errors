"""Pair every annotated step instance with the verdict that speaks about it.

The unit of analysis is the step *instance* — one occurrence of a step in one
recording — and ``(recording, step_id)`` is not a key: a recipe step
performed twice yields two instances. A run, on the other hand, emits one
verdict per SOP step, because that is what the question asks for. Scoring
therefore needs an explicit correspondence: by occurrence, in temporal order.

An occurrence with no verdict is not an error and not a negative either; it
is unclassified, which is what ``verdict is None`` means downstream.

This is also where the instance universe of a recording is decided, so it is
where optional steps are handled: an optional step that was not performed is
not a unit of analysis at all (``is_optional_unperformed``).
"""

from collections import defaultdict
from dataclasses import dataclass

from video_agent.task.annotations import MISSING_TAG, Recording, StepInstance
from video_agent.task.verdicts import Execution, RecordingVerdicts, StepVerdict


@dataclass(frozen=True)
class MatchedInstance:
    """One annotated occurrence and the verdict assigned to it.

    ``verdict is None`` covers both halves of "the system said nothing about
    this occurrence": the recording was never run (or its run failed before
    producing an answer) and the step was answered for fewer times than it
    was performed. Both count as unclassified, hence as a negative
    prediction.
    """

    instance: StepInstance
    verdict: StepVerdict | None


def is_optional_unperformed(instance: StepInstance) -> bool:
    """Whether ``instance`` is an optional step that was simply not done.

    The annotation writes no interval for a step that never happened and
    tags it ``Missing Step`` when the missing step is the error. Exactly one SOP
    step of the whole dataset is untimed *without* that tag — step 195 "Top
    with more parmesan if desired" of recipe 18, in six recordings (18_3,
    18_11, 18_27 in dev; 18_19, 18_33, 18_101 in test). It is optional, it
    was not done, and that is not an error, so the correct verdict "não
    executado" (not executed) would be scored as a false positive of a
    missing step.

    These instances are therefore dropped from every metric rather
    than counting them anywhere — not as a true negative either, which is
    why the filter lives in ``match_recording`` and not in each metric.
    """
    return instance.start_s is None and not any(
        MISSING_TAG in tag for tag in instance.tags
    )


def count_optional_unperformed(recording: Recording) -> int:
    """How many instances of ``recording`` are excluded as optional and unperformed.

    The report has to say how many units it dropped, and the exclusion
    happens inside ``match_recording``, where the count is no longer visible.
    """
    return sum(is_optional_unperformed(instance) for instance in recording.steps)


def _anchor_order(verdict: StepVerdict) -> tuple[int, float]:
    """Sort key placing anchored verdicts first, in time; unanchored last."""
    if verdict.anchor is None:
        return (1, 0.0)
    return (0, verdict.anchor.start_s)


def match_recording(
    recording: Recording, verdicts: RecordingVerdicts | None
) -> list[MatchedInstance]:
    """Assign one verdict (or none) to each annotated instance of ``recording``.

    Both sides are grouped by step id and ordered in time — the annotation by
    ``occurrence``, which ``load_recordings`` numbers chronologically, and the
    verdicts by the start of their anchor — and then paired k-th with k-th
    The reference system emits exactly one verdict per step, so the pairing
    only ever matters for a step performed twice: the first occurrence takes the
    verdict and the second has none.

    One verdict escapes the positional rule. ``não executado`` (not
    executed) is a claim
    about the whole recording — the step never happened — so it applies to
    every occurrence of that step rather than to the first one only;
    otherwise a system that missed a repeated step would be scored as right
    about it once and silent about it once.

    ``verdicts is None`` (the recording was not run, or its record is
    missing) leaves every instance unmatched.

    Instances that ``is_optional_unperformed`` recognizes never reach the
    result: they are outside the instance universe, so no metric sees them
    in any cell.
    """
    instances = [
        instance
        for instance in recording.steps
        if not is_optional_unperformed(instance)
    ]
    if verdicts is None:
        return [
            MatchedInstance(instance=instance, verdict=None)
            for instance in instances
        ]

    by_step: dict[int, list[StepVerdict]] = defaultdict(list)
    for verdict in verdicts.steps:
        by_step[verdict.step_id].append(verdict)
    for candidates in by_step.values():
        candidates.sort(key=_anchor_order)

    matched: list[MatchedInstance] = []
    for instance in instances:
        candidates = by_step.get(instance.step_id, [])
        if instance.occurrence < len(candidates):
            verdict: StepVerdict | None = candidates[instance.occurrence]
        else:
            verdict = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.execution is Execution.NOT_EXECUTED
                ),
                None,
            )
        matched.append(MatchedInstance(instance=instance, verdict=verdict))
    return matched

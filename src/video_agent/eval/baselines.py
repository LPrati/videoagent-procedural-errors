"""Constant baselines that every results table reports.

Two baselines are not systems at all but constants: "everything normal" and
"everything is an error". They cost nothing to run and give the reader a
reference point: the first shows how high EDA climbs on the prior alone, the
second that recall 1 is free and its precision equals the prevalence.
Producing them as ordinary ``RecordingVerdicts`` means they go through the
same matching and metric code as a real run, so a mistake in the metric code
shows up in them too.

Two further baselines, the empty memory and the shuffled memory, are runs of
the real system under a modified memory, not constants; they are produced
with ``eval.synthetic`` and the memory options of the run scripts.
"""

from typing import Mapping

from video_agent.task.annotations import Sop
from video_agent.task.verdicts import (
    Execution,
    Order,
    RecordingVerdicts,
    StepVerdict,
)

ALL_NORMAL_STATUS = "baseline:all_normal"
ALL_ERROR_STATUS = "baseline:all_error"
"""Status strings marking verdicts that no model produced."""


def _verdicts(
    recording_id: str, sop: Sop, execution: Execution, order: Order, status: str
) -> RecordingVerdicts:
    return RecordingVerdicts(
        recording_id=recording_id,
        steps=tuple(
            StepVerdict(
                step_id=step_id,
                execution=execution,
                order=order,
                anchor=None,
                prerequisite_step_id=None,
                prerequisite_anchor=None,
                raw=None,
            )
            for step_id, _ in sop.steps
        ),
        status=status,
    )


def all_normal(recording_id: str, sop: Sop) -> RecordingVerdicts:
    """"All normal" baseline: every step executed, in order, nothing flagged.

    No anchors: the baseline claims no localization, so it is unclassified
    for localization rather than wrong, and it predicts no error segment for
    EDA.
    """
    return _verdicts(
        recording_id, sop, Execution.EXECUTED, Order.OK, ALL_NORMAL_STATUS
    )


def all_error(recording_id: str, sop: Sop) -> RecordingVerdicts:
    """"All error" baseline: every step flagged as missing *and* out of order.

    The two categories are scored as separate binary problems, so this
    baseline is positive in both at once — an inconsistent verdict that no
    system would emit, and exactly the point: it recovers recall 1 in both
    categories at a precision equal to the prevalence.
    """
    return _verdicts(
        recording_id, sop, Execution.NOT_EXECUTED, Order.VIOLATION, ALL_ERROR_STATUS
    )


def noise_floor(
    metrics_run1: Mapping[str, float | None], metrics_run2: Mapping[str, float | None]
) -> dict[str, float | None]:
    """Absolute per-metric difference between two runs of one configuration.

    The orchestrator is a sampler: two executions of the same configuration
    over the same videos disagree, and that disagreement is the noise floor
    below which a difference between configurations is not a result.
    A metric undefined in either run has no floor, hence ``None``; keys are
    the union of both runs so a metric that only one run defined is visible
    rather than silently dropped.
    """
    keys = sorted(set(metrics_run1) | set(metrics_run2))
    floor: dict[str, float | None] = {}
    for key in keys:
        first = metrics_run1.get(key)
        second = metrics_run2.get(key)
        floor[key] = (
            None if first is None or second is None else abs(float(first) - float(second))
        )
    return floor

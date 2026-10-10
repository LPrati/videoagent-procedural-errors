"""The discrete verdicts the evaluation scores, and how a model reply becomes one.

Every SOP step gets two verdicts, execution and order, each anchored in
time. A missing or unusable answer is unclassified (``indeterminado``),
which counts as a negative for both error categories and is reported
separately. ``normalize`` is where that conservatism lives: it never
guesses. A step the model skipped, an ``executed`` value it invented, a
segment range it wrote as prose: all of them come out unclassified with the
raw entry kept beside the verdict, so error analysis can see what was
actually said.

The label values are Portuguese strings and are part of the stored run
record format; they must not be translated:

* execution: ``"executado"`` (executed), ``"não executado"`` (not executed,
  i.e. a missing step), ``"indeterminado"`` (unclassified);
* order: ``"ordem ok"`` (order ok), ``"violação de ordem"`` (order
  violation), ``"indeterminado"`` (unclassified), ``"n/a"`` (not applicable,
  for a step reported as not executed).
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from video_agent.memory.store import segment_interval
from video_agent.memory.types import Segment
from video_agent.task.annotations import Sop


class Execution(StrEnum):
    """Did the step happen? Values are stored record labels (see module docstring)."""

    EXECUTED = "executado"
    NOT_EXECUTED = "não executado"
    INDETERMINATE = "indeterminado"


class Order(StrEnum):
    """Was the step's order respected? Only meaningful when it was executed.

    Values are stored record labels (see module docstring).
    """

    OK = "ordem ok"
    VIOLATION = "violação de ordem"
    INDETERMINATE = "indeterminado"
    NOT_APPLICABLE = "n/a"


@dataclass(frozen=True)
class Anchor:
    """Where a verdict was seen: a time span and the segments it came from.

    The model answers in segment ids because that is what its tools return;
    localization is scored in seconds. Both are kept, and ``segments`` holds
    the clamped ids the span was actually computed from, not what the model
    asked for.
    """

    start_s: float
    end_s: float
    segments: tuple[int, int]

    def to_dict(self) -> dict:
        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "segments": list(self.segments),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Anchor":
        first, last = data["segments"]
        return cls(
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            segments=(int(first), int(last)),
        )


@dataclass(frozen=True)
class StepVerdict:
    """The two verdicts for one SOP step, with their anchors and their source."""

    step_id: int
    execution: Execution
    order: Order
    anchor: Anchor | None
    prerequisite_step_id: int | None
    prerequisite_anchor: Anchor | None
    raw: dict | None  # the model's own entry, kept for error analysis

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "execution": self.execution.value,
            "order": self.order.value,
            "anchor": None if self.anchor is None else self.anchor.to_dict(),
            "prerequisite_step_id": self.prerequisite_step_id,
            "prerequisite_anchor": (
                None
                if self.prerequisite_anchor is None
                else self.prerequisite_anchor.to_dict()
            ),
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StepVerdict":
        anchor = data["anchor"]
        prerequisite_anchor = data["prerequisite_anchor"]
        return cls(
            step_id=int(data["step_id"]),
            execution=Execution(data["execution"]),
            order=Order(data["order"]),
            anchor=None if anchor is None else Anchor.from_dict(anchor),
            prerequisite_step_id=(
                None
                if data["prerequisite_step_id"] is None
                else int(data["prerequisite_step_id"])
            ),
            prerequisite_anchor=(
                None
                if prerequisite_anchor is None
                else Anchor.from_dict(prerequisite_anchor)
            ),
            raw=data["raw"],
        )


@dataclass(frozen=True)
class RecordingVerdicts:
    """One recording's verdicts: one per SOP step, in SOP order."""

    recording_id: str
    steps: tuple[StepVerdict, ...]
    status: str

    def to_dict(self) -> dict:
        return {
            "recording_id": self.recording_id,
            "steps": [step.to_dict() for step in self.steps],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecordingVerdicts":
        return cls(
            recording_id=data["recording_id"],
            steps=tuple(StepVerdict.from_dict(step) for step in data["steps"]),
            status=data["status"],
        )


_EXECUTION_BY_ANSWER = {"yes": Execution.EXECUTED, "no": Execution.NOT_EXECUTED}
_ORDER_BY_ANSWER = {"ok": Order.OK, "violation": Order.VIOLATION}


def _is_integer(value: object) -> bool:
    """Whether ``value`` is a JSON integer (``True``/``False`` are not)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_anchor(value: object, segments: Sequence[Segment]) -> Anchor | None:
    """Turn a ``[first, last]`` segment range into a time anchor.

    Out-of-range ids are clamped to the grid and a reversed range is swapped,
    because both are the model miscounting rather than refusing to answer;
    anything that is not a pair of integers is no anchor at all.
    """
    if not segments or not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    first, last = value
    if not _is_integer(first) or not _is_integer(last):
        return None
    if first > last:
        first, last = last, first
    highest = len(segments) - 1
    first = min(max(int(first), 0), highest)
    last = min(max(int(last), 0), highest)
    start_s, end_s = segment_interval(segments, first, last)
    return Anchor(start_s=start_s, end_s=end_s, segments=(first, last))


def _parse_prerequisite(
    value: object, sop: Sop, segments: Sequence[Segment]
) -> tuple[int | None, Anchor | None]:
    """Read the ``violated_prerequisite`` object of one entry.

    The object names the step the answer says this step is out of order
    with. A cited step that is not a step of this SOP is not a citation that
    can be audited, so it is dropped together with its anchor.
    """
    if not isinstance(value, dict):
        return None, None
    step_id = value.get("step_id")
    if not _is_integer(step_id) or step_id not in sop.step_ids():
        return None, None
    return int(step_id), _parse_anchor(value.get("segments"), segments)


def _entries_by_step(raw: dict | None) -> dict[int, dict]:
    """Index the model's answer by step id, keeping the first entry of each.

    Extra step ids the SOP does not contain are ignored here; they survive in
    the run record's ``final_answer_text``.
    """
    entries: dict[int, dict] = {}
    if raw is None:
        return entries
    for entry in raw.get("steps") or ():
        if not isinstance(entry, dict):
            continue
        step_id = entry.get("step_id")
        if _is_integer(step_id):
            entries.setdefault(int(step_id), entry)
    return entries


def normalize(
    raw: dict | None, sop: Sop, segments: Sequence[Segment]
) -> tuple[StepVerdict, ...]:
    """Turn a parsed Final Answer into one verdict per SOP step, in SOP order.

    The SOP drives the loop, not the answer: every step of the recipe is
    scored, so a step the model left out is an unclassified verdict and not
    a missing row. The order verdict is read from the answer only for an
    executed step (for a step reported as not executed it is ``n/a`` and for
    anything else it is unclassified), because an order verdict is defined
    only where there is an execution to order.
    """
    entries = _entries_by_step(raw)
    verdicts: list[StepVerdict] = []
    for step_id, _ in sop.steps:
        entry = entries.get(step_id)
        if entry is None:
            verdicts.append(
                StepVerdict(
                    step_id=step_id,
                    execution=Execution.INDETERMINATE,
                    order=Order.INDETERMINATE,
                    anchor=None,
                    prerequisite_step_id=None,
                    prerequisite_anchor=None,
                    raw=None,
                )
            )
            continue
        execution = _EXECUTION_BY_ANSWER.get(
            entry.get("executed"), Execution.INDETERMINATE
        )
        if execution is Execution.EXECUTED:
            order = _ORDER_BY_ANSWER.get(entry.get("order"), Order.INDETERMINATE)
        elif execution is Execution.NOT_EXECUTED:
            order = Order.NOT_APPLICABLE
        else:
            order = Order.INDETERMINATE
        prerequisite_step_id, prerequisite_anchor = _parse_prerequisite(
            entry.get("violated_prerequisite"), sop, segments
        )
        verdicts.append(
            StepVerdict(
                step_id=step_id,
                execution=execution,
                order=order,
                anchor=_parse_anchor(entry.get("segments"), segments),
                prerequisite_step_id=prerequisite_step_id,
                prerequisite_anchor=prerequisite_anchor,
                raw=entry,
            )
        )
    return tuple(verdicts)

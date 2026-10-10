"""Compute the order verdict from the anchors instead of the model's word.

The task question asks the orchestrator for each step's order verdict, and
that column (the *judged order*) can contradict the model's own anchors: the
anchors may place a step before something it requires while the ``order``
field written next to them says ``ordem ok`` (order ok). This module
provides the second reading, the *computed order*.

It takes the verdicts a run produced, keeps the execution column and the
anchors exactly as they are, and rewrites only ``order``: a step is a
violation when the timeline the run itself reported puts one of its
prerequisites after it. Two design choices are deliberate:

* **Both ends of a displacement are violations.** The CaptainCook4D
  annotation tags the late step *and* the early one when a block of steps
  moves, so a rule that only flags the dependent recovers about half the
  labels. The ``edges`` option keeps the narrow reading available.
* **A missing prerequisite is not an order violation.** Counting it as one
  would make every dependent of a missing step an order error, turning a
  single missing step into a cascade of false positives. Here an unanchored
  or unexecuted prerequisite marks nobody: the missing step is scored once,
  as a missing step, on the step that was not performed.

The reading is a policy and not a default: ``LLM_ORDER`` (judged order) is
what every caller gets unless it asks otherwise. Everything here is a pure
function of its arguments: new verdicts come out, and the input objects are
never touched.
"""

from dataclasses import dataclass, replace
from typing import Literal

from video_agent.task.annotations import TaskGraph
from video_agent.task.verdicts import (
    Anchor,
    Execution,
    Order,
    RecordingVerdicts,
    StepVerdict,
)

OrderSource = Literal["llm", "anchors"]
OrderRule = Literal["direct", "transitive"]
OrderEdges = Literal["early", "both"]
OrderMargin = Literal["start", "end"]


@dataclass(frozen=True)
class OrderPolicy:
    """Where the order verdict comes from, and under which rule.

    Four independent choices, each exposed as a flag of
    ``scripts/evaluate.py`` so a table can say which reading produced it:

    ``source``
        ``"llm"`` keeps what the model wrote (judged order); ``"anchors"``
        derives the column from the anchors of the same record (computed
        order).
    ``rule``
        ``"transitive"`` reads the whole closure of prerequisites
        (``TaskGraph.ancestors``), ``"direct"`` only the immediate ones. The
        annotation's convention is transitive; direct is the narrower
        reading, kept for comparison.
    ``edges``
        ``"both"`` flags the displaced step and the step it displaced,
        ``"early"`` only the one that ran too soon.
    ``margin``
        ``"end"`` requires the prerequisite to start after the dependent
        *finished*, so an overlap of two anchors is not an error; ``"start"``
        compares starts, which is stricter and noisier on wide anchors.
    """

    source: OrderSource = "llm"
    rule: OrderRule = "transitive"
    edges: OrderEdges = "both"
    margin: OrderMargin = "end"

    @property
    def label(self) -> str:
        """A short name for a status line, a CSV cell or a table header.

        ``"llm"`` alone when the model wrote the column: the other three
        knobs describe a rule that was not applied, and naming them would
        suggest the reading depended on them.
        """
        if self.source == "llm":
            return "llm"
        return f"anchors:{self.rule},{self.edges},{self.margin}"

    def to_dict(self) -> dict:
        """The policy as plain JSON, for the report and the comparison."""
        return {
            "source": self.source,
            "rule": self.rule,
            "edges": self.edges,
            "margin": self.margin,
        }


LLM_ORDER = OrderPolicy()
"""Judged order: the order column is whatever the orchestrator answered.

The default everywhere, so a report changes only when it asks for the
computed order.
"""


def _chosen_anchored(verdicts: RecordingVerdicts) -> dict[int, int]:
    """Index, per step id, of the one anchored verdict that carries its position.

    A run emits one verdict per SOP step, but nothing forbids two verdicts
    for one step, and the timeline needs a single position per step. The
    earliest anchor wins, which is the criterion ``matching._anchor_order``
    already uses to pair verdicts with occurrences, so the step is placed
    where the evaluation says its first occurrence is. Ties keep the first
    verdict in answer order.
    """
    chosen: dict[int, int] = {}
    for index, verdict in enumerate(verdicts.steps):
        if verdict.execution is not Execution.EXECUTED or verdict.anchor is None:
            continue
        current = chosen.get(verdict.step_id)
        if current is None:
            chosen[verdict.step_id] = index
            continue
        incumbent = verdicts.steps[current].anchor
        assert incumbent is not None  # only anchored verdicts are ever stored
        if verdict.anchor.start_s < incumbent.start_s:
            chosen[verdict.step_id] = index
    return chosen


def _prerequisites_of(
    graph: TaskGraph, step_id: int, rule: OrderRule
) -> frozenset[int]:
    """The steps ``step_id`` must follow, under ``rule``.

    A step the graph does not know — an answer about a step outside this
    activity, or a graph that is missing a node — constrains nothing rather
    than raising: this is a reading of a record, and a record that names an
    unknown step is a fact to score, not a crash.
    """
    if step_id not in graph.prerequisites:
        return frozenset()
    if rule == "transitive":
        return graph.ancestors(step_id)
    return graph.prerequisites[step_id]


def _offenders(
    anchors: dict[int, Anchor], graph: TaskGraph, policy: OrderPolicy
) -> dict[int, list[int]]:
    """For each anchored step, the anchored prerequisites that came after it.

    "After" is measured against the end of the step's own anchor when
    ``margin == "end"``: an anchor is a span, and a prerequisite that starts
    while the dependent is still running is an overlap the model's
    segmentation could have invented, not evidence of a displacement. The
    comparison is strict, so two steps in the same segment are not an error.

    Only anchored prerequisites can offend. A prerequisite with no anchor —
    unexecuted, unclassified, or executed without one — is silence, and
    silence is scored as a missing step, not as everybody else's order error.
    """
    offenders: dict[int, list[int]] = {}
    for step_id, anchor in anchors.items():
        reference = anchor.end_s if policy.margin == "end" else anchor.start_s
        late = sorted(
            prerequisite
            for prerequisite in _prerequisites_of(graph, step_id, policy.rule)
            if prerequisite in anchors
            and anchors[prerequisite].start_s > reference
        )
        if late:
            offenders[step_id] = late
    return offenders


def _citations(
    anchors: dict[int, Anchor], offenders: dict[int, list[int]], policy: OrderPolicy
) -> dict[int, int]:
    """Which step each violation cites, as every order violation must.

    A step that ran too early cites the *latest* prerequisite that followed
    it: of the several steps it jumped over, that is the most flagrant, and
    it is the same worst-first choice ``synthetic.violated_prerequisites``
    makes for the ground truth.

    A step that was left behind (``edges == "both"``) cites the *earliest*
    descendant that overtook it, for the mirror reason. When a step is both —
    it overtook something and was overtaken — the early-end citation wins,
    because that is the verdict that says what the step itself did wrong.
    Ties on the anchor go to the smaller step id, so the citation is a
    function of the data and not of dict order.
    """
    cited: dict[int, int] = {}
    if policy.edges == "both":
        overtaken: dict[int, list[int]] = {}
        for step_id, late in offenders.items():
            for prerequisite in late:
                overtaken.setdefault(prerequisite, []).append(step_id)
        for prerequisite, dependents in overtaken.items():
            cited[prerequisite] = min(
                dependents, key=lambda step: (anchors[step].start_s, step)
            )
    for step_id, late in offenders.items():
        cited[step_id] = max(late, key=lambda step: (anchors[step].start_s, -step))
    return cited


def derive_order_from_anchors(
    verdicts: RecordingVerdicts, graph: TaskGraph, policy: OrderPolicy
) -> RecordingVerdicts:
    """Rewrite the order column of ``verdicts`` from their own anchors.

    One verdict out per verdict in, in the same order, with ``execution``,
    ``anchor`` and ``raw`` untouched — the reading changes what is concluded
    about order, never what the run reported about execution or where it saw
    it. The order column becomes:

    * ``n/a`` for a step reported as not executed;
    * ``indeterminado`` (unclassified) for an unclassified step, for an
      executed step with no anchor, and for a second anchored verdict of a
      step the timeline already placed — none of them has a position this
      rule could compare;
    * ``violação de ordem`` (order violation) for an anchored step with an
      offender (or, under ``edges == "both"``, for one that was overtaken),
      citing the prerequisite step and its anchor;
    * ``ordem ok`` (order ok) for every other anchored step.

    ``policy.source`` is not read here: this function *is* the anchors
    reading. ``apply_order_policy`` is what dispatches on it.
    """
    chosen = _chosen_anchored(verdicts)
    anchors: dict[int, Anchor] = {}
    for step_id, index in chosen.items():
        anchor = verdicts.steps[index].anchor
        assert anchor is not None
        anchors[step_id] = anchor
    offenders = _offenders(anchors, graph, policy)
    cited = _citations(anchors, offenders, policy)

    steps: list[StepVerdict] = []
    for index, verdict in enumerate(verdicts.steps):
        if verdict.execution is Execution.NOT_EXECUTED:
            order, prerequisite = Order.NOT_APPLICABLE, None
        elif chosen.get(verdict.step_id) != index:
            # Unclassified, unanchored, or the runner-up verdict of a step
            # that already has a position: no timeline entry, no order.
            order, prerequisite = Order.INDETERMINATE, None
        elif verdict.step_id in cited:
            order, prerequisite = Order.VIOLATION, cited[verdict.step_id]
        else:
            order, prerequisite = Order.OK, None
        steps.append(
            replace(
                verdict,
                order=order,
                prerequisite_step_id=prerequisite,
                prerequisite_anchor=(
                    None if prerequisite is None else anchors[prerequisite]
                ),
            )
        )
    return RecordingVerdicts(
        recording_id=verdicts.recording_id,
        steps=tuple(steps),
        status=verdicts.status,
    )


def apply_order_policy(
    verdicts: RecordingVerdicts | None, graph: TaskGraph, policy: OrderPolicy
) -> RecordingVerdicts | None:
    """Hand the verdicts to the reading ``policy`` names, or leave them alone.

    The one place the evaluation branches on the policy, so every caller —
    the report, the comparison, the baselines — reads the same column. A
    recording with no record has no verdicts to reread and stays ``None``:
    an unanswered recording is unclassified under either reading.
    """
    if verdicts is None or policy.source == "llm":
        return verdicts
    return derive_order_from_anchors(verdicts, graph, policy)

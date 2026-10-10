"""The order verdict read off the anchors: the rule, and the harness against it.

Two halves. The first builds a task graph and a handful of verdicts by hand,
so every marking below can be checked by reading four numbers; the second
runs the rule over the seven synthetic cases and compares it with a
brute-force oracle written here, independently of the module under test — a
rule and its own implementation agreeing proves nothing.

The oracle recomputes the transitive closure from ``TaskGraph.prerequisites``
by relaxation instead of calling ``TaskGraph.ancestors``, for the same reason.
"""

import re
from dataclasses import replace

import pytest

from video_agent.eval.order_from_anchors import (
    LLM_ORDER,
    OrderPolicy,
    apply_order_policy,
    derive_order_from_anchors,
)
from video_agent.eval.synthetic import (
    Case,
    first_starts,
    is_order_clean,
    make_cases,
)
from video_agent.task.annotations import AnnotationSet, TaskGraph, load_split
from video_agent.task.verdicts import (
    Anchor,
    Execution,
    Order,
    RecordingVerdicts,
    StepVerdict,
)

from tests._paths import DEV_SPLIT, load_annotations

ANCHORS = OrderPolicy(source="anchors")
"""The default computed order: transitive closure, both ends, end margin."""

NARROW = OrderPolicy(source="anchors", rule="direct", edges="early", margin="start")
"""The narrow computed order: direct prerequisites, early end only, start margin."""


# --------------------------------------------------------------------------
# hand-built fixtures
# --------------------------------------------------------------------------


def anchor(start_s: float, end_s: float) -> Anchor:
    """An anchor at ``[start_s, end_s]``; the segment ids are decoration here."""
    return Anchor(start_s=start_s, end_s=end_s, segments=(int(start_s), int(end_s)))


def executed(step_id: int, start_s: float, end_s: float) -> StepVerdict:
    """An executed step anchored at ``[start_s, end_s]``, order left blank."""
    return StepVerdict(
        step_id=step_id,
        execution=Execution.EXECUTED,
        order=Order.INDETERMINATE,
        anchor=anchor(start_s, end_s),
        prerequisite_step_id=None,
        prerequisite_anchor=None,
        raw={"step_id": step_id, "executed": "yes"},
    )


def unanchored(step_id: int, execution: Execution) -> StepVerdict:
    """A verdict with no anchor: not executed, unclassified, or executed blind."""
    return StepVerdict(
        step_id=step_id,
        execution=execution,
        order=Order.OK,
        anchor=None,
        prerequisite_step_id=None,
        prerequisite_anchor=None,
        raw={"step_id": step_id},
    )


def recording_verdicts(*steps: StepVerdict) -> RecordingVerdicts:
    return RecordingVerdicts(recording_id="hand", steps=tuple(steps), status="ok")


def task_graph(prerequisites: dict[int, set[int]]) -> TaskGraph:
    """A graph from ``{step: direct prerequisites}``, edges derived from it."""
    return TaskGraph(
        activity_id=999,
        prerequisites={
            step: frozenset(preds) for step, preds in prerequisites.items()
        },
        edges=frozenset(
            (prerequisite, step)
            for step, preds in prerequisites.items()
            for prerequisite in preds
        ),
    )


BLOCK_GRAPH = task_graph({1: set(), 2: {1}, 3: {1}, 4: {2}, 5: set(), 6: set()})
"""Step 1 precedes 2 and 3, and 2 precedes 4; steps 5 and 6 depend on nothing."""


def violations(verdicts: RecordingVerdicts) -> set[int]:
    return {
        step.step_id for step in verdicts.steps if step.order is Order.VIOLATION
    }


def citation(verdicts: RecordingVerdicts, step_id: int) -> tuple[int | None, ...]:
    """The ``(prerequisite_step_id, prerequisite start)`` one verdict cites."""
    step = next(item for item in verdicts.steps if item.step_id == step_id)
    return (
        step.prerequisite_step_id,
        None if step.prerequisite_anchor is None else step.prerequisite_anchor.start_s,
    )


class TestADisplacedBlock:
    """One step performed late, with everything that required it done first."""

    @pytest.fixture
    def displaced(self) -> RecordingVerdicts:
        # 2 and 3 need 1 directly, 4 needs it through 2; 1 comes last.
        return recording_verdicts(
            executed(1, 100.0, 110.0),
            executed(2, 0.0, 10.0),
            executed(3, 20.0, 30.0),
            executed(4, 40.0, 50.0),
        )

    def test_the_transitive_reading_flags_the_whole_block(
        self, displaced: RecordingVerdicts
    ) -> None:
        derived = derive_order_from_anchors(displaced, BLOCK_GRAPH, ANCHORS)

        assert violations(derived) == {1, 2, 3, 4}

    def test_the_narrow_reading_flags_only_the_direct_dependents(
        self, displaced: RecordingVerdicts
    ) -> None:
        """``direct``/``early``: 4 reaches 1 only through 2, and 1 is a late end."""
        derived = derive_order_from_anchors(displaced, BLOCK_GRAPH, NARROW)

        assert violations(derived) == {2, 3}

    def test_the_early_end_cites_the_latest_prerequisite_it_jumped(
        self, displaced: RecordingVerdicts
    ) -> None:
        derived = derive_order_from_anchors(displaced, BLOCK_GRAPH, ANCHORS)

        assert citation(derived, 2) == (1, 100.0)
        assert citation(derived, 4) == (1, 100.0)

    def test_the_late_end_cites_the_earliest_step_that_overtook_it(
        self, displaced: RecordingVerdicts
    ) -> None:
        """Step 1 was overtaken by 2, 3 and 4; the first of them is the citation."""
        derived = derive_order_from_anchors(displaced, BLOCK_GRAPH, ANCHORS)

        assert citation(derived, 1) == (2, 0.0)

    def test_transitive_and_direct_agree_when_the_closure_adds_nothing(
        self, displaced: RecordingVerdicts
    ) -> None:
        direct = OrderPolicy(source="anchors", rule="direct")
        derived = derive_order_from_anchors(displaced, BLOCK_GRAPH, direct)

        # 4's only direct prerequisite is 2, which ran before it.
        assert violations(derived) == {1, 2, 3}


class TestTheCitationOfAStepWithSeveralOffenders:
    GRAPH = task_graph({10: set(), 11: set(), 12: {10, 11}})

    def test_the_latest_offender_is_the_one_named(self) -> None:
        verdicts = recording_verdicts(
            executed(12, 0.0, 10.0), executed(10, 50.0, 60.0), executed(11, 100.0, 110.0)
        )

        derived = derive_order_from_anchors(verdicts, self.GRAPH, ANCHORS)

        assert violations(derived) == {10, 11, 12}
        assert citation(derived, 12) == (11, 100.0)

    def test_a_tie_on_the_anchor_goes_to_the_smaller_step_id(self) -> None:
        """The citation has to be a function of the data, not of dict order."""
        verdicts = recording_verdicts(
            executed(12, 0.0, 10.0), executed(10, 50.0, 60.0), executed(11, 50.0, 60.0)
        )

        derived = derive_order_from_anchors(verdicts, self.GRAPH, ANCHORS)

        assert citation(derived, 12) == (10, 50.0)

    def test_being_early_beats_being_late_in_the_citation(self) -> None:
        """A step that overtook something and was overtaken cites what it jumped."""
        graph = task_graph({20: set(), 21: {20}, 22: {21}})
        verdicts = recording_verdicts(
            executed(20, 200.0, 210.0), executed(21, 100.0, 110.0), executed(22, 0.0, 10.0)
        )

        derived = derive_order_from_anchors(verdicts, graph, ANCHORS)

        assert violations(derived) == {20, 21, 22}
        # 21 is both: it overtook 20 and was overtaken by 22. It cites 20.
        assert citation(derived, 21) == (20, 200.0)
        assert citation(derived, 20) == (22, 0.0)


class TestTheMargin:
    GRAPH = task_graph({1: set(), 2: {1}})

    @pytest.fixture
    def overlapping(self) -> RecordingVerdicts:
        # 1 starts after 2 started, but well before 2 finished.
        return recording_verdicts(executed(2, 0.0, 100.0), executed(1, 50.0, 60.0))

    def test_an_overlap_is_an_error_only_when_starts_are_compared(
        self, overlapping: RecordingVerdicts
    ) -> None:
        start = OrderPolicy(source="anchors", margin="start")

        assert violations(derive_order_from_anchors(overlapping, self.GRAPH, start)) == {
            1,
            2,
        }

    def test_the_end_margin_forgives_a_prerequisite_inside_the_step(
        self, overlapping: RecordingVerdicts
    ) -> None:
        """Wide anchors overlap by construction; the end margin is what absorbs it."""
        assert (
            violations(derive_order_from_anchors(overlapping, self.GRAPH, ANCHORS))
            == set()
        )

    def test_anchors_that_merely_touch_are_not_an_error_under_either_margin(
        self,
    ) -> None:
        """The comparison is strict, so a shared boundary is not a displacement."""
        verdicts = recording_verdicts(executed(2, 0.0, 50.0), executed(1, 50.0, 60.0))

        assert (
            violations(derive_order_from_anchors(verdicts, self.GRAPH, ANCHORS))
            == set()
        )


class TestWhatIsNeverAnError:
    def test_two_unlinked_steps_may_be_performed_in_any_order(self) -> None:
        verdicts = recording_verdicts(executed(6, 0.0, 10.0), executed(5, 90.0, 100.0))

        for policy in (ANCHORS, NARROW, OrderPolicy(source="anchors", margin="start")):
            assert violations(
                derive_order_from_anchors(verdicts, BLOCK_GRAPH, policy)
            ) == set()

    @pytest.mark.parametrize(
        "missing", [Execution.NOT_EXECUTED, Execution.INDETERMINATE, Execution.EXECUTED]
    )
    def test_a_prerequisite_without_an_anchor_marks_nobody(
        self, missing: Execution
    ) -> None:
        """A prerequisite that was not localized is not an order violation.

        Counting a missing, unsure or unanchored prerequisite as a violation
        would make every dependent an order violation, turning one missing
        step into a cascade of false positives. Here it is silence.
        """
        verdicts = recording_verdicts(
            unanchored(1, missing), executed(2, 0.0, 10.0), executed(4, 20.0, 30.0)
        )

        derived = derive_order_from_anchors(verdicts, BLOCK_GRAPH, ANCHORS)

        assert violations(derived) == set()

    def test_a_step_the_graph_does_not_know_constrains_nothing(self) -> None:
        verdicts = recording_verdicts(executed(77, 0.0, 10.0), executed(2, 20.0, 30.0))

        derived = derive_order_from_anchors(verdicts, BLOCK_GRAPH, ANCHORS)

        assert violations(derived) == set()


class TestTheExecutionColumnDecidesWhatOrderIsPossible:
    def test_each_execution_value_gets_the_order_it_admits(self) -> None:
        verdicts = recording_verdicts(
            unanchored(1, Execution.NOT_EXECUTED),
            unanchored(2, Execution.INDETERMINATE),
            unanchored(3, Execution.EXECUTED),
            executed(4, 0.0, 10.0),
        )

        derived = derive_order_from_anchors(verdicts, BLOCK_GRAPH, ANCHORS)
        order = {step.step_id: step.order for step in derived.steps}

        assert order == {
            1: Order.NOT_APPLICABLE,
            2: Order.INDETERMINATE,
            3: Order.INDETERMINATE,
            4: Order.OK,
        }
        assert all(
            step.prerequisite_step_id is None and step.prerequisite_anchor is None
            for step in derived.steps
        )

    def test_the_second_verdict_of_a_step_has_no_position_to_compare(self) -> None:
        """Two anchored verdicts for one step: the earliest places it, the other
        goes unclassified — the same criterion ``matching`` pairs occurrences by."""
        verdicts = recording_verdicts(
            executed(1, 100.0, 110.0),
            executed(2, 0.0, 10.0),
            executed(2, 200.0, 210.0),
        )

        derived = derive_order_from_anchors(verdicts, BLOCK_GRAPH, ANCHORS)

        assert [step.order for step in derived.steps] == [
            Order.VIOLATION,
            Order.VIOLATION,
            Order.INDETERMINATE,
        ]
        # The block was placed by the earliest anchor of step 2, not the late one.
        assert citation(derived, 1) == (2, 0.0)


class TestPurity:
    def test_the_anchors_the_execution_and_the_raw_entry_survive(self) -> None:
        verdicts = recording_verdicts(executed(1, 100.0, 110.0), executed(2, 0.0, 10.0))

        derived = derive_order_from_anchors(verdicts, BLOCK_GRAPH, ANCHORS)

        for before, after in zip(verdicts.steps, derived.steps):
            assert after.step_id == before.step_id
            assert after.execution is before.execution
            assert after.anchor == before.anchor
            assert after.raw == before.raw
        assert derived.recording_id == verdicts.recording_id
        assert derived.status == verdicts.status

    def test_the_input_is_not_touched(self) -> None:
        verdicts = recording_verdicts(executed(1, 100.0, 110.0), executed(2, 0.0, 10.0))
        before = verdicts.to_dict()

        derive_order_from_anchors(verdicts, BLOCK_GRAPH, ANCHORS)

        assert verdicts.to_dict() == before


class TestApplyOrderPolicy:
    def test_the_llm_reading_hands_back_the_very_same_object(self) -> None:
        verdicts = recording_verdicts(executed(1, 100.0, 110.0), executed(2, 0.0, 10.0))

        assert apply_order_policy(verdicts, BLOCK_GRAPH, LLM_ORDER) is verdicts

    def test_a_recording_with_no_verdicts_has_nothing_to_reread(self) -> None:
        assert apply_order_policy(None, BLOCK_GRAPH, ANCHORS) is None
        assert apply_order_policy(None, BLOCK_GRAPH, LLM_ORDER) is None

    def test_the_anchors_reading_is_the_derivation(self) -> None:
        verdicts = recording_verdicts(executed(1, 100.0, 110.0), executed(2, 0.0, 10.0))

        assert apply_order_policy(
            verdicts, BLOCK_GRAPH, ANCHORS
        ) == derive_order_from_anchors(verdicts, BLOCK_GRAPH, ANCHORS)


class TestThePolicyItself:
    def test_the_label_of_the_llm_reading_names_no_rule(self) -> None:
        assert LLM_ORDER.label == "llm"
        assert OrderPolicy(rule="direct", edges="early", margin="start").label == "llm"

    def test_the_anchors_label_names_the_three_knobs(self) -> None:
        assert ANCHORS.label == "anchors:transitive,both,end"
        assert NARROW.label == "anchors:direct,early,start"

    def test_the_default_is_the_judged_order(self) -> None:
        assert LLM_ORDER.to_dict() == {
            "source": "llm",
            "rule": "transitive",
            "edges": "both",
            "margin": "end",
        }

    def test_a_policy_round_trips_through_its_dictionary(self) -> None:
        assert OrderPolicy(**NARROW.to_dict()) == NARROW


# --------------------------------------------------------------------------
# the seven cases of the synthetic harness
# --------------------------------------------------------------------------


def transitive_prerequisites(graph: TaskGraph) -> dict[int, set[int]]:
    """The closure of ``graph.prerequisites``, computed by relaxation.

    Written here rather than taken from ``TaskGraph.ancestors`` so the oracle
    below shares no code with what it checks.
    """
    closure = {step: set(preds) for step, preds in graph.prerequisites.items()}
    changed = True
    while changed:
        changed = False
        for preds in closure.values():
            grown = set(preds)
            for prerequisite in preds:
                grown |= closure.get(prerequisite, set())
            if grown != preds:
                preds |= grown
                changed = True
    return closure


def oracle_violations(
    verdicts: RecordingVerdicts, graph: TaskGraph, policy: OrderPolicy
) -> set[int]:
    """Every step the computed-order rule marks, by brute force over all pairs."""
    anchors: dict[int, Anchor] = {}
    for step in verdicts.steps:
        if step.execution is not Execution.EXECUTED or step.anchor is None:
            continue
        held = anchors.get(step.step_id)
        if held is None or step.anchor.start_s < held.start_s:
            anchors[step.step_id] = step.anchor
    closure = transitive_prerequisites(graph)
    flagged: set[int] = set()
    for dependent, anchor_of in anchors.items():
        if policy.rule == "transitive":
            prerequisites = closure.get(dependent, set())
        else:
            prerequisites = set(graph.prerequisites.get(dependent, frozenset()))
        reference = anchor_of.end_s if policy.margin == "end" else anchor_of.start_s
        for prerequisite in prerequisites:
            if prerequisite not in anchors:
                continue
            if anchors[prerequisite].start_s > reference:
                flagged.add(dependent)
                if policy.edges == "both":
                    flagged.add(prerequisite)
    return flagged


N_RECORDINGS = 5
LINKED_NOTE = re.compile(
    r"steps (\d+) and (\d+) exchanged intervals, and \1 is a prerequisite of \2"
)
"""Reads the (prerequisite, dependent) pair case (c) swapped out of its note."""


@pytest.fixture(scope="module")
def annotations() -> AnnotationSet:
    return load_annotations()


@pytest.fixture(scope="module")
def base_recordings(annotations: AnnotationSet) -> list[str]:
    """The same five normal dev recordings ``test_synthetic`` builds its cases on."""
    _, normal = load_split(DEV_SPLIT)
    chosen: list[str] = []
    seen: set[int] = set()
    for recording_id in normal:
        recording = annotations.recordings[recording_id]
        graph = annotations.graphs[recording.activity_id]
        if recording.activity_id in seen:
            continue
        if any(start is None for start in first_starts(recording).values()):
            continue
        if not is_order_clean(recording, graph):
            continue
        seen.add(recording.activity_id)
        chosen.append(recording_id)
        if len(chosen) == N_RECORDINGS:
            break
    assert len(chosen) == N_RECORDINGS
    return chosen


@pytest.fixture(scope="module")
def cases(
    annotations: AnnotationSet, base_recordings: list[str]
) -> dict[tuple[str, str], Case]:
    built: dict[tuple[str, str], Case] = {}
    for recording_id in base_recordings:
        recording = annotations.recordings[recording_id]
        sop, graph = annotations.for_recording(recording_id)
        for case in make_cases(recording, sop, graph):
            built[(recording_id, case.name)] = case
    return built


def graph_of(annotations: AnnotationSet, recording_id: str) -> TaskGraph:
    return annotations.for_recording(recording_id)[1]


SCORED_CASES = (
    "normal",
    "omission",
    "linked_swap",
    "unlinked_swap",
    "omission_and_swap",
    "repeated_step",
)
"""Every case with an expected answer; (g) shuffled has none to reread."""


class TestTheHarnessAgainstAnIndependentOracle:
    @pytest.mark.parametrize("policy", [ANCHORS, NARROW], ids=["anchors", "narrow"])
    @pytest.mark.parametrize("name", SCORED_CASES)
    def test_the_rule_marks_what_brute_force_marks(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
        name: str,
        policy: OrderPolicy,
    ) -> None:
        for recording_id in base_recordings:
            case = cases[(recording_id, name)]
            assert case.expected is not None
            graph = graph_of(annotations, recording_id)

            derived = derive_order_from_anchors(case.expected, graph, policy)

            assert violations(derived) == oracle_violations(
                case.expected, graph, policy
            ), f"{recording_id}/{name} under {policy.label}"

    def test_every_citation_names_a_step_with_an_anchor(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        """A violation cites a step and its interval; both or neither."""
        for recording_id in base_recordings:
            for name in SCORED_CASES:
                case = cases[(recording_id, name)]
                derived = derive_order_from_anchors(
                    case.expected, graph_of(annotations, recording_id), ANCHORS
                )
                for step in derived.steps:
                    if step.order is Order.VIOLATION:
                        assert step.prerequisite_step_id is not None
                        assert step.prerequisite_anchor is not None
                    else:
                        assert step.prerequisite_step_id is None
                        assert step.prerequisite_anchor is None


class TestWhatEachCaseShouldLookLike:
    def test_the_untouched_recording_has_no_order_error_under_any_policy(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        for recording_id in base_recordings:
            graph = graph_of(annotations, recording_id)
            expected = cases[(recording_id, "normal")].expected
            for policy in (ANCHORS, NARROW):
                assert (
                    violations(derive_order_from_anchors(expected, graph, policy))
                    == set()
                )

    def test_a_missing_step_is_a_missing_step_and_not_an_order_error(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        """The one deliberate disagreement with the harness's own expectation.

        Case (b) removes a step, and the harness's direct-prerequisite rule
        makes every step that required it an order violation (a prerequisite
        that was not performed counts). Those verdicts are recognizable: they
        cite a prerequisite that has no anchor, because there is no interval
        to cite. The computed order does not count a missing prerequisite, so
        it marks nobody, and the missing step is scored once, as a missing
        step.
        """
        for recording_id in base_recordings:
            graph = graph_of(annotations, recording_id)
            expected = cases[(recording_id, "omission")].expected
            harness = [
                step for step in expected.steps if step.order is Order.VIOLATION
            ]

            assert harness, f"{recording_id}: case (b) flags nothing to compare with"
            assert all(step.prerequisite_anchor is None for step in harness)
            for policy in (ANCHORS, NARROW):
                assert (
                    violations(derive_order_from_anchors(expected, graph, policy))
                    == set()
                )

    def test_swapping_two_unlinked_steps_is_never_an_error(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        for recording_id in base_recordings:
            graph = graph_of(annotations, recording_id)
            expected = cases[(recording_id, "unlinked_swap")].expected
            for policy in (ANCHORS, NARROW):
                assert (
                    violations(derive_order_from_anchors(expected, graph, policy))
                    == set()
                )

    def test_repeating_a_step_is_not_an_error(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        for recording_id in base_recordings:
            graph = graph_of(annotations, recording_id)
            expected = cases[(recording_id, "repeated_step")].expected
            for policy in (ANCHORS, NARROW):
                assert (
                    violations(derive_order_from_anchors(expected, graph, policy))
                    == set()
                )

    def test_a_linked_swap_flags_both_ends_when_starts_are_compared(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        """Case (c): the dependent that ran first and the prerequisite it passed."""
        both_start = OrderPolicy(source="anchors", margin="start")
        for recording_id in base_recordings:
            case = cases[(recording_id, "linked_swap")]
            graph = graph_of(annotations, recording_id)
            match = LINKED_NOTE.search(case.note)
            assert match, case.note
            prerequisite, dependent = int(match.group(1)), int(match.group(2))

            flagged = violations(
                derive_order_from_anchors(case.expected, graph, both_start)
            )
            narrow = violations(derive_order_from_anchors(case.expected, graph, NARROW))

            assert {prerequisite, dependent} <= flagged
            assert dependent in narrow
            assert prerequisite not in narrow  # ``edges="early"`` flags one end

    def test_the_end_margin_cannot_see_a_swap_of_two_adjacent_steps(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        """The known blind spot of ``margin="end"``, measured rather than assumed.

        Exchanging the intervals of two steps performed back to back leaves
        the prerequisite starting exactly where the dependent's anchor ends —
        and the grid rounds both outwards, so they usually overlap by a
        segment. The end margin asks for a strict gap, so the displacement is
        invisible to it, while ``margin="start"`` sees it. This is the price
        of forgiving wide overlapping anchors, and it is why
        ``--order-margin`` is a flag.
        """
        blind = 0
        for recording_id in base_recordings:
            case = cases[(recording_id, "linked_swap")]
            graph = graph_of(annotations, recording_id)
            match = LINKED_NOTE.search(case.note)
            assert match, case.note
            prerequisite, dependent = int(match.group(1)), int(match.group(2))
            anchors = {
                step.step_id: step.anchor
                for step in case.expected.steps
                if step.anchor is not None
            }
            gap = anchors[prerequisite].start_s - anchors[dependent].end_s

            flagged = violations(
                derive_order_from_anchors(case.expected, graph, ANCHORS)
            )

            if gap > 0:
                assert {prerequisite, dependent} <= flagged
            else:
                blind += 1
                assert prerequisite not in flagged
        assert blind, "no adjacent swap among the five recordings to demonstrate it"

    def test_the_two_errors_at_once_keep_only_the_swap(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        """Case (e): the omitted step contributes nothing to the order column."""
        both_start = OrderPolicy(source="anchors", margin="start")
        for recording_id in base_recordings:
            case = cases[(recording_id, "omission_and_swap")]
            graph = graph_of(annotations, recording_id)
            omitted = {
                step.step_id
                for step in case.expected.steps
                if step.execution is Execution.NOT_EXECUTED
            }

            flagged = violations(
                derive_order_from_anchors(case.expected, graph, both_start)
            )

            assert flagged
            assert not flagged & omitted


class TestTheRuleOverTheWholeHarness:
    def test_no_case_ever_flags_a_step_the_run_did_not_anchor(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        for recording_id in base_recordings:
            graph = graph_of(annotations, recording_id)
            for name in SCORED_CASES:
                expected = cases[(recording_id, name)].expected
                anchored = {
                    step.step_id
                    for step in expected.steps
                    if step.execution is Execution.EXECUTED and step.anchor is not None
                }
                for policy in (ANCHORS, NARROW):
                    assert (
                        violations(derive_order_from_anchors(expected, graph, policy))
                        <= anchored
                    )

    def test_rereading_a_derived_answer_changes_nothing(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        """The reading depends on anchors only, so it is idempotent."""
        for recording_id in base_recordings:
            graph = graph_of(annotations, recording_id)
            for name in SCORED_CASES:
                once = derive_order_from_anchors(
                    cases[(recording_id, name)].expected, graph, ANCHORS
                )
                twice = derive_order_from_anchors(once, graph, ANCHORS)

                assert twice == once

    def test_the_harness_answers_are_left_untouched_by_the_llm_policy(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        for recording_id in base_recordings:
            graph = graph_of(annotations, recording_id)
            for name in SCORED_CASES:
                expected = cases[(recording_id, name)].expected

                assert apply_order_policy(expected, graph, LLM_ORDER) is expected


class TestTheRuleOnRewrittenExecutions:
    def test_unanchoring_a_prerequisite_removes_it_from_the_timeline(
        self,
        annotations: AnnotationSet,
        base_recordings: list[str],
        cases: dict[tuple[str, str], Case],
    ) -> None:
        """A run that answers "unsure" about a step neither accuses nor is accused."""
        both_start = OrderPolicy(source="anchors", margin="start")
        recording_id = base_recordings[0]
        case = cases[(recording_id, "linked_swap")]
        graph = graph_of(annotations, recording_id)
        match = LINKED_NOTE.search(case.note)
        assert match
        prerequisite = int(match.group(1))
        blinded = replace(
            case.expected,
            steps=tuple(
                replace(
                    step,
                    execution=Execution.INDETERMINATE,
                    anchor=None,
                )
                if step.step_id == prerequisite
                else step
                for step in case.expected.steps
            ),
        )

        derived = derive_order_from_anchors(blinded, graph, both_start)

        assert prerequisite not in violations(derived)

"""The annotation loaders: the tiny cases, and the real annotation files."""

import json
from pathlib import Path

import pytest

from video_agent.task.annotations import (
    MISSING_TAG,
    ORDER_TAG,
    AnnotationSet,
    Sop,
    TaskGraph,
    canonical_sop,
    clean_step,
    load_activities,
    load_recordings,
    load_split,
    load_task_graph,
    load_task_graphs,
    normalize_description,
)

from tests._paths import ANNOTATIONS_DIR, load_annotations

# Counted from the annotation files; a change means the data moved, not the code.
N_RECORDINGS = 384
N_INSTANCES = 5700
N_ORDER_INSTANCES = 794
N_MISSING_INSTANCES = 285
N_ACTIVITIES = 24
N_GRAPH_NODES = 358

CSV_HEADER = '"activity_idx","activity_name","step_index","step_description"\n'


def write_csv(path: Path, rows: list[tuple[int, str, int, str]]) -> Path:
    lines = [CSV_HEADER]
    for activity_id, name, step_id, description in rows:
        lines.append(f'"{activity_id}","{name}","{step_id}","{description}"\n')
    path.write_text("".join(lines), encoding="utf-8")
    return path


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def activity_csv(tmp_path: Path) -> Path:
    return write_csv(
        tmp_path / "steps.csv",
        [
            (1, "Toast", 10, "Slice-Slice the bread"),
            (1, "Toast", 11, "Toast-Toast the bread"),
            (1, "Toast", 12, "Spread-Spread the butter"),
            (1, "Toast", 10, "Slice-Slice the bread"),  # repeated row, as in the file
        ],
    )


@pytest.fixture
def recordings_json(tmp_path: Path) -> Path:
    """Three recordings covering the cases the loader has to get right."""
    return write_json(
        tmp_path / "error_annotations.json",
        [
            {
                "recording_id": "1_1",
                "activity_id": 1,
                "is_error": False,
                "step_annotations": [
                    {
                        "description": "Toast-Toast the bread",
                        "step_id": 11,
                        "start_time": 20.0,
                        "end_time": 30.0,
                    },
                    {
                        "description": "Slice-Slice the bread",
                        "step_id": 10,
                        "start_time": 5.0,
                        "end_time": 10.0,
                    },
                    {
                        "description": "Spread-Spread the butter",
                        "step_id": 12,
                        "start_time": 40.0,
                        "end_time": 50.0,
                    },
                ],
            },
            {
                "recording_id": "1_2",
                "activity_id": 1,
                "is_error": True,
                "step_annotations": [
                    {
                        "description": "Slice-Slice the bread",
                        "step_id": 10,
                        "start_time": 30.0,
                        "end_time": 35.0,
                        "errors": [{"tag": ORDER_TAG, "description": "late"}],
                    },
                    {
                        "description": "Toast-Toast the bread",
                        "step_id": 11,
                        "start_time": 5.0,
                        "end_time": 10.0,
                    },
                    {
                        "description": "Slice-Slice the bread",
                        "step_id": 10,
                        "start_time": 12.0,
                        "end_time": 18.0,
                    },
                ],
            },
            {
                "recording_id": "1_3",
                "activity_id": 1,
                "is_error": True,
                "step_annotations": [
                    {
                        "description": "Slice-Slice the bread",
                        "step_id": 10,
                        "start_time": -1.0,
                        "end_time": -1.0,
                        "errors": [{"tag": MISSING_TAG, "description": "skipped"}],
                    },
                    {
                        "description": "Toast-Toast the bread",
                        "step_id": 11,
                        "start_time": 8.0,
                        "end_time": 12.0,
                        "errors": [
                            {"tag": MISSING_TAG, "description": "barely"},
                            {"tag": MISSING_TAG, "description": "duplicated tag"},
                        ],
                    },
                ],
            },
        ],
    )


class TestCleanStep:
    @pytest.mark.parametrize(
        "description, expected",
        [
            # The text opens with the verb.
            (
                "Coat -Coat a 6-oz. ramekin cup with cooking spray",
                "Coat a 6-oz. ramekin cup with cooking spray",
            ),
            # The text says the verb further in, so the prefix adds nothing.
            (
                "Microwave -Continue to Microwave for 15-30 more seconds or "
                "until the egg is almost set",
                "Continue to Microwave for 15-30 more seconds or until the egg "
                "is almost set",
            ),
            ("dry-gently dry it with a paper/tea towel", "gently dry it with a paper/tea towel"),
            ("remove-then carefully remove the paper liner", "then carefully remove the paper liner"),
            ("stir-then stir the bowl", "then stir the bowl"),
            # The text never says the verb, so joining it keeps the imperative.
            (
                "Add-1/2 tsp baking powder to a blender",
                "Add 1/2 tsp baking powder to a blender",
            ),
            ("Season-pepper on pan to taste", "Season pepper on pan to taste"),
            ("season-pepper to taste", "season pepper to taste"),
            # A compound prefix is matched on its last word.
            (
                "Measure and add-a pinch of salt to the mixing bowl",
                "Measure and add a pinch of salt to the mixing bowl",
            ),
        ],
    )
    def test_the_prefix_goes_only_when_the_text_says_the_verb_itself(
        self, description: str, expected: str
    ) -> None:
        assert clean_step(description) == expected

    def test_leaves_a_description_without_a_prefix_alone(self) -> None:
        assert clean_step("  stir the pot until it thickens ") == (
            "stir the pot until it thickens"
        )


def test_normalize_description_keeps_the_prefix_it_matches_on() -> None:
    assert normalize_description("  Coat -Coat   a CUP ") == "coat -coat a cup"
    # Cleaning first would lose exactly what makes the two files comparable.
    assert normalize_description("Coat -Coat a cup") != normalize_description(
        clean_step("Coat -Coat a cup")
    )


def test_activities_are_read_per_recipe_with_raw_descriptions(
    activity_csv: Path,
) -> None:
    activities = load_activities(activity_csv)

    assert set(activities) == {1}
    assert activities[1].name == "Toast"
    assert activities[1].step_descriptions[10] == "Slice-Slice the bread"


def test_instances_are_ordered_in_time_with_untimed_ones_last(
    recordings_json: Path,
) -> None:
    recordings = load_recordings(recordings_json)

    assert [step.step_id for step in recordings["1_1"].steps] == [10, 11, 12]
    assert [step.step_id for step in recordings["1_2"].steps] == [11, 10, 10]
    assert [step.step_id for step in recordings["1_3"].steps] == [11, 10]


def test_repeated_steps_are_numbered_by_occurrence(recordings_json: Path) -> None:
    steps = load_recordings(recordings_json)["1_2"].steps

    assert [(step.step_id, step.occurrence) for step in steps] == [
        (11, 0),
        (10, 0),
        (10, 1),
    ]
    # The occurrence at 12 s comes first, so the Order Error is the later one.
    assert steps[1].tags == frozenset()
    assert steps[2].tags == {ORDER_TAG}


def test_a_skipped_step_has_no_times_and_keeps_its_tag(recordings_json: Path) -> None:
    skipped = load_recordings(recordings_json)["1_3"].steps[1]

    assert skipped.step_id == 10
    assert skipped.start_s is None and skipped.end_s is None
    assert skipped.tags == {MISSING_TAG}


def test_a_missing_step_with_real_times_keeps_them(recordings_json: Path) -> None:
    timed = load_recordings(recordings_json)["1_3"].steps[0]

    assert (timed.start_s, timed.end_s) == (8.0, 12.0)
    # The same tag twice is one category: the scoring counts instances.
    assert timed.tags == {MISSING_TAG}


def test_instance_descriptions_are_cleaned(recordings_json: Path) -> None:
    assert load_recordings(recordings_json)["1_1"].steps[0].description == (
        "Slice the bread"
    )


def test_repeated_graph_nodes_merge_into_one_step(
    tmp_path: Path, activity_csv: Path
) -> None:
    activity = load_activities(activity_csv)[1]
    path = write_json(
        tmp_path / "toast.json",
        {
            "steps": {
                "0": "START",
                "1": "Slice-Slice   the bread",
                "2": "Toast-Toast the bread",
                "3": "toast-toast the bread",  # same step, executed twice
                "4": "Spread-Spread the butter",
                "5": "END",
            },
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 4], [4, 5]],
        },
    )

    graph = load_task_graph(path, activity)

    assert graph.prerequisites == {
        10: frozenset(),
        11: frozenset({10}),
        12: frozenset({11}),
    }
    # The self-loop the merge creates and the START/END edges are dropped.
    assert graph.edges == frozenset({(10, 11), (11, 12)})


def test_only_the_first_execution_of_a_repeated_step_keeps_its_prerequisites(
    tmp_path: Path, activity_csv: Path
) -> None:
    """The rule that keeps a repeated step from ordering itself.

    Toasting happens twice, the second time after spreading; unioning the two
    nodes' prerequisites would make step 11 require step 12 and step 12
    require step 11.
    """
    activity = load_activities(activity_csv)[1]
    path = write_json(
        tmp_path / "toast.json",
        {
            "steps": {
                "0": "START",
                "1": "Slice-Slice the bread",
                "2": "Toast-Toast the bread",
                "3": "Spread-Spread the butter",
                "4": "toast-toast the bread",  # a second helping
                "5": "END",
            },
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
        },
    )

    graph = load_task_graph(path, activity)

    assert graph.prerequisites == {
        10: frozenset(),
        11: frozenset({10}),
        12: frozenset({11}),
    }
    assert graph.topological_order() == [10, 11, 12]


def test_a_later_execution_still_unlocks_what_waits_on_it(
    tmp_path: Path, activity_csv: Path
) -> None:
    activity = load_activities(activity_csv)[1]
    path = write_json(
        tmp_path / "toast.json",
        {
            "steps": {
                "0": "START",
                "1": "Slice-Slice the bread",
                "2": "slice-slice the bread",  # second execution
                "3": "Toast-Toast the bread",
                "4": "Spread-Spread the butter",
                "5": "END",
            },
            # The second slicing is what the spreading waits for.
            "edges": [[0, 1], [1, 3], [3, 2], [2, 4], [4, 5]],
        },
    )

    graph = load_task_graph(path, activity)

    # Slicing keeps only the first execution's (empty) prerequisites, and its
    # outgoing edges are the union: toasting and spreading both follow it.
    assert graph.prerequisites[10] == frozenset()
    assert graph.prerequisites[11] == frozenset({10})
    assert graph.prerequisites[12] == frozenset({10})


def test_a_cycle_has_no_sop_order(tmp_path: Path) -> None:
    cyclic = TaskGraph(
        activity_id=1,
        prerequisites={10: frozenset({11}), 11: frozenset({10}), 12: frozenset()},
        edges=frozenset({(10, 11), (11, 10)}),
    )

    with pytest.raises(ValueError, match="cycle"):
        cyclic.topological_order()


def test_prerequisites_are_transitive_through_ancestors(
    tmp_path: Path, activity_csv: Path
) -> None:
    activity = load_activities(activity_csv)[1]
    path = write_json(
        tmp_path / "toast.json",
        {
            "steps": {
                "0": "START",
                "1": "Slice-Slice the bread",
                "2": "Toast-Toast the bread",
                "3": "Spread-Spread the butter",
                "4": "END",
            },
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        },
    )

    graph = load_task_graph(path, activity)

    assert graph.ancestors(12) == frozenset({10, 11})
    assert graph.ancestors(10) == frozenset()
    with pytest.raises(ValueError):
        graph.ancestors(99)


def test_a_node_that_matches_no_step_is_a_contract_violation(
    tmp_path: Path, activity_csv: Path
) -> None:
    activity = load_activities(activity_csv)[1]
    path = write_json(
        tmp_path / "toast.json",
        {
            "steps": {"0": "START", "1": "Fry-Fry an egg", "2": "END"},
            "edges": [[0, 1], [1, 2]],
        },
    )

    with pytest.raises(ValueError, match="matches no step"):
        load_task_graph(path, activity)


def test_task_graphs_resolve_to_their_activity_by_description(
    tmp_path: Path, activity_csv: Path
) -> None:
    activities = load_activities(activity_csv)
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    write_json(
        graphs_dir / "toast.json",
        {
            "steps": {"0": "START", "1": "Slice-Slice the bread", "2": "END"},
            "edges": [[0, 1], [1, 2]],
        },
    )

    graphs = load_task_graphs(graphs_dir, activities)

    assert set(graphs) == {1}


def test_a_graph_no_activity_owns_is_refused(tmp_path: Path, activity_csv: Path) -> None:
    activities = load_activities(activity_csv)
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    write_json(
        graphs_dir / "omelette.json",
        {"steps": {"0": "START", "1": "Fry-Fry an egg", "2": "END"}, "edges": []},
    )

    with pytest.raises(ValueError, match="0 activities match"):
        load_task_graphs(graphs_dir, activities)


def test_the_sop_is_a_linear_extension_of_the_task_graph(
    tmp_path: Path, activity_csv: Path
) -> None:
    activities = load_activities(activity_csv)
    path = write_json(
        tmp_path / "toast.json",
        {
            "steps": {
                "0": "START",
                "1": "Slice-Slice the bread",
                "2": "Toast-Toast the bread",
                "3": "Spread-Spread the butter",
                "4": "END",
            },
            "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        },
    )
    graph = load_task_graph(path, activities[1])

    sop = canonical_sop(activities[1], graph, [])

    assert sop == Sop(
        activity_id=1,
        recipe_name="Toast",
        steps=(
            (10, "Slice the bread"),
            (11, "Toast the bread"),
            (12, "Spread the butter"),
        ),
    )


def test_unconstrained_steps_follow_the_median_position_of_normal_recordings(
    tmp_path: Path, activity_csv: Path, recordings_json: Path
) -> None:
    activities = load_activities(activity_csv)
    path = write_json(
        tmp_path / "toast.json",
        {
            "steps": {
                "0": "START",
                "1": "Slice-Slice the bread",
                "2": "Toast-Toast the bread",
                "3": "Spread-Spread the butter",
                "4": "END",
            },
            # Nothing but START/END constrains the three steps.
            "edges": [[0, 1], [0, 2], [0, 3], [1, 4], [2, 4], [3, 4]],
        },
    )
    graph = load_task_graph(path, activities[1])
    recordings = load_recordings(recordings_json)

    without_votes = canonical_sop(activities[1], graph, [])
    with_votes = canonical_sop(activities[1], graph, recordings.values())

    # No evidence: ties fall back to the step id.
    assert [step_id for step_id, _ in without_votes.steps] == [10, 11, 12]
    # The one normal recording slices, then toasts, then spreads; the error
    # recordings, which toast first, do not vote.
    assert [step_id for step_id, _ in with_votes.steps] == [10, 11, 12]

    reversed_normal = write_json(
        tmp_path / "reversed.json",
        [
            {
                "recording_id": "1_9",
                "activity_id": 1,
                "is_error": False,
                "step_annotations": [
                    {
                        "description": "Spread-Spread the butter",
                        "step_id": 12,
                        "start_time": 1.0,
                        "end_time": 2.0,
                    },
                    {
                        "description": "Toast-Toast the bread",
                        "step_id": 11,
                        "start_time": 3.0,
                        "end_time": 4.0,
                    },
                    {
                        "description": "Slice-Slice the bread",
                        "step_id": 10,
                        "start_time": 5.0,
                        "end_time": 6.0,
                    },
                ],
            }
        ],
    )
    flipped = canonical_sop(
        activities[1], graph, load_recordings(reversed_normal).values()
    )

    assert [step_id for step_id, _ in flipped.steps] == [12, 11, 10]


def test_load_split_returns_the_error_and_normal_ids(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "dev.json",
        {"split": "dev", "error": ["1_2", "1_3"], "normal": ["1_1"], "n_error": 2},
    )

    assert load_split(path) == (["1_2", "1_3"], ["1_1"])


@pytest.fixture(scope="module")
def annotations() -> AnnotationSet:
    return load_annotations()


class TestRealAnnotations:
    """Pins the facts about the real files that the task layer relies on."""

    def test_every_recording_and_instance_is_loaded(
        self, annotations: AnnotationSet
    ) -> None:
        instances = [
            step for recording in annotations.recordings.values() for step in recording.steps
        ]

        assert len(annotations.recordings) == N_RECORDINGS
        assert len(instances) == N_INSTANCES

    def test_the_two_scored_categories_have_the_expected_instance_counts(
        self, annotations: AnnotationSet
    ) -> None:
        instances = [
            step for recording in annotations.recordings.values() for step in recording.steps
        ]

        # Instances, not tag occurrences: one instance carries Order Error twice.
        assert sum(ORDER_TAG in step.tags for step in instances) == N_ORDER_INSTANCES
        assert sum(MISSING_TAG in step.tags for step in instances) == N_MISSING_INSTANCES

    def test_every_task_graph_resolves_to_a_distinct_activity(
        self, annotations: AnnotationSet
    ) -> None:
        files = sorted((ANNOTATIONS_DIR / "task_graphs").glob("*.json"))

        assert len(files) == N_ACTIVITIES
        assert len(annotations.graphs) == N_ACTIVITIES
        assert set(annotations.graphs) == set(annotations.activities)

    def test_every_graph_node_matches_exactly_one_step_of_its_activity(
        self, annotations: AnnotationSet
    ) -> None:
        nodes = 0
        for path in sorted((ANNOTATIONS_DIR / "task_graphs").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            nodes += sum(
                1
                for description in payload["steps"].values()
                if description not in ("START", "END")
            )

        # ``AnnotationSet.load`` already raised if any node had failed to match.
        assert nodes == N_GRAPH_NODES

    def test_every_sop_covers_every_step_of_its_recipe_exactly_once(
        self, annotations: AnnotationSet
    ) -> None:
        for activity_id, sop in annotations.sops.items():
            step_ids = [step_id for step_id, _ in sop.steps]
            expected = set(annotations.activities[activity_id].step_descriptions)

            assert len(step_ids) == len(set(step_ids)) == len(expected)
            assert set(step_ids) == expected

    def test_sop_descriptions_carry_no_verb_prefix(
        self, annotations: AnnotationSet
    ) -> None:
        for sop in annotations.sops.values():
            for _, description in sop.steps:
                assert description == description.strip()
                assert not description.startswith("-")

    def test_every_task_graph_is_acyclic_once_repeated_steps_are_merged(
        self, annotations: AnnotationSet
    ) -> None:
        for activity_id, graph in annotations.graphs.items():
            order = graph.topological_order()

            assert len(order) == len(graph.prerequisites)

    @pytest.mark.parametrize("activity_id", [2, 20])
    def test_the_recipes_that_repeat_a_step_have_no_mutual_constraints(
        self, annotations: AnnotationSet, activity_id: int
    ) -> None:
        graph = annotations.graphs[activity_id]

        assert [(a, b) for a, b in graph.edges if (b, a) in graph.edges] == []
        for step_id, prerequisites in graph.prerequisites.items():
            assert step_id not in graph.ancestors(step_id)
            assert step_id not in prerequisites

    def test_every_sop_places_every_prerequisite_before_the_step(
        self, annotations: AnnotationSet
    ) -> None:
        for activity_id, sop in annotations.sops.items():
            position = {step_id: index for index, (step_id, _) in enumerate(sop.steps)}

            for prerequisite, step in annotations.graphs[activity_id].edges:
                assert position[prerequisite] < position[step]

    def test_every_recording_can_be_resolved_to_its_sop_and_graph(
        self, annotations: AnnotationSet
    ) -> None:
        for recording_id in annotations.recordings:
            sop, graph = annotations.for_recording(recording_id)

            assert sop.activity_id == graph.activity_id

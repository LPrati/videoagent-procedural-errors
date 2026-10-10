"""Read CaptainCook4D's annotations into the entities the task layer needs.

Four files describe a recipe and its recordings, and none of them is quite
the SOP the task needs: the CSV lists a recipe's steps without an
order, the task graph orders them but numbers its nodes locally and repeats a
node when a step is executed twice, and the recordings hold the ground truth
the run must not see. This module resolves all of that into ``Activity``,
``Recording``, ``TaskGraph`` and ``Sop``, and does it programmatically —
the graph files are named after the recipe, not the activity id, and the link
between the two is recovered by matching step descriptions rather than by a
hand-written table that could silently rot.

The descriptions carry a ``"Verb -Text"`` prefix with inconsistent spacing
and capitalization; ``clean_step`` strips it for everything the model reads
while ``normalize_description`` works on the raw text because
that is what the graph files and the CSV have in common.
"""

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Mapping

ORDER_TAG = "Order Error"
"""Ground-truth tag of an order violation (``error_category_idx.json``)."""

MISSING_TAG = "Missing Step"
"""Ground-truth tag of a missing step (``error_category_idx.json``)."""

START_NODE = "START"
END_NODE = "END"
"""Sentinel nodes of a task graph; they are not steps and carry no step id."""

ACTIVITY_STEPS_CSV = "annotation_csv/activity_step_description.csv"
ERROR_ANNOTATIONS_JSON = "annotation_json/error_annotations.json"
TASK_GRAPHS_DIR = "task_graphs"

_STEP_PREFIX = re.compile(r"^\s*([A-Za-z][A-Za-z ]*?)\s*-\s*(.*)$")
_WHITESPACE = re.compile(r"\s+")


def clean_step(description: str) -> str:
    """Strip the ``"Verb -Text"`` prefix from a step description.

    The prefix is a formatting artifact of the annotation, not part of the
    instruction, so it is removed before the description is used as SOP
    text or as a query. What decides whether the verb
    is dropped is whether the text still says what to do: the prefix goes
    when the text repeats the verb, either at its start ("Coat -Coat a 6-oz.
    ramekin cup") or further in ("Microwave -Continue to Microwave..."), and
    it is joined to the text when it does not, so the step keeps an
    imperative ("Add-1/2 tsp baking powder...", "Season-pepper to taste").
    Only the last word of the verb is looked for, because a compound prefix
    names the action with it ("Measure and add-a pinch of salt...").
    """
    match = _STEP_PREFIX.match(description)
    if match is None:
        return description.strip()
    verb, rest = match.groups()
    if rest.lower().startswith(verb.lower()):
        return rest
    last_word = verb.split()[-1]
    if re.search(rf"\b{re.escape(last_word)}", rest, re.IGNORECASE):
        return rest
    return f"{verb} {rest}"


def normalize_description(description: str) -> str:
    """Collapse whitespace and case, keeping the raw ``"Verb -Text"`` prefix.

    This is the key that matches a task-graph node to a CSV row, and both
    sides carry the prefix: cleaning first would merge steps that differ only
    in their verb and would make the match depend on ``clean_step``'s rules.
    """
    return _WHITESPACE.sub(" ", description.strip()).lower()


@dataclass(frozen=True)
class StepInstance:
    """One occurrence of a step in one recording, with its ground truth.

    ``start_s``/``end_s`` are ``None`` for a step that was never performed:
    the annotation writes ``-1`` for both, and those instances are excluded
    from every localization metric.
    """

    recording_id: str
    step_id: int
    occurrence: int  # 0, 1, ... among the instances of this step id, in time
    description: str  # cleaned
    start_s: float | None
    end_s: float | None
    tags: frozenset[str]


@dataclass(frozen=True)
class Recording:
    """One recorded attempt at a recipe, with its annotated step instances.

    ``(recording_id, step_id)`` is not a key: a step performed twice yields
    two instances, distinguished by ``occurrence`` in temporal order.
    """

    recording_id: str
    activity_id: int
    is_error: bool
    steps: tuple[StepInstance, ...]


@dataclass(frozen=True)
class Sop:
    """A recipe's steps in canonical order, as the model receives them."""

    activity_id: int
    recipe_name: str
    steps: tuple[tuple[int, str], ...]  # (step_id, cleaned description)

    def step_ids(self) -> tuple[int, ...]:
        return tuple(step_id for step_id, _ in self.steps)


@dataclass(frozen=True)
class TaskGraph:
    """A recipe's ordering constraints: which steps must precede which.

    ``prerequisites`` has an entry for every step of the activity, empty when
    the step depends on nothing, so a caller can iterate the recipe from the
    graph alone. This is recipe structure, not video annotation, and is
    therefore legitimate input to a run.
    """

    activity_id: int
    prerequisites: dict[int, frozenset[int]]
    edges: frozenset[tuple[int, int]]  # (prerequisite, step)

    def ancestors(self, step_id: int) -> frozenset[int]:
        """Return every step that must transitively precede ``step_id``."""
        if step_id not in self.prerequisites:
            raise ValueError(
                f"step {step_id} is not in activity {self.activity_id}"
            )
        seen: set[int] = set()
        stack = list(self.prerequisites[step_id])
        while stack:
            step = stack.pop()
            if step in seen:
                continue
            seen.add(step)
            stack.extend(self.prerequisites.get(step, frozenset()))
        return frozenset(seen)

    def topological_order(
        self, key: Callable[[int], float] | None = None
    ) -> list[int]:
        """Order the steps so prerequisites come first, breaking ties by ``key``.

        Kahn's algorithm over a priority queue: among the steps whose
        prerequisites are all placed, the one with the smallest ``key`` wins
        and the step id decides the rest, so the order is a function of the
        data and not of dict iteration.

        A cycle is a contract violation rather than something to recover
        from: the SOP is an ordered list, and a recipe whose constraints
        cannot be satisfied has no ordered list. ``load_task_graph``'s merge
        rule is what keeps the graphs acyclic.
        """
        rank: Callable[[int], float] = (lambda step: 0.0) if key is None else key
        indegree = {step: 0 for step in self.prerequisites}
        dependents: dict[int, list[int]] = {step: [] for step in self.prerequisites}
        for prerequisite, step in self.edges:
            indegree[step] += 1
            dependents[prerequisite].append(step)

        queued = {step for step, degree in indegree.items() if degree == 0}
        ready = [(rank(step), step) for step in queued]
        heapify(ready)
        order: list[int] = []
        while ready:
            _, step = heappop(ready)
            order.append(step)
            for dependent in dependents[step]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0 and dependent not in queued:
                    queued.add(dependent)
                    heappush(ready, (rank(dependent), dependent))
        if len(order) != len(indegree):
            unplaced = sorted(set(indegree) - set(order))
            raise ValueError(
                f"activity {self.activity_id} has a cycle through steps {unplaced}"
            )
        return order


@dataclass(frozen=True)
class Activity:
    """One recipe: its name and the raw description of each of its steps.

    Step ids are global in the annotation but not unique across activities
    (three ids are shared by two recipes each), so descriptions are always
    looked up per activity.
    """

    activity_id: int
    name: str
    step_descriptions: dict[int, str]  # raw, prefix included


def load_activities(csv_path: Path) -> dict[int, Activity]:
    """Read the recipe catalogue from ``activity_step_description.csv``.

    The file has one row per step *instance*, so the same step repeats many
    times; the descriptions agree across rows and the last one wins.
    """
    names: dict[int, str] = {}
    steps: dict[int, dict[int, str]] = defaultdict(dict)
    with Path(csv_path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            activity_id = int(row["activity_idx"])
            names[activity_id] = row["activity_name"]
            steps[activity_id][int(row["step_index"])] = row["step_description"]
    return {
        activity_id: Activity(
            activity_id=activity_id,
            name=name,
            step_descriptions=dict(steps[activity_id]),
        )
        for activity_id, name in names.items()
    }


def _instance_sort_key(annotation: Mapping) -> tuple[int, float]:
    """Sort key placing timed instances first, in time; untimed ones last."""
    start = annotation.get("start_time")
    if start is None or float(start) < 0.0:
        return (1, 0.0)
    return (0, float(start))


def load_recordings(error_annotations: Path) -> dict[str, Recording]:
    """Read every recorded attempt from ``error_annotations.json``.

    Instances are ordered by annotated start time so that ``occurrence``
    numbers the repetitions of a step chronologically, which is how a verdict
    is matched to an annotation. A step that was never performed has no
    time; those instances keep the file's own order at the end.
    """
    payload = json.loads(Path(error_annotations).read_text(encoding="utf-8"))
    recordings: dict[str, Recording] = {}
    for entry in payload:
        recording_id = str(entry["recording_id"])
        annotations = sorted(entry["step_annotations"], key=_instance_sort_key)
        seen: dict[int, int] = defaultdict(int)
        instances: list[StepInstance] = []
        for annotation in annotations:
            step_id = int(annotation["step_id"])
            start = float(annotation["start_time"])
            end = float(annotation["end_time"])
            timed = start >= 0.0 and end >= 0.0
            occurrence = seen[step_id]
            seen[step_id] += 1
            instances.append(
                StepInstance(
                    recording_id=recording_id,
                    step_id=step_id,
                    occurrence=occurrence,
                    description=clean_step(annotation["description"]),
                    start_s=start if timed else None,
                    end_s=end if timed else None,
                    # One instance repeats a tag; a set counts the categories
                    # the instance belongs to, which is what is scored.
                    tags=frozenset(
                        error["tag"] for error in annotation.get("errors") or ()
                    ),
                )
            )
        recordings[recording_id] = Recording(
            recording_id=recording_id,
            activity_id=int(entry["activity_id"]),
            is_error=bool(entry["is_error"]),
            steps=tuple(instances),
        )
    return recordings


def _graph_descriptions(payload: Mapping) -> dict[int, str]:
    """Return the graph's real nodes, dropping the START/END sentinels."""
    return {
        int(node): description
        for node, description in payload["steps"].items()
        if description not in (START_NODE, END_NODE)
    }


def _local_ancestors(
    nodes: Iterable[int], edges: Iterable[tuple[int, int]]
) -> dict[int, frozenset[int]]:
    """Transitive predecessors of every node of the graph file, as written."""
    parents: dict[int, set[int]] = {node: set() for node in nodes}
    for source, target in edges:
        parents[target].add(source)
    ancestors: dict[int, frozenset[int]] = {}
    for node in parents:
        seen: set[int] = set()
        stack = list(parents[node])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(parents[current])
        ancestors[node] = frozenset(seen)
    return ancestors


def _elect_first_executions(
    local_to_global: Mapping[int, int], ancestors: Mapping[int, frozenset[int]]
) -> dict[int, int]:
    """Pick, per global step, the node that stands for its first execution.

    Fewest ancestors first, then the lowest local id, so the choice depends
    only on the graph and not on iteration order.
    """
    grouped: dict[int, list[int]] = defaultdict(list)
    for node, step_id in local_to_global.items():
        grouped[step_id].append(node)
    return {
        step_id: min(nodes, key=lambda node: (len(ancestors[node]), node))
        for step_id, nodes in grouped.items()
    }


def load_task_graph(path: Path, activity: Activity) -> TaskGraph:
    """Build the prerequisite relation of ``activity`` from its graph file.

    Node ids are local to the file, so each node is mapped to a global step
    id by its description. A recipe that executes one step twice has several
    nodes with the same description, and they are one SOP step: the merged
    step takes the incoming edges of the *first* execution only — the node
    with the fewest ancestors — and the outgoing edges of all of them. That
    is the reading the SOP needs: what a constraint "before this step" refers
    to is the first time the step is done, while every later execution still
    unlocks whatever waits on it.

    Merging this way cannot create a cycle. Suppose the merged graph had a
    path from a successor of some node ``A_i`` back to a predecessor of the
    elected ``A_e``: since only ``A_e`` keeps incoming edges, that path plus
    the original edges would make every ancestor of ``A_e`` an ancestor of
    ``A_i`` and add at least one more, so ``A_i`` would have fewer ancestors
    than ``A_e`` — contradicting how ``A_e`` was chosen. This holds for all
    24 CaptainCook4D task graphs, three of which repeat a step.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    by_description = {
        normalize_description(description): step_id
        for step_id, description in activity.step_descriptions.items()
    }
    local_to_global: dict[int, int] = {}
    for node, description in _graph_descriptions(payload).items():
        step_id = by_description.get(normalize_description(description))
        if step_id is None:
            raise ValueError(
                f"{Path(path).name}: node {node} {description!r} matches no step "
                f"of activity {activity.activity_id} ({activity.name})"
            )
        local_to_global[node] = step_id

    local_edges = [(int(source), int(target)) for source, target in payload["edges"]]
    # Ancestors are counted on the file's own graph, sentinels included: they
    # rank the executions of a repeated step by how far into the recipe they
    # are, which is exactly the "first execution" the merge elects.
    first_execution = _elect_first_executions(
        local_to_global,
        _local_ancestors(
            {int(node) for node in payload["steps"]}
            | {node for edge in local_edges for node in edge},
            local_edges,
        ),
    )

    edges: set[tuple[int, int]] = set()
    for source, target in local_edges:
        prerequisite = local_to_global.get(source)
        step = local_to_global.get(target)
        if prerequisite is None or step is None or prerequisite == step:
            continue
        if target != first_execution[step]:
            continue  # a later execution constrains nothing
        edges.add((prerequisite, step))

    prerequisites: dict[int, set[int]] = {
        step_id: set() for step_id in activity.step_descriptions
    }
    for prerequisite, step in edges:
        prerequisites[step].add(prerequisite)
    return TaskGraph(
        activity_id=activity.activity_id,
        prerequisites={
            step: frozenset(parents) for step, parents in prerequisites.items()
        },
        edges=frozenset(edges),
    )


def load_task_graphs(
    task_graphs_dir: Path, activities: Mapping[int, Activity]
) -> dict[int, TaskGraph]:
    """Load every task graph, resolving each file to the activity it describes.

    The files are named after the recipe and carry no activity id. A graph's
    set of node descriptions is a subset of exactly one activity's step
    descriptions, which identifies the activity without a hand-written table;
    anything else is a mismatch between the annotation files and is refused.
    """
    graphs: dict[int, TaskGraph] = {}
    by_activity = {
        activity_id: {
            normalize_description(description)
            for description in activity.step_descriptions.values()
        }
        for activity_id, activity in activities.items()
    }
    for path in sorted(Path(task_graphs_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        wanted = {
            normalize_description(description)
            for description in _graph_descriptions(payload).values()
        }
        candidates = [
            activity_id
            for activity_id, descriptions in by_activity.items()
            if wanted <= descriptions
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{path.name}: {len(candidates)} activities match its steps "
                f"({sorted(candidates)}), expected exactly 1"
            )
        activity_id = candidates[0]
        if activity_id in graphs:
            raise ValueError(
                f"{path.name}: activity {activity_id} already has a task graph"
            )
        graphs[activity_id] = load_task_graph(path, activities[activity_id])
    return graphs


def canonical_sop(
    activity: Activity, graph: TaskGraph, recordings: Iterable[Recording]
) -> Sop:
    """Order a recipe's steps into the SOP the model is shown.

    The task graph is a partial order, and the model is shown an ordered
    list, so the remaining freedom is resolved by how people
    actually cook: among steps whose prerequisites are equally satisfied, the
    one performed earlier in normal recordings comes first. The position of a
    step in a recording is its rank divided by the number of instances, which
    makes recordings of different lengths comparable without their durations.
    Only normal recordings vote, because a recording with an order error is
    evidence about the error, not about the recipe.
    """
    positions: dict[int, list[float]] = defaultdict(list)
    for recording in recordings:
        if recording.activity_id != activity.activity_id or recording.is_error:
            continue
        total = len(recording.steps)
        if total == 0:
            continue
        for rank, instance in enumerate(recording.steps):
            positions[instance.step_id].append(rank / total)

    def observed_position(step_id: int) -> float:
        # A step no normal recording performed has no evidence; it sorts last
        # among its ties rather than silently first.
        values = positions.get(step_id)
        return median(values) if values else float("inf")

    order = graph.topological_order(key=observed_position)
    return Sop(
        activity_id=activity.activity_id,
        recipe_name=activity.name,
        steps=tuple(
            (step_id, clean_step(activity.step_descriptions[step_id]))
            for step_id in order
        ),
    )


def load_split(path: Path) -> tuple[list[str], list[str]]:
    """Return the ``(error, normal)`` recording ids of a split file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        [str(recording_id) for recording_id in payload.get("error", [])],
        [str(recording_id) for recording_id in payload.get("normal", [])],
    )


@dataclass(frozen=True)
class AnnotationSet:
    """Everything the task layer reads from the annotation directory.

    Assembled once per process: resolving the task graphs and the canonical
    SOPs costs a pass over all 384 recordings, and every recording of a run
    needs the same four tables.
    """

    activities: dict[int, Activity]
    recordings: dict[str, Recording]
    graphs: dict[int, TaskGraph]
    sops: dict[int, Sop]

    @classmethod
    def load(cls, annotations_dir: Path) -> "AnnotationSet":
        """Read the CaptainCook4D annotations from ``annotations_dir``."""
        root = Path(annotations_dir)
        activities = load_activities(root / ACTIVITY_STEPS_CSV)
        recordings = load_recordings(root / ERROR_ANNOTATIONS_JSON)
        graphs = load_task_graphs(root / TASK_GRAPHS_DIR, activities)
        sops = {
            activity_id: canonical_sop(
                activities[activity_id], graph, recordings.values()
            )
            for activity_id, graph in graphs.items()
        }
        return cls(
            activities=activities, recordings=recordings, graphs=graphs, sops=sops
        )

    def for_recording(self, recording_id: str) -> tuple[Sop, TaskGraph]:
        """Return the SOP and task graph of the recipe ``recording_id`` attempts."""
        recording = self.recordings.get(recording_id)
        if recording is None:
            raise ValueError(f"unknown recording {recording_id!r}")
        activity_id = recording.activity_id
        if activity_id not in self.sops:
            raise ValueError(
                f"recording {recording_id!r} needs activity {activity_id}, "
                f"which has no task graph"
            )
        return self.sops[activity_id], self.graphs[activity_id]

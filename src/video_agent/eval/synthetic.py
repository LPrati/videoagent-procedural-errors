"""Synthetic harness: check the evaluation pipeline against known answers.

A synthetic memory is what a *perfect* perception layer would have produced:
one caption per segment, equal to the description of the step being
performed at that time, taken from the annotation of a normal recording.
Feeding it to the real agent separates two failures that must not be
confused: a system that cannot see (perception) and one that sees and still
reasons wrongly (the orchestrator and the decision rule).

Seven cases perturb one normal recording: nothing, a missing step, a swap of
two linked steps, a swap of two independent ones, both errors at once, a
repeated step, and a fully shuffled timeline. Each case carries the modified
annotation, which *is* its ground truth, and the verdicts a perfect system
would emit for it, so the metrics can be checked against numbers worked out
by hand.

One rule decides everything: ``derive_order_truth``, the direct-prerequisite
rule of ``violated_prerequisites`` (a step violates the order when one of its
direct prerequisites in the task graph started after it or was not
performed). The error tags of the modified annotation and the expected
verdicts are both read off it, so an expected verdict and a tag can never
disagree. ``scripts/score_synthetic.py`` re-scores the cases with the order
definition of the task question.
"""

import math
import random
import re
import zlib
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from video_agent.media import Clip, Video
from video_agent.memory.preprocess import SECONDS_PER_SEGMENT, get_video_segments
from video_agent.memory.store import segment_interval
from video_agent.memory.types import Segment, TemporalMemory
from video_agent.models.interfaces import TextEmbedder, VideoTextAligner
from video_agent.task.annotations import (
    MISSING_TAG,
    ORDER_TAG,
    Recording,
    Sop,
    StepInstance,
    TaskGraph,
)
from video_agent.task.verdicts import (
    Anchor,
    Execution,
    Order,
    RecordingVerdicts,
    StepVerdict,
)

IDLE_CAPTION = "The person is standing in the kitchen."
"""Caption of a segment no annotated step covers.

Deliberately uninformative: the gaps between annotated steps are real time in
the video, and filling them with silence would tell the agent where the steps
are not.
"""

SYNTHETIC_SCHEME = "synthetic:"
"""First component of the fake video path, so a harness record is never mistaken
for a run over a real file.

A path component and not a URL scheme: ``Path`` collapses the ``//`` of
``"synthetic://1_25"`` into a single separator, so the marker would silently
stop being the prefix it was written as. ``Path(SYNTHETIC_SCHEME, recording_id)``
round-trips instead, and ``is_synthetic_video`` reads it back by component
rather than by string prefix.
"""

DEFAULT_SEED = 20260903
"""Fixed seed of the case choices, so the cases are reproducible."""

CASE_NAMES: tuple[str, ...] = (
    "normal",
    "omission",
    "linked_swap",
    "unlinked_swap",
    "omission_and_swap",
    "repeated_step",
    "shuffled",
)

CASE_LETTERS: dict[str, str] = dict(zip("abcdefg", CASE_NAMES))
"""Short letters (a)-(g) naming the cases, for the harness CLI."""

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingTextEmbedder:
    """Bag-of-words hashing embedder — the stand-in for a real text tower.

    The harness must run without a GPU and without downloading a checkpoint,
    but the agent's retrieval tool still has to behave like retrieval: a
    query about a step should rank that step's segments first. Hashing the
    words into a fixed space and normalizing gives exactly that — cosine
    similarity becomes word overlap — with no model and no randomness.
    ``zlib.crc32`` rather than ``hash`` because Python randomizes string
    hashing per process, and a harness that changes answers between runs
    would be useless as a test.

    Fills both roles the toolkit needs: the caption embedder and the
    video/text aligner's text side. Encoding video is refused rather than
    faked: synthetic memories are built from captions, and a silent zero
    vector here would look like a working visual channel.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            vector[zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized bag-of-words embeddings, one row per text."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(text) for text in texts])

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Same space as ``embed``; the aligner's text tower for the harness."""
        return self.embed(texts)

    def encode_video(self, clips: Sequence[Clip]) -> np.ndarray:
        """Refuse: the harness has no pixels, only captions."""
        raise ValueError(
            "HashingTextEmbedder cannot encode video: synthetic memories are "
            "built from captions, not from frames"
        )


def synthetic_video(
    recording: Recording,
    *,
    fps: float = 30.0,
    seconds_per_segment: float = SECONDS_PER_SEGMENT,
) -> Video:
    """Describe the video a recording would have, without any file behind it.

    Only the metadata matters: ``get_video_segments`` derives the whole
    segment grid from fps and frame count, and the harness never decodes a
    frame. The video runs one segment past the last annotated end so the last
    step is not sitting on the boundary of a grid that drops its trailing
    partial segment.
    """
    frames_per_segment = int(round(fps) * seconds_per_segment)
    if frames_per_segment <= 0:
        raise ValueError(f"segment shorter than one frame at {fps} fps")
    ends = [step.end_s for step in recording.steps if step.end_s is not None]
    last_end = max(ends) if ends else 0.0
    segment_s = frames_per_segment / fps
    n_segments = math.ceil(last_end / segment_s) + 1
    return Video(
        path=Path(SYNTHETIC_SCHEME, recording.recording_id),
        fps=fps,
        frame_count=n_segments * frames_per_segment,
    )


def is_synthetic_video(video: Video) -> bool:
    """Whether ``video`` is one of the harness's file-less videos.

    Reads the marker as the first path component, which is how
    ``synthetic_video`` writes it; a caller that decodes frames can refuse
    such a video instead of failing on a path that does not exist.
    """
    parts = video.path.parts
    return bool(parts) and parts[0] == SYNTHETIC_SCHEME


def _covering_step(
    recording: Recording, sop: Sop, time_s: float
) -> StepInstance | None:
    """The annotated instance covering ``time_s``, first in SOP order on a tie."""
    position = {step_id: index for index, (step_id, _) in enumerate(sop.steps)}
    covering = [
        step
        for step in recording.steps
        if step.start_s is not None
        and step.end_s is not None
        and step.start_s <= time_s < step.end_s
    ]
    if not covering:
        return None
    return min(
        covering,
        key=lambda step: (
            position.get(step.step_id, len(position)),
            step.step_id,
            step.occurrence,
        ),
    )


def synthetic_temporal_memory(
    recording: Recording,
    sop: Sop,
    *,
    video: Video,
    text_embedder: TextEmbedder,
    aligner: VideoTextAligner,
    idle_caption: str = IDLE_CAPTION,
    seconds_per_segment: float = SECONDS_PER_SEGMENT,
) -> TemporalMemory:
    """Build the memory a perfect captioner would have produced.

    A segment is described by the step being performed at its midpoint —
    midpoint rather than overlap so every segment gets exactly one caption,
    and the annotated boundaries decide which. Both embedding channels come
    from the captions: there are no pixels, so the aligner's *text* tower
    stands in for its visual one, which is what makes the retrieval tool
    return the segments whose step a query names.
    """
    segments = get_video_segments(video, seconds_per_segment)
    descriptions = dict(sop.steps)
    captions: list[str] = []
    for segment in segments:
        step = _covering_step(recording, sop, (segment.start_s + segment.end_s) / 2.0)
        if step is None:
            captions.append(idle_caption)
        else:
            captions.append(descriptions.get(step.step_id, step.description))
    return TemporalMemory(
        segments=segments,
        captions=captions,
        caption_embeddings=np.asarray(text_embedder.embed(captions), dtype=np.float32),
        visual_embeddings=np.asarray(aligner.encode_text(captions), dtype=np.float32),
    )


def first_starts(recording: Recording) -> dict[int, float | None]:
    """Start of each step's first occurrence; ``None`` when it has no times.

    The task graph constrains the first execution of a step (that is how the
    graph's repeated nodes are merged), so this is the timeline the order
    rule reads.
    """
    starts: dict[int, float | None] = {}
    for step in recording.steps:
        if step.step_id not in starts or starts[step.step_id] is None:
            starts[step.step_id] = step.start_s
    return starts


def violated_prerequisites(
    step_id: int, starts: Mapping[int, float | None], graph: TaskGraph
) -> list[int]:
    """Prerequisites of ``step_id`` that make it an order violation.

    A prerequisite violates the order when it was performed after the step or
    was not performed at all. They are returned worst-first — never performed
    before merely late, and the latest of the late first — so a citation that
    must name one names the most flagrant.
    """
    start = starts.get(step_id)
    if start is None:
        return []
    offenders: list[tuple[int, float, int]] = []
    for prerequisite in graph.prerequisites.get(step_id, frozenset()):
        prerequisite_start = starts.get(prerequisite)
        if prerequisite_start is None:
            offenders.append((0, 0.0, prerequisite))
        elif prerequisite_start > start:
            offenders.append((1, -prerequisite_start, prerequisite))
    return [prerequisite for _, _, prerequisite in sorted(offenders)]


def derive_order_truth(
    recording_times: Mapping[int, float | None], graph: TaskGraph
) -> set[int]:
    """The steps that are order violations on this timeline.

    A step *k* violates the order when at least one direct prerequisite of *k*
    was performed after *k*, or was not performed at all. Everything the
    harness asserts is read off this function: the ``Order Error`` tags of the
    modified annotation and the verdicts a perfect system would emit. A step
    with no times is not executed, so it has no order to violate.
    """
    return {
        step_id
        for step_id in recording_times
        if violated_prerequisites(step_id, recording_times, graph)
    }


def is_order_clean(recording: Recording, graph: TaskGraph) -> bool:
    """Whether the annotated times of ``recording`` respect the task graph.

    Not every recording the annotation calls normal does: some normal
    recordings perform a step before a prerequisite without the annotators
    tagging it. Such a recording is a fine video and a bad base
    for case (a), whose expectation is "no error at all", so the harness
    filters on this.
    """
    return not derive_order_truth(first_starts(recording), graph)


def _segment_span(
    segments: Sequence[Segment], start_s: float, end_s: float
) -> tuple[int, int]:
    """Segment ids covering ``[start_s, end_s)``, as the model would answer."""
    starts = [segment.start_s for segment in segments]
    first = max(bisect_right(starts, start_s) - 1, 0)
    last = min(max(bisect_left(starts, end_s) - 1, first), len(segments) - 1)
    return first, last


def _anchor(segments: Sequence[Segment], start_s: float, end_s: float) -> Anchor:
    """The anchor a perfect system would give for an annotated interval.

    Answers are segment ranges (the anchor is requested as segment ids and
    converted to seconds), so the expected anchor is the annotated interval
    rounded out to the grid. That rounding is the only error the
    localization metric sees in a perfect system.
    """
    first, last = _segment_span(segments, start_s, end_s)
    anchor_start, anchor_end = segment_interval(segments, first, last)
    return Anchor(start_s=anchor_start, end_s=anchor_end, segments=(first, last))


@dataclass
class _Slot:
    """One annotated instance while a case is being built (mutable on purpose)."""

    step_id: int
    start_s: float | None
    end_s: float | None
    tags: frozenset[str]


def _slots(recording: Recording) -> list[_Slot]:
    return [
        _Slot(
            step_id=step.step_id,
            start_s=step.start_s,
            end_s=step.end_s,
            tags=step.tags,
        )
        for step in recording.steps
    ]


def _rebuild(
    recording: Recording,
    slots: Sequence[_Slot],
    graph: TaskGraph,
    *,
    omitted: frozenset[int] = frozenset(),
    derive_tags: bool = True,
) -> Recording:
    """Rebuild a recording from modified slots, re-deriving its error tags.

    Instances are re-sorted and re-numbered exactly as ``load_recordings``
    does — by start time, untimed last, ``occurrence`` counted in that order —
    because the whole evaluation matches verdicts to occurrences that way. The
    tags are then written from ``derive_order_truth`` and from the missing
    steps the case injected, never by hand, and they land on the first occurrence of
    the step, the one the order rule looked at.
    """
    descriptions = {step.step_id: step.description for step in recording.steps}
    ordered = sorted(
        slots, key=lambda slot: (slot.start_s is None, slot.start_s or 0.0)
    )
    starts: dict[int, float | None] = {}
    for slot in ordered:
        if slot.step_id not in starts or starts[slot.step_id] is None:
            starts[slot.step_id] = slot.start_s
    violations = derive_order_truth(starts, graph) if derive_tags else set()

    seen: dict[int, int] = {}
    instances: list[StepInstance] = []
    for slot in ordered:
        occurrence = seen.get(slot.step_id, 0)
        seen[slot.step_id] = occurrence + 1
        tags = set(slot.tags) - {ORDER_TAG, MISSING_TAG}
        if derive_tags and occurrence == 0:
            if slot.step_id in omitted:
                tags.add(MISSING_TAG)
            if slot.step_id in violations:
                tags.add(ORDER_TAG)
        instances.append(
            StepInstance(
                recording_id=recording.recording_id,
                step_id=slot.step_id,
                occurrence=occurrence,
                description=descriptions.get(slot.step_id, ""),
                start_s=slot.start_s,
                end_s=slot.end_s,
                tags=frozenset(tags),
            )
        )
    return Recording(
        recording_id=recording.recording_id,
        activity_id=recording.activity_id,
        is_error=any(instance.tags for instance in instances),
        steps=tuple(instances),
    )


def expected_verdicts(
    recording: Recording, sop: Sop, graph: TaskGraph, segments: Sequence[Segment]
) -> RecordingVerdicts:
    """What a perfect system would answer about ``recording``.

    One verdict per SOP step, in SOP order, exactly as ``verdicts.normalize``
    produces them: executed with the anchor of the first occurrence, not
    executed when the step has no times, and — read off the same
    ``derive_order_truth`` the tags come from — an order violation citing the
    prerequisite responsible, with that prerequisite's anchor when it has one.
    """
    starts = first_starts(recording)
    intervals: dict[int, tuple[float, float]] = {}
    for step in recording.steps:
        if (
            step.step_id not in intervals
            and step.start_s is not None
            and step.end_s is not None
        ):
            intervals[step.step_id] = (step.start_s, step.end_s)
    anchors = {
        step_id: _anchor(segments, start, end)
        for step_id, (start, end) in intervals.items()
    }

    steps: list[StepVerdict] = []
    for step_id, _ in sop.steps:
        anchor = anchors.get(step_id)
        if anchor is None:
            steps.append(
                StepVerdict(
                    step_id=step_id,
                    execution=Execution.NOT_EXECUTED,
                    order=Order.NOT_APPLICABLE,
                    anchor=None,
                    prerequisite_step_id=None,
                    prerequisite_anchor=None,
                    raw=None,
                )
            )
            continue
        offenders = violated_prerequisites(step_id, starts, graph)
        prerequisite = offenders[0] if offenders else None
        steps.append(
            StepVerdict(
                step_id=step_id,
                execution=Execution.EXECUTED,
                order=Order.VIOLATION if offenders else Order.OK,
                anchor=anchor,
                prerequisite_step_id=prerequisite,
                prerequisite_anchor=(
                    None if prerequisite is None else anchors.get(prerequisite)
                ),
                raw=None,
            )
        )
    return RecordingVerdicts(
        recording_id=recording.recording_id, steps=tuple(steps), status="expected"
    )


@dataclass(frozen=True)
class Case:
    """One perturbation of a normal recording and the answer it deserves.

    ``recording`` is the modified annotation: it is the ground truth of the
    case, tags included. ``expected`` is what a perfect system emits for it,
    and is ``None`` for the shuffled case, whose point is not a right answer
    but the ratio η between the answers with and without a coherent memory.
    """

    name: str
    recording: Recording
    expected: RecordingVerdicts | None
    note: str


def _single_timed_steps(recording: Recording) -> list[int]:
    """Steps performed exactly once, with times — the ones a case may move.

    A step the recording performs twice has two intervals and no single one
    to swap or remove; excluding them keeps every case's effect on the
    timeline unambiguous.
    """
    counts: dict[int, int] = {}
    timed: dict[int, bool] = {}
    for step in recording.steps:
        counts[step.step_id] = counts.get(step.step_id, 0) + 1
        timed[step.step_id] = timed.get(step.step_id, True) and step.start_s is not None
    return sorted(
        step_id for step_id, count in counts.items() if count == 1 and timed[step_id]
    )


def _first_slot(slots: Sequence[_Slot], step_id: int) -> _Slot:
    for slot in slots:
        if slot.step_id == step_id:
            return slot
    raise ValueError(f"step {step_id} has no instance in this recording")


def _swap(slots: Sequence[_Slot], first: int, second: int) -> None:
    """Exchange the intervals of two steps, in place."""
    a = _first_slot(slots, first)
    b = _first_slot(slots, second)
    a.start_s, b.start_s = b.start_s, a.start_s
    a.end_s, b.end_s = b.end_s, a.end_s


def _starts_of(slots: Sequence[_Slot]) -> dict[int, float | None]:
    starts: dict[int, float | None] = {}
    for slot in sorted(slots, key=lambda s: (s.start_s is None, s.start_s or 0.0)):
        if slot.step_id not in starts or starts[slot.step_id] is None:
            starts[slot.step_id] = slot.start_s
    return starts


def _pick_edge(
    recording: Recording,
    graph: TaskGraph,
    candidates: Sequence[int],
    rng: random.Random,
    *,
    forbid: frozenset[int] = frozenset(),
) -> tuple[int, int]:
    """A prerequisite edge whose swap really turns the dependent into a violation."""
    edges = [
        (prerequisite, step)
        for prerequisite, step in sorted(graph.edges)
        if prerequisite in candidates
        and step in candidates
        and prerequisite not in forbid
        and step not in forbid
    ]
    rng.shuffle(edges)
    for prerequisite, step in edges:
        slots = _slots(recording)
        _swap(slots, prerequisite, step)
        if step in derive_order_truth(_starts_of(slots), graph):
            return prerequisite, step
    raise ValueError(
        f"{recording.recording_id}: no task-graph edge whose two steps are timed "
        f"once and whose swap creates an order violation"
    )


def _pick_independent_pair(
    recording: Recording,
    graph: TaskGraph,
    candidates: Sequence[int],
    rng: random.Random,
) -> tuple[int, int]:
    """Two steps with no path between them whose swap breaks no other order.

    Independence in the graph is necessary but not sufficient: moving a step
    later can push it past something that requires it. The rule decides, as
    everywhere else — a pair is accepted only when the swapped timeline has
    exactly the violations the original had.
    """
    base = derive_order_truth(first_starts(recording), graph)
    ancestors = {step_id: graph.ancestors(step_id) for step_id in candidates}
    pairs = [
        (first, second)
        for index, first in enumerate(candidates)
        for second in candidates[index + 1 :]
        if first not in ancestors[second] and second not in ancestors[first]
    ]
    rng.shuffle(pairs)
    for first, second in pairs:
        slots = _slots(recording)
        _swap(slots, first, second)
        if derive_order_truth(_starts_of(slots), graph) == base:
            return first, second
    raise ValueError(
        f"{recording.recording_id}: no pair of independent steps whose swap "
        f"leaves the order truth unchanged"
    )


def _case(
    name: str,
    recording: Recording,
    sop: Sop,
    graph: TaskGraph,
    slots: Sequence[_Slot],
    note: str,
    *,
    omitted: frozenset[int] = frozenset(),
    derive_tags: bool = True,
    expected: bool = True,
) -> Case:
    """Assemble one case: rebuild the annotation, then read the answer off it."""
    modified = _rebuild(
        recording, slots, graph, omitted=omitted, derive_tags=derive_tags
    )
    segments = get_video_segments(synthetic_video(modified))
    return Case(
        name=name,
        recording=modified,
        expected=(
            expected_verdicts(modified, sop, graph, segments) if expected else None
        ),
        note=note,
    )


def make_cases(
    recording: Recording,
    sop: Sop,
    graph: TaskGraph,
    *,
    seed: int = DEFAULT_SEED,
) -> list[Case]:
    """Build the seven harness cases from one normal recording.

    Every choice — which step is omitted, which edge is swapped, which
    independent pair is moved — is drawn from ``random.Random(seed)`` over a
    sorted candidate list, so the same recording and seed always give the same
    cases. Candidates are the steps performed exactly once and with times: a
    step the recording repeats has no single interval to move.

    The consequences of a perturbation are never written by hand. Removing a
    step makes every step that required it an order violation (a prerequisite
    that was not performed counts as a violation) and both the tag and the
    expected verdict come from ``derive_order_truth``; the same holds for the
    swaps, which may drag other steps into a violation.

    Raises ``ValueError`` when the recording cannot host a case — no timed
    edge to swap, no independent pair — rather than returning a case that
    tests nothing.
    """
    rng = random.Random(seed)
    candidates = _single_timed_steps(recording)
    if not candidates:
        raise ValueError(
            f"{recording.recording_id}: no step is performed exactly once with times"
        )

    omitted_step = rng.choice(candidates)
    linked = _pick_edge(recording, graph, candidates, rng)
    independent = _pick_independent_pair(recording, graph, candidates, rng)
    linked_elsewhere = _pick_edge(
        recording, graph, candidates, rng, forbid=frozenset({omitted_step})
    )
    repeated_step = rng.choice(candidates)
    shuffle_order = list(range(len(recording.steps)))
    rng.shuffle(shuffle_order)

    cases: list[Case] = []

    cases.append(
        _case(
            "normal",
            recording,
            sop,
            graph,
            _slots(recording),
            "(a) the annotation untouched: a perfect system reports every step "
            "executed, in order, anchored on its annotated interval.",
        )
    )

    slots = _slots(recording)
    omitted_slot = _first_slot(slots, omitted_step)
    omitted_slot.start_s = None
    omitted_slot.end_s = None
    cases.append(
        _case(
            "omission",
            recording,
            sop,
            graph,
            slots,
            f"(b) step {omitted_step} was never performed. It is tagged "
            f"'{MISSING_TAG}', and every step that required it becomes an order "
            f"violation citing it, because a prerequisite that was not found "
            f"violates the order.",
            omitted=frozenset({omitted_step}),
        )
    )

    slots = _slots(recording)
    _swap(slots, *linked)
    cases.append(
        _case(
            "linked_swap",
            recording,
            sop,
            graph,
            slots,
            f"(c) steps {linked[0]} and {linked[1]} exchanged intervals, and "
            f"{linked[0]} is a prerequisite of {linked[1]}; the dependent now "
            f"runs first, which is an order error.",
        )
    )

    slots = _slots(recording)
    _swap(slots, *independent)
    cases.append(
        _case(
            "unlinked_swap",
            recording,
            sop,
            graph,
            slots,
            f"(d) steps {independent[0]} and {independent[1]} exchanged "
            f"intervals but neither precedes the other in the task graph; the "
            f"order between independent steps is free, so nothing is an error.",
        )
    )

    slots = _slots(recording)
    omitted_slot = _first_slot(slots, omitted_step)
    omitted_slot.start_s = None
    omitted_slot.end_s = None
    _swap(slots, *linked_elsewhere)
    cases.append(
        _case(
            "omission_and_swap",
            recording,
            sop,
            graph,
            slots,
            f"(e) both errors at once: step {omitted_step} omitted and steps "
            f"{linked_elsewhere[0]}/{linked_elsewhere[1]} swapped, on steps that "
            f"do not touch each other.",
            omitted=frozenset({omitted_step}),
        )
    )

    slots = _slots(recording)
    repeated = _first_slot(slots, repeated_step)
    last_end = max(slot.end_s for slot in slots if slot.end_s is not None)
    duration = (repeated.end_s or 0.0) - (repeated.start_s or 0.0)
    slots.append(
        _Slot(
            step_id=repeated_step,
            start_s=last_end,
            end_s=last_end + duration,
            tags=frozenset(),
        )
    )
    cases.append(
        _case(
            "repeated_step",
            recording,
            sop,
            graph,
            slots,
            f"(f) step {repeated_step} is performed a second time at the end of "
            f"the recording. Repetition is not an error: both occurrences are "
            f"executed and in order, and the single verdict the SOP asks for "
            f"belongs to the first one.",
        )
    )

    slots = _slots(recording)
    intervals = [(slot.start_s, slot.end_s) for slot in slots]
    for slot, index in zip(slots, shuffle_order):
        slot.start_s, slot.end_s = intervals[index]
    cases.append(
        _case(
            "shuffled",
            recording,
            sop,
            graph,
            slots,
            "(g) every interval permuted, so the memory no longer tells a "
            "coherent story. There is no right answer to score: the harness "
            "reports η, the ratio between the verdicts the system still gets "
            "right here and the ones it got right on case (a) (the "
            "shuffled-memory baseline). "
            "The annotation carries no tags — the recording was not altered, "
            "its memory was.",
            derive_tags=False,
            expected=False,
        )
    )
    return cases


def eta(correct_shuffled: int, correct_normal: int) -> float | None:
    """Ratio of correct verdicts with a shuffled memory over an ordered one.

    Near 1 means the answers did not depend on the memory being coherent —
    the prior did the work. ``None`` when the ordered memory produced no
    correct verdict at all, since a ratio to zero says nothing about the
    memory.
    """
    if correct_normal == 0:
        return None
    return correct_shuffled / correct_normal


def count_matching(
    expected: RecordingVerdicts, got: RecordingVerdicts
) -> tuple[int, int]:
    """Count the steps whose execution *and* order verdicts both match.

    Anchors are not compared: they are scored by IoU in the localization
    metric, and demanding an exact interval here would make every case fail for
    a reason the case is not about.
    """
    answers = {verdict.step_id: verdict for verdict in got.steps}
    matched = 0
    for verdict in expected.steps:
        answer = answers.get(verdict.step_id)
        if (
            answer is not None
            and answer.execution is verdict.execution
            and answer.order is verdict.order
        ):
            matched += 1
    return matched, len(expected.steps)


def score_case(case: Case, got: RecordingVerdicts) -> dict:
    """Compare a run's verdicts with the ones the case expects.

    A case passes when every step matches: the harness is a test, not a
    benchmark, so a partial score is a diagnosis and not a result. The
    shuffled case has no expectation to compare against — it is scored
    against case (a)'s expectation by the harness, which is where η comes
    from — and reports ``scored: false`` here.
    """
    if case.expected is None:
        return {
            "case": case.name,
            "recording_id": case.recording.recording_id,
            "scored": False,
            "total": len(got.steps),
            "matched": None,
            "passed": None,
            "mismatches": [],
        }
    matched, total = count_matching(case.expected, got)
    answers = {verdict.step_id: verdict for verdict in got.steps}
    mismatches = []
    for verdict in case.expected.steps:
        answer = answers.get(verdict.step_id)
        if (
            answer is not None
            and answer.execution is verdict.execution
            and answer.order is verdict.order
        ):
            continue
        mismatches.append(
            {
                "step_id": verdict.step_id,
                "expected_execution": verdict.execution.value,
                "expected_order": verdict.order.value,
                "got_execution": None if answer is None else answer.execution.value,
                "got_order": None if answer is None else answer.order.value,
            }
        )
    return {
        "case": case.name,
        "recording_id": case.recording.recording_id,
        "scored": True,
        "total": total,
        "matched": matched,
        "passed": matched == total,
        "mismatches": mismatches,
    }

"""Every evaluation quantity, as a pure function of the verdicts.

The per-category counts that are the main result, the per-recording variant,
the pairwise order accuracy that asks whether the system read the video or
the recipe, the localization of the anchors, the unclassified rate that keeps
abstention visible, the segment-level EDA used in the literature, and the
cost counters.

Two conventions hold throughout. Undefined is ``None``, never zero: a
precision with no predictions and an IoU with no anchors are missing
measurements, and averaging a zero in their place would quietly reward a
system for saying nothing. And an unclassified verdict, including the
absence of one, is a negative prediction for both categories, the
conservative reading: the system did not point at the error.
"""

import statistics
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from video_agent.eval.matching import MatchedInstance
from video_agent.memory.types import Segment
from video_agent.task.annotations import (
    MISSING_TAG,
    ORDER_TAG,
    Recording,
    StepInstance,
    TaskGraph,
)
from video_agent.task.verdicts import Execution, Order, RecordingVerdicts, StepVerdict

Category = Literal["order", "omission"]
"""The two error categories the evaluation measures, always scored separately.

``"omission"`` is the missing-step category."""

CATEGORIES: tuple[Category, ...] = ("order", "omission")

_TRUTH_TAG: dict[str, str] = {"order": ORDER_TAG, "omission": MISSING_TAG}

RESPECTS = "respects"
VIOLATES = "violates"
"""The two strata of the pairwise metric: pairs the annotated order
respects, and pairs that carry the order error."""

HIT_IOU = 0.5
"""IoU a localization must reach to count as a hit."""


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return ``numerator / denominator``, or ``None`` when it is 0/0."""
    if denominator == 0:
        return None
    return numerator / denominator


def _check_category(category: str) -> str:
    if category not in _TRUTH_TAG:
        raise ValueError(
            f"unknown category {category!r}, expected one of {CATEGORIES}"
        )
    return category


@dataclass(frozen=True)
class PRF:
    """The confusion counts of one binary problem and what they imply.

    ``precision``/``recall``/``f1`` are ``None`` when their denominator is
    zero. F1 is undefined whenever either of its arguments is, and also when
    both are zero: the harmonic mean of nothing is not zero.
    """

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    f1: float | None

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int, tn: int) -> "PRF":
        """Build the derived rates from the four counts."""
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        if precision is None or recall is None or precision + recall == 0.0:
            f1: float | None = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return cls(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1)

    @property
    def support(self) -> int:
        """Number of positives in the ground truth (VP + FN)."""
        return self.tp + self.fn

    @property
    def prevalence(self) -> float | None:
        """Fraction of units that carry the tag; the precision of "all error"."""
        return _ratio(self.tp + self.fn, self.tp + self.fp + self.fn + self.tn)

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
            "prevalence": self.prevalence,
        }


def is_positive_truth(instance: StepInstance, category: Category) -> bool:
    """Whether the annotation tags ``instance`` with the category's error.

    Instances carrying only other tags (Technique, Measurement, ...) are
    negatives here by design: the categories are order errors and missing
    steps, not "some error".
    """
    return _TRUTH_TAG[_check_category(category)] in instance.tags


def is_positive_prediction(verdict: StepVerdict | None, category: Category) -> bool:
    """Whether the verdict points at the category's error.

    ``indeterminado`` (unclassified) and a missing verdict are negatives, the
    conservative choice, and are counted apart by ``indeterminate_rate``.
    """
    _check_category(category)
    if verdict is None:
        return False
    if category == "order":
        return verdict.order is Order.VIOLATION
    return verdict.execution is Execution.NOT_EXECUTED


def category_prf(matched: Sequence[MatchedInstance], category: Category) -> PRF:
    """Score one error category over step instances (the main metric)."""
    _check_category(category)
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for entry in matched:
        truth = is_positive_truth(entry.instance, category)
        prediction = is_positive_prediction(entry.verdict, category)
        counts[_cell(truth, prediction)] += 1
    return PRF.from_counts(**counts)


def _cell(truth: bool, prediction: bool) -> str:
    if truth and prediction:
        return "tp"
    if prediction:
        return "fp"
    if truth:
        return "fn"
    return "tn"


def per_video_prf(
    matched_by_recording: Mapping[str, Sequence[MatchedInstance]], category: Category
) -> PRF:
    """Score the same category per recording instead of per instance.

    The unit is "this recording has an order error / a missing step": the truth
    is any instance carrying the tag and the prediction is any verdict
    pointing at it. Secondary metric — it is closest to the practical use
    (flagging a recording) but its N is the number of recordings.
    """
    _check_category(category)
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for matched in matched_by_recording.values():
        truth = any(is_positive_truth(entry.instance, category) for entry in matched)
        prediction = any(
            is_positive_prediction(entry.verdict, category) for entry in matched
        )
        counts[_cell(truth, prediction)] += 1
    return PRF.from_counts(**counts)


def aggregate_any_error_prf(matched: Sequence[MatchedInstance]) -> PRF:
    """Score "the instance has some error", reported beside the two categories.

    Never reported alone: an aggregate number hides that a system may be
    detecting technique errors while missing the missing steps, which is
    exactly what the per-category scores are there to show.
    """
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for entry in matched:
        truth = bool(entry.instance.tags)
        prediction = entry.verdict is not None and (
            entry.verdict.order is Order.VIOLATION
            or entry.verdict.execution is Execution.NOT_EXECUTED
        )
        counts[_cell(truth, prediction)] += 1
    return PRF.from_counts(**counts)


@dataclass(frozen=True)
class PairwiseStats:
    """How many prerequisite pairs the system ordered correctly.

    ``n`` counts the pairs of the stratum, ``determinate`` those the verdicts
    actually decide, and the accuracy is over the latter: a system that
    abstains is not credited with a coin flip.
    """

    n: int
    determinate: int
    correct: int
    accuracy: float | None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "determinate": self.determinate,
            "correct": self.correct,
            "accuracy": self.accuracy,
        }


def _first_timed_starts(recording: Recording) -> dict[int, float]:
    """Start of the first timed occurrence of each step, by step id.

    The task graph constrains the *first* execution of a step (that is how
    ``load_task_graph`` merges a repeated node), so a repetition later in the
    recording never decides whether an order was respected.
    """
    starts: dict[int, float] = {}
    for instance in recording.steps:
        if instance.start_s is not None and instance.step_id not in starts:
            starts[instance.step_id] = instance.start_s
    return starts


def _pairs(graph: TaskGraph, edges: str) -> list[tuple[int, int]]:
    """The prerequisite pairs to score: the graph's edges or its closure."""
    if edges == "direct":
        return sorted(graph.edges)
    if edges == "transitive":
        return sorted(
            (ancestor, step)
            for step in graph.prerequisites
            for ancestor in graph.ancestors(step)
        )
    raise ValueError(f"unknown edges {edges!r}, expected 'direct' or 'transitive'")


def _predicted_stratum(
    prerequisite: StepVerdict | None, step: StepVerdict | None, prerequisite_id: int
) -> str | None:
    """Read the pair's predicted order out of two verdicts, or abstain.

    Anchors are the primary evidence: they are what every executed verdict
    must carry and what makes the answer auditable. Without them the order verdict
    is still an answer about this pair when it names this prerequisite as the
    violated one, or when it says the step's order is fine; anything else the
    system did not decide.
    """
    if (
        prerequisite is not None
        and step is not None
        and prerequisite.anchor is not None
        and step.anchor is not None
    ):
        return (
            RESPECTS
            if prerequisite.anchor.start_s < step.anchor.start_s
            else VIOLATES
        )
    if step is None:
        return None
    if step.order is Order.VIOLATION and step.prerequisite_step_id == prerequisite_id:
        return VIOLATES
    if step.order is Order.OK:
        return RESPECTS
    return None


def pairwise_order_accuracy(
    recording: Recording,
    verdicts: RecordingVerdicts | None,
    graph: TaskGraph,
    *,
    edges: Literal["direct", "transitive"] = "direct",
) -> dict[str, PairwiseStats]:
    """Ask, for every prerequisite pair, whether the system got the order right.

    Both strata are always returned, zeros included, because their sizes are
    part of the result: the "respects" stratum is the bulk of the pairs and
    the "violates" one carries the order errors, and the accuracy in the
    second is the direct measure of whether the system read the video or
    recited the recipe.

    Only pairs whose two steps were annotated with times are scored: a
    missing step has no annotated order to be right or wrong about.
    """
    starts = _first_timed_starts(recording)
    by_step: dict[int, StepVerdict] = {}
    if verdicts is not None:
        for verdict in verdicts.steps:
            by_step.setdefault(verdict.step_id, verdict)

    tally = {
        RESPECTS: {"n": 0, "determinate": 0, "correct": 0},
        VIOLATES: {"n": 0, "determinate": 0, "correct": 0},
    }
    for prerequisite_id, step_id in _pairs(graph, edges):
        if prerequisite_id not in starts or step_id not in starts:
            continue
        truth = (
            RESPECTS if starts[prerequisite_id] < starts[step_id] else VIOLATES
        )
        predicted = _predicted_stratum(
            by_step.get(prerequisite_id), by_step.get(step_id), prerequisite_id
        )
        tally[truth]["n"] += 1
        if predicted is not None:
            tally[truth]["determinate"] += 1
            tally[truth]["correct"] += int(predicted == truth)
    return {
        stratum: PairwiseStats(
            n=counts["n"],
            determinate=counts["determinate"],
            correct=counts["correct"],
            accuracy=_ratio(counts["correct"], counts["determinate"]),
        )
        for stratum, counts in tally.items()
    }


DURATION_BINS: tuple[tuple[float, float], ...] = (
    (0, 10),
    (10, 20),
    (20, 40),
    (40, 80),
    (80, float("inf")),
)
"""Step-duration bins of the localization metric, in seconds.

Duration dominates how well a step is recovered."""


@dataclass(frozen=True)
class LocalizationStats:
    """Anchor quality inside one duration bin.

    ``n`` counts the executed instances of the bin and ``anchored`` those
    whose verdict carried an interval; the averages are over the anchored
    ones, because a verdict without an anchor is unclassified for this
    metric, not a localization of IoU 0.
    """

    n: int
    anchored: int
    mean_iou: float | None
    hit_at_half: float | None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "anchored": self.anchored,
            "mean_iou": self.mean_iou,
            "hit_at_half": self.hit_at_half,
        }


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection over union of two time intervals; 0 when they are disjoint."""
    start = max(a[0], b[0])
    end = min(a[1], b[1])
    intersection = max(0.0, end - start)
    union = max(a[1], b[1]) - min(a[0], b[0])
    if union <= 0.0:
        return 0.0
    return intersection / union


def _bin_of(duration: float, bins: Sequence[tuple[float, float]]) -> tuple[float, float]:
    for low, high in bins:
        if low <= duration < high:
            return (low, high)
    raise ValueError(f"duration {duration} falls outside the bins {tuple(bins)}")


def localization(
    matched: Sequence[MatchedInstance],
    bins: Sequence[tuple[float, float]] = DURATION_BINS,
) -> dict[tuple[float, float], LocalizationStats]:
    """Score the anchors of executed steps, by step duration.

    Only instances that were actually performed (they have annotated times)
    and that the system reported as executed take part: a missing step has
    no interval to localize and a step the system did not
    claim to have seen has nothing to compare. Every bin appears in the
    result even when empty, so a table row never silently disappears.
    """
    tally: dict[tuple[float, float], list[float]] = {tuple(b): [] for b in bins}
    counted: dict[tuple[float, float], int] = {tuple(b): 0 for b in bins}
    for entry in matched:
        instance = entry.instance
        if instance.start_s is None or instance.end_s is None:
            continue
        verdict = entry.verdict
        if verdict is None or verdict.execution is not Execution.EXECUTED:
            continue
        key = _bin_of(instance.end_s - instance.start_s, bins)
        counted[key] += 1
        if verdict.anchor is None:
            continue
        tally[key].append(
            interval_iou(
                (verdict.anchor.start_s, verdict.anchor.end_s),
                (instance.start_s, instance.end_s),
            )
        )
    return {
        key: LocalizationStats(
            n=counted[key],
            anchored=len(ious),
            mean_iou=statistics.fmean(ious) if ious else None,
            hit_at_half=(
                sum(iou >= HIT_IOU for iou in ious) / len(ious) if ious else None
            ),
        )
        for key, ious in tally.items()
    }


def indeterminate_rate(
    matched: Sequence[MatchedInstance], category: Category
) -> float | None:
    """Fraction of instances the system left unclassified, per category.

    Precision can always be bought with abstention, so the report shows what
    abstention cost. The order rate skips instances the system called
    ``não executado`` (not executed): an order verdict is defined only where
    there is an execution to order, so ``n/a`` there is not an abstention.
    """
    _check_category(category)
    total = 0
    undecided = 0
    for entry in matched:
        verdict = entry.verdict
        if category == "order":
            if verdict is not None and verdict.execution is Execution.NOT_EXECUTED:
                continue
            total += 1
            undecided += verdict is None or verdict.order is Order.INDETERMINATE
        else:
            total += 1
            undecided += (
                verdict is None or verdict.execution is Execution.INDETERMINATE
            )
    return _ratio(undecided, total)


def _overlaps(segment: Segment, start_s: float, end_s: float) -> bool:
    return start_s < segment.end_s and segment.start_s < end_s


def _inside(segment: Segment, start_s: float, end_s: float) -> bool:
    return start_s <= segment.start_s and segment.end_s <= end_s


def _segment_truth(recording: Recording, segments: Sequence[Segment]) -> list[bool]:
    """Label each segment as an error when it holds any frame of an error instance."""
    intervals = [
        (instance.start_s, instance.end_s)
        for instance in recording.steps
        if instance.tags and instance.start_s is not None and instance.end_s is not None
    ]
    return [
        any(_overlaps(segment, start, end) for start, end in intervals)
        for segment in segments
    ]


def eda(
    recording: Recording,
    verdicts: RecordingVerdicts | None,
    segments: Sequence[Segment],
) -> tuple[int, int]:
    """Segment-level error-detection accuracy, EgoPED's definition.

    A segment is an error in the truth when it contains any frame of an
    instance with any error tag, and an error in the prediction when it falls
    inside the anchor of a verdict that reports an error. In practice only
    order violations contribute: a ``não executado`` (not executed) verdict
    is about a step that has no interval, so it carries no anchor to fall
    inside.

    Returned as counts, not as an accuracy, so the caller can pool segments
    across recordings before dividing — a recording is not a segment.
    """
    truth = _segment_truth(recording, segments)
    anchors: list[tuple[float, float]] = []
    if verdicts is not None:
        for verdict in verdicts.steps:
            is_error = (
                verdict.order is Order.VIOLATION
                or verdict.execution is Execution.NOT_EXECUTED
            )
            if is_error and verdict.anchor is not None:
                anchors.append((verdict.anchor.start_s, verdict.anchor.end_s))
    correct = 0
    for segment, labelled in zip(segments, truth):
        predicted = any(_inside(segment, start, end) for start, end in anchors)
        correct += int(predicted == labelled)
    return correct, len(segments)


def eda_all_normal(
    recording: Recording, segments: Sequence[Segment]
) -> tuple[int, int]:
    """EDA of the "everything is normal" baseline, reported beside any EDA.

    Most segments are normal, so this number is high; reporting it beside
    the system's EDA shows how much of the score is the prior.
    """
    truth = _segment_truth(recording, segments)
    return sum(not labelled for labelled in truth), len(segments)


@dataclass(frozen=True)
class CostStats:
    """What a configuration spent, per recording and in total."""

    recordings: int
    mean_tool_calls: float
    mean_llm_calls: float
    mean_tokens_in: float
    mean_tokens_out: float
    mean_wall_s: float
    median_wall_s: float
    total_tokens_in: int
    total_tokens_out: int

    def to_dict(self) -> dict:
        return {
            "recordings": self.recordings,
            "mean_tool_calls": self.mean_tool_calls,
            "mean_llm_calls": self.mean_llm_calls,
            "mean_tokens_in": self.mean_tokens_in,
            "mean_tokens_out": self.mean_tokens_out,
            "mean_wall_s": self.mean_wall_s,
            "median_wall_s": self.median_wall_s,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
        }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def cost_summary(records: Sequence[dict]) -> CostStats:
    """Summarize the cost counters of a run's records.

    Reads ``usage_total`` rather than ``usage``: the object-memory
    sub-agent's episodes are
    part of what the configuration cost. Wall time comes from the record
    itself, which measures the whole recording including the tools. An empty
    sequence yields zeros with ``recordings = 0``, the only honest reading of
    "no runs" for a total, and the field that says the averages mean nothing.
    """
    usages = [record.get("usage_total") or {} for record in records]
    walls = [float(record.get("wall_s") or 0.0) for record in records]
    return CostStats(
        recordings=len(records),
        mean_tool_calls=_mean([float(u.get("tool_calls", 0)) for u in usages]),
        mean_llm_calls=_mean([float(u.get("llm_calls", 0)) for u in usages]),
        mean_tokens_in=_mean([float(u.get("tokens_in", 0)) for u in usages]),
        mean_tokens_out=_mean([float(u.get("tokens_out", 0)) for u in usages]),
        mean_wall_s=_mean(walls),
        median_wall_s=statistics.median(walls) if walls else 0.0,
        total_tokens_in=sum(int(u.get("tokens_in", 0)) for u in usages),
        total_tokens_out=sum(int(u.get("tokens_out", 0)) for u in usages),
    )

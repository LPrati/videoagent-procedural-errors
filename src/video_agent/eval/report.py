"""One run's records in, the report tables out.

This is the only module of the package that touches a disk: it reads a
``RunStore`` and assembles every evaluation quantity into one ``Evaluation``,
which serializes to plain JSON and renders into Markdown and CSV tables.
Keeping the I/O here and the arithmetic in ``metrics``/``bootstrap`` means a
number in a table can always be traced back to a pure function of the
verdicts.

The renderers are deliberately simple string builders: what a script
produces is a row to paste into a document, not a layout engine. Rendered
tables show percentages; the JSON and the CSV keep the fractions a later
script would want.
"""

import itertools
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from video_agent.eval.baselines import all_error, all_normal

# ``baselines.noise_floor`` computes |A − B| per metric, which *is* the noise
# floor only when A and B are two executions of one configuration. Comparing
# two different configurations with it would make every difference its own
# floor, so it is aliased to what it actually computes.
from video_agent.eval.baselines import noise_floor as absolute_difference
from video_agent.eval.bootstrap import (
    CI,
    Statistic,
    bootstrap_by_recording,
    discordant_counts,
    mcnemar,
    paired_difference_ci,
)
from video_agent.eval.matching import (
    MatchedInstance,
    count_optional_unperformed,
    match_recording,
)
from video_agent.eval.metrics import (
    CATEGORIES,
    RESPECTS,
    VIOLATES,
    Category,
    CostStats,
    LocalizationStats,
    PairwiseStats,
    PRF,
    aggregate_any_error_prf,
    category_prf,
    cost_summary,
    eda,
    eda_all_normal,
    indeterminate_rate,
    is_positive_prediction,
    is_positive_truth,
    localization,
    pairwise_order_accuracy,
    per_video_prf,
)
from video_agent.eval.order_from_anchors import (
    LLM_ORDER,
    OrderPolicy,
    apply_order_policy,
)
from video_agent.media import Video
from video_agent.memory.store import segment_grid
from video_agent.memory.types import Segment
from video_agent.task.annotations import AnnotationSet
from video_agent.task.run import RunStore
from video_agent.task.verdicts import RecordingVerdicts, StepVerdict

RATE_METRICS: tuple[str, ...] = ("precision", "recall", "f1")
"""The three rates each category reports, and the ones that get an interval."""

MISSING_STATUS = "missing"
"""Status of a recording the run has no record for; its verdicts are ``None``."""

RUN_STATUSES: tuple[str, ...] = (
    "ok",
    "parse_failed",
    "max_iterations",
    "error",
    MISSING_STATUS,
)
"""Every outcome a recording can have, in the order the tables list them.

Fixed rather than read off the data so a column never disappears from the
main results table because one run happened to have no failure of that
kind. A status outside this tuple is still counted in ``status_counts`` and
still shows up in the JSON; the table just has no column for it.
"""

LENGTH_FINISH_REASON = "length"
"""``finish_reason`` of a turn the server truncated at the token cap.

Counted per run because a truncated turn is how an answer silently loses its
last steps, and the status alone does not show it.
"""


def bin_label(bounds: tuple[float, float]) -> str:
    """Name a duration bin for a JSON key or a table row: ``"20-40"``, ``"80-inf"``."""
    low, high = bounds
    return f"{_number(low)}-{_number(high)}"


def _number(value: float) -> str:
    if value == float("inf"):
        return "inf"
    return str(int(value)) if float(value).is_integer() else str(value)


@dataclass(frozen=True)
class EdaStats:
    """Pooled segment-level accuracy over the recordings that have a grid."""

    correct: int
    total: int
    accuracy: float | None
    recordings: int

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "total": self.total,
            "accuracy": self.accuracy,
            "recordings": self.recordings,
        }


@dataclass
class Evaluation:
    """Every evaluation quantity of one run over one set of recordings.

    Mutable and assembled field by field, because it is a report and not a
    value: ``evaluate_run`` fills it, ``to_dict`` writes it, the renderers
    read it.
    """

    run_name: str
    recording_ids: tuple[str, ...]
    status_counts: dict[str, int]
    instances: int
    excluded_optional: int
    """Instances dropped as optional steps that were not performed."""
    length_turns: int
    """LLM turns the server truncated at the token cap (``finish_reason``)."""
    answer_candidates: int
    """Objects with a ``steps`` list the models wrote, summed over records."""
    config_ids: tuple[str, ...]
    """The distinct configurations the records claim; one, unless mixing was allowed."""
    category: dict[str, PRF]
    category_ci: dict[str, dict[str, CI]]
    per_video: dict[str, PRF]
    any_error: PRF
    pairwise: dict[str, PairwiseStats]
    localization: dict[tuple[float, float], LocalizationStats]
    indeterminate: dict[str, float | None]
    eda: EdaStats
    eda_all_normal: EdaStats
    cost: CostStats
    baselines: dict[str, dict[str, PRF]]
    n_bootstrap: int = 2000
    seed: int = 0
    subset: str = "all"
    order_policy: dict = field(default_factory=LLM_ORDER.to_dict)
    """Which reading produced the order column (``OrderPolicy.to_dict``).

    Part of the report and not of the command line only: two results rows
    computed from the same records under different readings are not the same
    measurement, and the JSON has to say which one it is.
    """

    def to_dict(self) -> dict:
        """Render to plain JSON types: no numpy, no tuples as keys."""
        return {
            "run_name": self.run_name,
            "subset": self.subset,
            "recordings": len(self.recording_ids),
            "recording_ids": list(self.recording_ids),
            "status_counts": dict(sorted(self.status_counts.items())),
            "instances": self.instances,
            "excluded_optional": self.excluded_optional,
            "length_turns": self.length_turns,
            "answer_candidates": self.answer_candidates,
            "config_ids": list(self.config_ids),
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "order_policy": dict(self.order_policy),
            "category": {
                name: prf.to_dict() for name, prf in sorted(self.category.items())
            },
            "category_ci": {
                name: {metric: ci.to_dict() for metric, ci in sorted(cis.items())}
                for name, cis in sorted(self.category_ci.items())
            },
            "per_video": {
                name: prf.to_dict() for name, prf in sorted(self.per_video.items())
            },
            "any_error": self.any_error.to_dict(),
            "pairwise": {
                stratum: stats.to_dict()
                for stratum, stats in sorted(self.pairwise.items())
            },
            "localization": {
                bin_label(bounds): stats.to_dict()
                for bounds, stats in self.localization.items()
            },
            "indeterminate": dict(sorted(self.indeterminate.items())),
            "eda": self.eda.to_dict(),
            "eda_all_normal": self.eda_all_normal.to_dict(),
            "cost": self.cost.to_dict(),
            "baselines": {
                name: {
                    category: prf.to_dict() for category, prf in sorted(scores.items())
                }
                for name, scores in sorted(self.baselines.items())
            },
        }


def read_verdicts(record: Mapping) -> RecordingVerdicts:
    """Rebuild the verdicts of a run record."""
    return RecordingVerdicts(
        recording_id=str(record["recording_id"]),
        steps=tuple(StepVerdict.from_dict(step) for step in record["verdicts"]),
        status=str(record.get("status", "")),
    )


def record_segments(record: Mapping) -> list[Segment] | None:
    """Rebuild the segment grid a record was anchored against, if it says.

    The grid is a function of the video's fps and frame count, both stored in
    the record precisely so evaluation never needs the video file.
    """
    video = record.get("video")
    if not video:
        return None
    return segment_grid(
        Video(
            path=Path(str(record.get("recording_id", "unknown"))),
            fps=float(video["fps"]),
            frame_count=int(video["frame_count"]),
        ),
        int(video["n_segments"]),
    )


def count_length_turns(record: Mapping) -> int:
    """LLM turns of one record the server cut at the token cap.

    Reads the transcript the record kept, sub-agent episodes included: a
    truncated turn inside the object-memory sub-agent loses observations the
    orchestrator then answers without, so it counts as much as a truncated
    turn of the parent loop. A record with no transcript — the run died
    before the runner could hand one over — contributes 0, which is what is
    known about it, not a claim that nothing was truncated.
    """
    transcript = record.get("transcript")
    return 0 if not transcript else _length_turns_of(transcript)


def _length_turns_of(transcript: Mapping) -> int:
    total = 0
    for step in transcript.get("steps") or ():
        turn = step.get("turn") or {}
        total += turn.get("finish_reason") == LENGTH_FINISH_REASON
        action = step.get("action") or {}
        nested = action.get("sub_transcript")
        if nested:
            total += _length_turns_of(nested)
    return total


def _record_config_id(record: Mapping) -> str | None:
    """The configuration a record claims, or ``None`` when it names none."""
    config = record.get("config") or {}
    config_id = config.get("config_id")
    return None if config_id is None else str(config_id)


def _check_one_configuration(
    first_of: Mapping[str, str], *, allow_mixed: bool
) -> tuple[str, ...]:
    """Refuse a run whose records were produced by different configurations.

    Scoring two configurations as one results row is not a comparison, it
    is an average of two systems, and nothing downstream would show it: the
    metrics are pooled over instances and the ``config_id`` is not in them.
    So the mixture is a contract violation by default, and the caller that
    means it (a rerun of a few recordings under a fixed prompt, say) has to
    say so — after which the ids travel in the report and in the status line.
    """
    ids = tuple(sorted(first_of))
    if len(ids) <= 1:
        return ids
    listed = ", ".join(
        f"{config_id} (first in {first_of[config_id]})" for config_id in ids
    )
    message = f"the run mixes {len(ids)} configurations: {listed}"
    if not allow_mixed:
        raise ValueError(
            f"{message}; pass allow_mixed_config to score them as one row anyway"
        )
    print(f"warning: {message}", file=sys.stderr, flush=True)
    return ids


def _counts(matched: Sequence[MatchedInstance], category: Category) -> tuple[int, ...]:
    """The four confusion cells of one recording, for fast resampling."""
    tp = fp = fn = tn = 0
    for entry in matched:
        truth = is_positive_truth(entry.instance, category)
        prediction = is_positive_prediction(entry.verdict, category)
        if truth and prediction:
            tp += 1
        elif prediction:
            fp += 1
        elif truth:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _rate_statistic(
    counts: Mapping[str, tuple[int, ...]], metric: str
) -> Statistic:
    """A metric recomputed by summing precomputed per-recording cells.

    The bootstrap draws 2 000 resamples of every metric; recomputing them
    from the instances each time is slow, and the cells are additive, so the
    resample only has to add four integers per recording.
    """

    def statistic(sample: Sequence[str]) -> float | None:
        totals = [0, 0, 0, 0]
        for recording_id in sample:
            cells = counts.get(recording_id)
            if cells is None:
                continue
            for index, value in enumerate(cells):
                totals[index] += value
        return getattr(PRF.from_counts(*totals), metric)

    return statistic


def evaluate_run(
    store: RunStore,
    recording_ids: Sequence[str],
    annotations: AnnotationSet,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    subset: str = "all",
    allow_mixed_config: bool = False,
    order_policy: OrderPolicy = LLM_ORDER,
) -> Evaluation:
    """Score one run over ``recording_ids``.

    A recording with no record is not dropped: it is a recording the system
    did not answer about, so its instances are unclassified and its
    status is counted as ``missing``. Silently evaluating only what succeeded
    would let a configuration buy precision by crashing.

    The records must all come from one configuration unless
    ``allow_mixed_config`` says otherwise, and unperformed optional steps are
    counted apart in ``excluded_optional`` rather than scored.

    ``order_policy`` decides where the order column comes from, and it is
    applied once, right after the verdicts are read, so every number below —
    the category rates, the pairwise accuracy, the unclassified rates, the
    baselines — is computed from one and the same set of verdicts. The
    default keeps the model's own answer (judged order).
    """
    ids = list(recording_ids)
    matched_by_recording: dict[str, list[MatchedInstance]] = {}
    verdicts_by_recording: dict[str, RecordingVerdicts | None] = {}
    records: list[dict] = []
    statuses: Counter[str] = Counter()
    first_recording_of: dict[str, str] = {}
    excluded_optional = 0
    length_turns = 0
    answer_candidates = 0
    pairwise: dict[str, list[PairwiseStats]] = {RESPECTS: [], VIOLATES: []}
    eda_correct = eda_total = eda_normal_correct = eda_normal_total = 0
    eda_recordings = 0
    baseline_matched: dict[str, list[MatchedInstance]] = {
        "all_normal": [],
        "all_error": [],
    }

    for recording_id in ids:
        recording = annotations.recordings.get(recording_id)
        if recording is None:
            raise ValueError(f"unknown recording {recording_id!r} in the annotations")
        sop, graph = annotations.for_recording(recording_id)
        record = store.load(recording_id) if store.has(recording_id) else None
        if record is None:
            statuses[MISSING_STATUS] += 1
            verdicts: RecordingVerdicts | None = None
        else:
            records.append(record)
            statuses[str(record.get("status", ""))] += 1
            verdicts = apply_order_policy(read_verdicts(record), graph, order_policy)
            length_turns += count_length_turns(record)
            # ``get``: a record may carry no ``n_answer_candidates`` key.
            answer_candidates += int(record.get("n_answer_candidates") or 0)
            config_id = _record_config_id(record)
            if config_id is not None:
                first_recording_of.setdefault(config_id, recording_id)
        verdicts_by_recording[recording_id] = verdicts
        excluded_optional += count_optional_unperformed(recording)
        matched_by_recording[recording_id] = match_recording(recording, verdicts)

        for stratum, stats in pairwise_order_accuracy(
            recording, verdicts, graph
        ).items():
            pairwise[stratum].append(stats)

        for name, builder in (("all_normal", all_normal), ("all_error", all_error)):
            baseline_matched[name].extend(
                match_recording(recording, builder(recording_id, sop))
            )

        segments = None if record is None else record_segments(record)
        if segments is not None:
            eda_recordings += 1
            correct, total = eda(recording, verdicts, segments)
            eda_correct += correct
            eda_total += total
            correct, total = eda_all_normal(recording, segments)
            eda_normal_correct += correct
            eda_normal_total += total

    config_ids = _check_one_configuration(
        first_recording_of, allow_mixed=allow_mixed_config
    )
    all_matched = [entry for entries in matched_by_recording.values() for entry in entries]
    category = {name: category_prf(all_matched, name) for name in CATEGORIES}
    category_ci: dict[str, dict[str, CI]] = {}
    for name in CATEGORIES:
        counts = {
            recording_id: _counts(matched, name)
            for recording_id, matched in matched_by_recording.items()
        }
        category_ci[name] = {
            metric: bootstrap_by_recording(
                _rate_statistic(counts, metric), ids, n=n_bootstrap, seed=seed
            )
            for metric in RATE_METRICS
        }

    return Evaluation(
        run_name=store.run_name,
        recording_ids=tuple(ids),
        status_counts=dict(statuses),
        instances=len(all_matched),
        excluded_optional=excluded_optional,
        length_turns=length_turns,
        answer_candidates=answer_candidates,
        config_ids=config_ids,
        category=category,
        category_ci=category_ci,
        per_video={
            name: per_video_prf(matched_by_recording, name) for name in CATEGORIES
        },
        any_error=aggregate_any_error_prf(all_matched),
        pairwise={
            stratum: _pooled_pairwise(entries) for stratum, entries in pairwise.items()
        },
        localization=localization(all_matched),
        indeterminate={
            name: indeterminate_rate(all_matched, name) for name in CATEGORIES
        },
        eda=_pooled_eda(eda_correct, eda_total, eda_recordings),
        eda_all_normal=_pooled_eda(
            eda_normal_correct, eda_normal_total, eda_recordings
        ),
        cost=cost_summary(records),
        baselines={
            name: {
                category_name: category_prf(matched, category_name)
                for category_name in CATEGORIES
            }
            for name, matched in baseline_matched.items()
        },
        n_bootstrap=n_bootstrap,
        seed=seed,
        subset=subset,
        order_policy=order_policy.to_dict(),
    )


def _pooled_pairwise(entries: Sequence[PairwiseStats]) -> PairwiseStats:
    """Add up one stratum over recordings; the unit of the metric is the pair."""
    n = sum(entry.n for entry in entries)
    determinate = sum(entry.determinate for entry in entries)
    correct = sum(entry.correct for entry in entries)
    return PairwiseStats(
        n=n,
        determinate=determinate,
        correct=correct,
        accuracy=None if determinate == 0 else correct / determinate,
    )


def _pooled_eda(correct: int, total: int, recordings: int) -> EdaStats:
    return EdaStats(
        correct=correct,
        total=total,
        accuracy=None if total == 0 else correct / total,
        recordings=recordings,
    )


def flat_metrics(evaluation: Evaluation) -> dict[str, float | None]:
    """Flatten the headline numbers into one ``name -> value`` table.

    This is what the noise floor compares between two executions of the same
    configuration: the floor is per metric, so the metrics need names.
    """
    flat: dict[str, float | None] = {}
    for name, prf in evaluation.category.items():
        for metric in RATE_METRICS:
            flat[f"{name}.{metric}"] = getattr(prf, metric)
    for name, prf in evaluation.per_video.items():
        for metric in RATE_METRICS:
            flat[f"per_video.{name}.{metric}"] = getattr(prf, metric)
    for metric in RATE_METRICS:
        flat[f"any_error.{metric}"] = getattr(evaluation.any_error, metric)
    for stratum, stats in evaluation.pairwise.items():
        flat[f"pairwise.{stratum}.accuracy"] = stats.accuracy
    for name, rate in evaluation.indeterminate.items():
        flat[f"indeterminate.{name}"] = rate
    flat["eda.accuracy"] = evaluation.eda.accuracy
    for bounds, stats in evaluation.localization.items():
        flat[f"localization.{bin_label(bounds)}.mean_iou"] = stats.mean_iou
    return flat


def _instance_predictions(
    matched: Sequence[MatchedInstance], category: Category
) -> tuple[list[bool], list[bool]]:
    return (
        [is_positive_prediction(entry.verdict, category) for entry in matched],
        [is_positive_truth(entry.instance, category) for entry in matched],
    )


REPLICA_IGNORED_KEYS: frozenset[str] = frozenset(
    {
        "seed",
        "code_version",
        "config_id",
        "prompt_template_source",
        "question_template_source",
    }
)
"""Configuration keys two replicas of one configuration may legitimately differ on.

``seed`` is what varies between the two executions (and is inert at
temperature 0 anyway); ``code_version`` moves whenever the repository does,
and two replicas need not run on the same code revision. ``config_id`` is
the digest of the whole dict, those two keys included, so comparing it would
refuse every replica pair by construction. The two ``*_template_source`` keys
are provenance (where the template text was read from: the package or a file
path) and may be absent from a record; the text itself is compared through
its digest (``prompt_sha256``, ``question_template_sha256``), which is not
ignored.
"""


def check_replica_configs(
    config_a: Mapping | None, config_b: Mapping | None
) -> tuple[str, ...]:
    """Refuse two runs that are not replicas of one configuration.

    ``--write-noise-floor`` is a claim: these two runs differ only in their
    execution, so |A − B| measures noise and not a system. The claim is
    checkable against what the records stamp, and a floor estimated
    from two different configurations would be a floor made of the very
    effect it is supposed to filter out. Returns the keys compared, so the
    caller can say what the check covered.
    """
    first = dict(config_a or {})
    second = dict(config_b or {})
    keys = sorted((set(first) | set(second)) - REPLICA_IGNORED_KEYS)
    differing = [key for key in keys if first.get(key) != second.get(key)]
    if differing:
        raise ValueError(
            "the two runs are not replicas of one configuration; they differ "
            f"on {', '.join(differing)}. Only "
            f"{', '.join(sorted(REPLICA_IGNORED_KEYS))} may differ between the "
            "two executions of a noise-floor estimate"
        )
    return tuple(keys)


def run_config(store: RunStore, recording_ids: Sequence[str]) -> dict | None:
    """The configuration stamped on the first record of ``store`` that has one.

    One record is enough because ``evaluate_run`` has already refused a run
    whose records disagree on their ``config_id``.
    """
    for recording_id in recording_ids:
        if store.has(recording_id):
            config = store.load(recording_id).get("config")
            if config:
                return dict(config)
    return None


def compare_runs(
    store_a: RunStore,
    store_b: RunStore,
    recording_ids: Sequence[str],
    annotations: AnnotationSet,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
    noise_floor: Mapping[str, float | None] | None = None,
    allow_mixed_config: bool = False,
    require_replica: bool = False,
    order_policy: OrderPolicy = LLM_ORDER,
) -> dict:
    """Compare two runs over the same recordings.

    Paired throughout: the same videos, the same bootstrap draws for both
    configurations, and McNemar over the instances they disagree on.

    ``absolute_difference`` is |A − B| per metric. Read on two *replicas* of
    one configuration (the same system run twice) it is one of the two
    estimates behind the noise floor, the other being half the width of the
    paired interval in ``differences``; ``noise_floor_detail`` puts the two
    together. Read on two different configurations it is just the difference
    being measured, so it can never be its own floor: the floor of a comparison
    has to come from outside it, which is what ``noise_floor`` is for. ``None``
    means none was supplied and no difference here may be called detected.

    ``require_replica`` is the check behind ``--write-noise-floor``: the two
    runs have to be stamped with the same configuration, ``seed`` and
    ``code_version`` apart, or the "floor" would contain the very effect it
    exists to filter out.

    ``order_policy`` applies to both sides: a comparison of a run read one
    way against a run read another way would measure the reading and call it
    the system, and the noise floor is the floor *of a reading*, which
    is why the policy travels in the result.
    """
    ids = list(recording_ids)
    if require_replica:
        check_replica_configs(run_config(store_a, ids), run_config(store_b, ids))
    matched_a = _matched(store_a, ids, annotations, order_policy)
    matched_b = _matched(store_b, ids, annotations, order_policy)

    differences: dict[str, dict[str, dict]] = {}
    mcnemar_table: dict[str, dict[str, float | int]] = {}
    for category in CATEGORIES:
        counts_a = {
            recording_id: _counts(matched, category)
            for recording_id, matched in matched_a.items()
        }
        counts_b = {
            recording_id: _counts(matched, category)
            for recording_id, matched in matched_b.items()
        }
        differences[category] = {
            metric: paired_difference_ci(
                _rate_statistic(counts_a, metric),
                _rate_statistic(counts_b, metric),
                ids,
                n=n_bootstrap,
                seed=seed,
            ).to_dict()
            for metric in RATE_METRICS
        }
        instances_a = list(itertools.chain.from_iterable(matched_a[i] for i in ids))
        instances_b = list(itertools.chain.from_iterable(matched_b[i] for i in ids))
        predictions_a, truth = _instance_predictions(instances_a, category)
        predictions_b, _ = _instance_predictions(instances_b, category)
        b, c = discordant_counts(predictions_a, predictions_b, truth)
        mcnemar_table[category] = {"b": b, "c": c, "p": mcnemar(b, c)}

    evaluation_a = evaluate_run(
        store_a,
        ids,
        annotations,
        n_bootstrap=0,
        seed=seed,
        allow_mixed_config=allow_mixed_config,
        order_policy=order_policy,
    )
    evaluation_b = evaluate_run(
        store_b,
        ids,
        annotations,
        n_bootstrap=0,
        seed=seed,
        allow_mixed_config=allow_mixed_config,
        order_policy=order_policy,
    )
    flat_a = flat_metrics(evaluation_a)
    flat_b = flat_metrics(evaluation_b)
    return {
        "run_a": store_a.run_name,
        "run_b": store_b.run_name,
        "recordings": len(ids),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "order_policy": order_policy.to_dict(),
        "differences": differences,
        "mcnemar": mcnemar_table,
        "flat_a": flat_a,
        "flat_b": flat_b,
        "absolute_difference": absolute_difference(flat_a, flat_b),
        "noise_floor": None if noise_floor is None else dict(noise_floor),
    }


def half_ci_width(interval: Mapping) -> float | None:
    """Half the width of a bootstrap interval, or ``None`` when it has none.

    This is the second estimate of the replica noise the floor takes the
    maximum with: a single |s1 − s2| is one draw of a
    noisy quantity, while the paired interval of s1 − s2 estimates the same
    spread out of 2 000 resamples.
    """
    low = interval.get("low")
    high = interval.get("high")
    if low is None or high is None:
        return None
    return abs(float(high) - float(low)) / 2.0


def noise_floor_detail(comparison: Mapping) -> dict[str, dict[str, float | None]]:
    """The noise floor per metric, and both numbers behind it.

    Read on two replicas of one configuration, the floor of a metric is the
    **larger** of |s1 − s2| and half the width of the 95% interval of the
    paired difference s1 − s2. Each entry keeps ``abs_difference``,
    ``half_ci_width`` and the ``floor`` that won, so a reader can see which
    estimate the floor came from instead of taking a single number on faith.

    Only the six per-category rates have a paired interval; every other
    metric of ``flat_metrics`` has ``half_ci_width`` ``None`` and falls back
    to the absolute difference. ``None`` in both leaves the floor undefined,
    and that metric then has no floor to clear.
    """
    absolute = comparison.get("absolute_difference") or {}
    halves: dict[str, float | None] = {}
    for category, metrics in (comparison.get("differences") or {}).items():
        for metric, interval in metrics.items():
            halves[f"{category}.{metric}"] = half_ci_width(interval)
    detail: dict[str, dict[str, float | None]] = {}
    for key in sorted(set(absolute) | set(halves)):
        difference = absolute.get(key)
        half = halves.get(key)
        candidates = [value for value in (difference, half) if value is not None]
        detail[key] = {
            "abs_difference": difference,
            "half_ci_width": half,
            "floor": max(candidates) if candidates else None,
        }
    return detail


def noise_floor_values(
    detail: Mapping[str, Mapping[str, float | None]]
) -> dict[str, float | None]:
    """Flatten ``noise_floor_detail`` to the ``{metric: floor}`` table."""
    return {key: entry["floor"] for key, entry in detail.items()}


def _matched(
    store: RunStore,
    recording_ids: Sequence[str],
    annotations: AnnotationSet,
    order_policy: OrderPolicy = LLM_ORDER,
) -> dict[str, list[MatchedInstance]]:
    """Pair every instance with its verdict, under the order reading in force.

    The task graph is loaded per recording because the anchors reading needs
    it; under the default reading it costs a dictionary lookup and changes
    nothing.
    """
    matched: dict[str, list[MatchedInstance]] = {}
    for recording_id in recording_ids:
        recording = annotations.recordings.get(recording_id)
        if recording is None:
            raise ValueError(f"unknown recording {recording_id!r} in the annotations")
        _, graph = annotations.for_recording(recording_id)
        record = store.load(recording_id) if store.has(recording_id) else None
        verdicts = None if record is None else read_verdicts(record)
        matched[recording_id] = match_recording(
            recording, apply_order_policy(verdicts, graph, order_policy)
        )
    return matched


def _percent(value: float | None, digits: int = 1) -> str:
    """Render a rate as a percentage, or an em dash when undefined."""
    if value is None:
        return "—"
    return f"{100.0 * value:.{digits}f}"


def _fixed(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _triple(prf: PRF) -> str:
    return (
        f"{_percent(prf.precision)} / {_percent(prf.recall)} / {_percent(prf.f1)}"
    )


TABLE_B_STATUS_COLUMNS: tuple[str, ...] = RUN_STATUSES + (
    "length_turns",
    "answer_candidates",
)
"""The run-hygiene columns of the results row: one per status, then the two counters.

Without them a results row read on its own cannot tell a configuration that
answered every recording from one that crashed on a third of them, and a
truncated turn (``length_turns``) is how an answer silently
loses its last steps. Named as the JSON and the CSV name them, not
translated, so a column and a key are the same word.
"""

TABLE_B_HEADER = (
    "| Run | Subset | Order P / R / F1 | Missing step P / R / F1 | "
    "Pairwise accuracy (violated pairs) | Unclassified (order / missing step) | "
    + " | ".join(TABLE_B_STATUS_COLUMNS)
    + " | Cost per recording |"
)
TABLE_B_RULE = "|" + "---|" * (7 + len(TABLE_B_STATUS_COLUMNS))

TABLE_B_FOOTNOTE = (
    "{count} instance(s) of unperformed optional steps excluded from all "
    "metrics."
)
"""Footer of the results row: how many units were dropped before scoring."""

ORDER_SOURCE_FOOTNOTE_LLM = "Order verdict: judged order (written by the LLM)."
ORDER_SOURCE_FOOTNOTE_ANCHORS = (
    "Order verdict: computed order, derived by the program from the anchors "
    "({rule}, {edges}, margin {margin})."
)
"""Second footer of the results row: which reading produced the order triple.

Two results rows can carry the same records and different order columns
(judged order or computed order), and the difference does not show anywhere
in the numbers, so the row names its reading.
"""


def results_row_footnote(evaluation: Evaluation) -> str:
    """The footer line of the results row, counting the excluded optional steps."""
    return TABLE_B_FOOTNOTE.format(count=evaluation.excluded_optional)


def order_source_footnote(evaluation: Evaluation) -> str:
    """The footer line of the results row naming where the order verdict came from."""
    policy = OrderPolicy(**evaluation.order_policy)
    if policy.source == "llm":
        return ORDER_SOURCE_FOOTNOTE_LLM
    return ORDER_SOURCE_FOOTNOTE_ANCHORS.format(
        rule=policy.rule, edges=policy.edges, margin=policy.margin
    )


def render_results_row(evaluation: Evaluation, name: str) -> str:
    """Render the run as the main results row (``results_row.md``).

    Percentages, one decimal, and an em dash for undefined, so a row can be
    pasted straight into a document. The order and missing-step triples are
    never collapsed into one number, the status columns say what the row is
    made of, and the two footers say what was left out of it (unperformed
    optional steps) and which reading produced the order triple.
    """
    cost = evaluation.cost
    cost_cell = (
        f"{cost.mean_llm_calls:.1f} LLM / {cost.mean_tool_calls:.1f} tools / "
        f"{cost.mean_tokens_in:.0f}+{cost.mean_tokens_out:.0f} tok / "
        f"{cost.mean_wall_s:.0f}s"
    )
    statuses = " | ".join(
        str(evaluation.status_counts.get(status, 0)) for status in RUN_STATUSES
    )
    row = (
        f"| {name} | {evaluation.subset} "
        f"| {_triple(evaluation.category['order'])} "
        f"| {_triple(evaluation.category['omission'])} "
        f"| {_percent(evaluation.pairwise[VIOLATES].accuracy)} "
        f"({evaluation.pairwise[VIOLATES].determinate} pairs) "
        f"| {_percent(evaluation.indeterminate['order'])} / "
        f"{_percent(evaluation.indeterminate['omission'])} "
        f"| {statuses} | {evaluation.length_turns} "
        f"| {evaluation.answer_candidates} "
        f"| {cost_cell} |"
    )
    return "\n".join(
        [
            TABLE_B_HEADER,
            TABLE_B_RULE,
            row,
            "",
            results_row_footnote(evaluation),
            order_source_footnote(evaluation),
        ]
    )


TABLE_B_CSV_HEADER = (
    "configuration,subset,recordings,instances,excluded_optional,"
    "order_tp,order_fp,order_fn,order_precision,order_recall,order_f1,"
    "order_recall_low,order_recall_high,"
    "omission_tp,omission_fp,omission_fn,omission_precision,omission_recall,"
    "omission_f1,omission_recall_low,omission_recall_high,"
    "pairwise_violates_accuracy,pairwise_violates_determinate,"
    "pairwise_respects_accuracy,pairwise_respects_determinate,"
    "indeterminate_order,indeterminate_omission,eda,eda_all_normal,"
    + ",".join(f"status_{status}" for status in RUN_STATUSES)
    + ",length_turns,answer_candidates,"
    "mean_llm_calls,mean_tool_calls,mean_tokens_in,mean_tokens_out,mean_wall_s,"
    "order_source"
)


def _csv_value(value: float | None) -> str:
    return "" if value is None else repr(float(value))


def _csv_text(value: str) -> str:
    """Quote a text cell that would otherwise split into several columns.

    The only cell that needs it is ``order_source``: an
    ``OrderPolicy`` label lists its three knobs comma-separated, and a bare
    comma there would silently shift every column after it. Quoting per RFC
    4180 keeps the file readable by any CSV parser.
    """
    if any(character in value for character in ',"\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def render_results_row_csv(evaluation: Evaluation, name: str) -> str:
    """The same row as raw fractions, for whatever reads it next.

    CSV keeps the numbers as computed — no percentages, empty for undefined —
    because a downstream script should not have to undo a formatting choice.
    The status counters, ``excluded_optional`` and the order reading are
    columns here rather than footnotes, since this is the file a later script
    reads — and a script concatenating rows of two readings needs the
    ``order_source`` column to tell them apart.
    """
    order = evaluation.category["order"]
    omission = evaluation.category["omission"]
    order_ci = evaluation.category_ci["order"]["recall"]
    omission_ci = evaluation.category_ci["omission"]["recall"]
    cost = evaluation.cost
    fields = [
        name,
        evaluation.subset,
        str(len(evaluation.recording_ids)),
        str(evaluation.instances),
        str(evaluation.excluded_optional),
        str(order.tp),
        str(order.fp),
        str(order.fn),
        _csv_value(order.precision),
        _csv_value(order.recall),
        _csv_value(order.f1),
        _csv_value(order_ci.low),
        _csv_value(order_ci.high),
        str(omission.tp),
        str(omission.fp),
        str(omission.fn),
        _csv_value(omission.precision),
        _csv_value(omission.recall),
        _csv_value(omission.f1),
        _csv_value(omission_ci.low),
        _csv_value(omission_ci.high),
        _csv_value(evaluation.pairwise[VIOLATES].accuracy),
        str(evaluation.pairwise[VIOLATES].determinate),
        _csv_value(evaluation.pairwise[RESPECTS].accuracy),
        str(evaluation.pairwise[RESPECTS].determinate),
        _csv_value(evaluation.indeterminate["order"]),
        _csv_value(evaluation.indeterminate["omission"]),
        _csv_value(evaluation.eda.accuracy),
        _csv_value(evaluation.eda_all_normal.accuracy),
        *(
            str(evaluation.status_counts.get(status, 0))
            for status in RUN_STATUSES
        ),
        str(evaluation.length_turns),
        str(evaluation.answer_candidates),
        _csv_value(cost.mean_llm_calls),
        _csv_value(cost.mean_tool_calls),
        _csv_value(cost.mean_tokens_in),
        _csv_value(cost.mean_tokens_out),
        _csv_value(cost.mean_wall_s),
        _csv_text(OrderPolicy(**evaluation.order_policy).label),
    ]
    return "\n".join([TABLE_B_CSV_HEADER, ",".join(fields)])


def render_literature_row(evaluation: Evaluation, name: str) -> str:
    """Render the run as a literature-comparison row (``literature_row.md``).

    Our EDA is a single point — the verdicts are discrete, there is no
    threshold to sweep — and the "everything normal" baseline goes on the
    line below it, because most segments are normal and the reader has to see
    how much of the number is that.
    """
    header = "| Method | Type | Oracle | Subset | EDA | Note |"
    rule = "|---|---|---|---|---|---|"
    system = (
        f"| **{name}** | zero-shot agent | L0 | {evaluation.subset} "
        f"({len(evaluation.recording_ids)} recordings) "
        f"| {_percent(evaluation.eda.accuracy)} "
        "| single point, no threshold sweep |"
    )
    baseline = (
        f"| All normal | baseline | — | {evaluation.subset} "
        f"| {_percent(evaluation.eda_all_normal.accuracy)} "
        f"| {evaluation.eda.total} segments in "
        f"{evaluation.eda.recordings} recordings |"
    )
    return "\n".join([header, rule, system, baseline])


def render_localization(evaluation: Evaluation) -> str:
    """Render the anchor quality by step duration."""
    lines = [
        "| Duration bin (s) | Instances | Anchored | Mean IoU | hit@IoU≥0.5 |",
        "|---|---|---|---|---|",
    ]
    for bounds, stats in evaluation.localization.items():
        lines.append(
            f"| {bin_label(bounds)} | {stats.n} | {stats.anchored} "
            f"| {_fixed(stats.mean_iou)} | {_fixed(stats.hit_at_half)} |"
        )
    return "\n".join(lines)


def render_localization_csv(evaluation: Evaluation) -> str:
    lines = ["bin,n,anchored,mean_iou,hit_at_half"]
    for bounds, stats in evaluation.localization.items():
        lines.append(
            f"{bin_label(bounds)},{stats.n},{stats.anchored},"
            f"{_csv_value(stats.mean_iou)},{_csv_value(stats.hit_at_half)}"
        )
    return "\n".join(lines)


FLOOR_NOTE = (
    "The noise floor is a necessary condition, not a sufficient one; the "
    "confidence interval of the paired difference is what confirms a difference."
)
"""Header line of every comparison: what clearing the floor does and does not mean.

Under the null the difference between two variants and the difference between
two replicas have the same distribution, so "exceeds the floor" happens half
the time by chance; the floor can only veto.
"""

ORDER_SOURCE_LINE = "Order verdict: "
"""Prefix of the comparison's header line naming the order reading in force."""

NO_FLOOR_NOTE = (
    "No noise floor supplied: no difference below can be read as detected. "
    "Run the same configuration twice with different seeds and pass the "
    "resulting `noise_floor.json` to `--noise-floor`."
)
"""Header line of a comparison that has no floor to check against."""


def render_comparison(comparison: Mapping) -> str:
    """Render a paired comparison: differences, McNemar, and the noise floor.

    The floor column is not decoration. A difference is reported as detected
    only when its interval excludes zero *and* it exceeds the floor, so the
    two numbers belong on the same line — and when no floor was supplied the
    verdict column is an em dash rather than a "no": the question was not
    answered, not answered negatively.

    The header says in one sentence what clearing the floor is worth
    (``FLOOR_NOTE``): it is a necessary condition and never a sufficient one,
    because with a single pair of replicas the floor is exceeded half the
    time under the null. It also names the order reading both sides were
    scored under, because the noise floor is the floor of a reading, and a
    comparison read one way does not clear a floor measured the other way.
    """
    floor = comparison.get("noise_floor")
    policy = OrderPolicy(**(comparison.get("order_policy") or LLM_ORDER.to_dict()))
    lines = [
        f"# {comparison['run_a']} vs {comparison['run_b']} "
        f"({comparison['recordings']} paired recordings)",
        "",
        f"{ORDER_SOURCE_LINE}{policy.label}",
        "",
        FLOOR_NOTE,
        "",
    ]
    if floor is None:
        lines.extend([NO_FLOOR_NOTE, ""])
    lines.extend(
        [
            "| Category | Metric | Δ (A − B) | 95% CI | Noise floor "
            "| Exceeds the floor? |",
            "|---|---|---|---|---|---|",
        ]
    )
    for category, metrics in comparison["differences"].items():
        for metric, interval in metrics.items():
            point = interval["point"]
            low = interval["low"]
            high = interval["high"]
            key = f"{category}.{metric}"
            floor_value = None if floor is None else floor.get(key)
            excludes_zero = (
                low is not None and high is not None and (low > 0.0 or high < 0.0)
            )
            if floor_value is None:
                verdict = "—"
            else:
                beats_floor = point is not None and abs(point) > floor_value
                verdict = "yes" if excludes_zero and beats_floor else "no"
            lines.append(
                f"| {category} | {metric} | {_percent(point)} "
                f"| [{_percent(low)}, {_percent(high)}] "
                f"| {_percent(floor_value)} "
                f"| {verdict} |"
            )
    lines.extend(
        [
            "",
            "| Category | b (only A correct) | c (only B correct) | p (exact McNemar) |",
            "|---|---|---|---|",
        ]
    )
    for category, cells in comparison["mcnemar"].items():
        lines.append(
            f"| {category} | {cells['b']} | {cells['c']} | {cells['p']:.4f} |"
        )
    return "\n".join(lines)

"""The report end to end: run records in, the report tables out.

Built over the real annotations and a hand-written store, so every
number below can be checked by reading the annotation: one normal recording
answered perfectly, one error recording whose single ``Order Error`` is the
only thing flagged, and a third recording the run never produced a record
for — because a report that quietly drops what crashed would let a
configuration buy precision by failing.
"""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from video_agent.eval.metrics import DURATION_BINS
from video_agent.eval.report import (
    FLOOR_NOTE,
    MISSING_STATUS,
    NO_FLOOR_NOTE,
    RUN_STATUSES,
    TABLE_B_CSV_HEADER,
    TABLE_B_HEADER,
    Evaluation,
    bin_label,
    check_replica_configs,
    compare_runs,
    count_length_turns,
    evaluate_run,
    flat_metrics,
    noise_floor_detail,
    noise_floor_values,
    order_source_footnote,
    render_comparison,
    render_localization,
    render_localization_csv,
    render_literature_row,
    render_results_row,
    render_results_row_csv,
    results_row_footnote,
)
from video_agent.eval.order_from_anchors import LLM_ORDER, OrderPolicy
from video_agent.eval.synthetic import expected_verdicts, synthetic_video
from video_agent.memory.preprocess import get_video_segments
from video_agent.models.types import Usage
from video_agent.task.annotations import (
    MISSING_TAG,
    ORDER_TAG,
    AnnotationSet,
    Recording,
)
from video_agent.task.run import RUN_SCHEMA_VERSION, RunStore
from video_agent.task.verdicts import (
    Execution,
    Order,
    RecordingVerdicts,
)

from tests._paths import load_annotations

NORMAL_ID = "1_25"
"""A normal dev recording: twelve steps, all timed, no tag."""

ERROR_ID = "23_32"
"""A dev error recording with exactly one Order Error and two Missing Steps."""

MISSING_ID = "2_8"
"""A normal dev recording deliberately left out of the store."""

OPTIONAL_ID = "18_3"
"""A dev recording with the optional step 195 that was not performed."""

CONFIG_ID = "0123456789abcdef"
"""The configuration every record of the store claims."""

FLAGGED_STEP = 261
FLAGGED_PREREQUISITE = 253
"""The tagged instance of ``ERROR_ID`` and the prerequisite the run cites."""

UNTAGGED_PAIR_NORMAL = (4, 1)
UNTAGGED_PAIR_ERROR = (255, 262)
"""(step, prerequisite) pairs of ``NORMAL_ID`` and ``ERROR_ID`` with no error tag.

Flagging one of them is a false positive of order by construction, which is
how the crossed runs of ``TestNoiseFloorEstimate`` are built.
"""

N_BOOTSTRAP = 50
"""Enough resamples to exercise the interval code; the numbers are not read."""


@pytest.fixture(scope="module")
def annotations() -> AnnotationSet:
    return load_annotations()


def test_the_fixture_recordings_are_what_the_test_assumes(
    annotations: AnnotationSet,
) -> None:
    """Pin the annotation this file computes its expected numbers from."""
    normal = annotations.recordings[NORMAL_ID]
    error = annotations.recordings[ERROR_ID]

    assert not any(step.tags for step in normal.steps)
    assert [
        step.step_id for step in error.steps if ORDER_TAG in step.tags
    ] == [FLAGGED_STEP]
    assert sum(1 for step in error.steps if MISSING_TAG in step.tags) == 2
    assert FLAGGED_PREREQUISITE in annotations.graphs[
        error.activity_id
    ].prerequisites[FLAGGED_STEP]
    assert not any(step.tags for step in annotations.recordings[MISSING_ID].steps)


def perfect_verdicts(
    annotations: AnnotationSet, recording: Recording
) -> tuple[RecordingVerdicts, dict]:
    """The verdicts a perfect system emits, plus the video metadata."""
    sop, graph = annotations.for_recording(recording.recording_id)
    video = synthetic_video(recording)
    segments = get_video_segments(video)
    verdicts = expected_verdicts(recording, sop, graph, segments)
    meta = {
        "fps": float(video.fps),
        "frame_count": int(video.frame_count),
        "n_segments": len(segments),
    }
    return verdicts, meta


def flag_violations(
    verdicts: RecordingVerdicts, violated: dict[int, int]
) -> RecordingVerdicts:
    """Rewrite the order verdicts so exactly ``violated`` names the violations.

    ``expected_verdicts`` derives the order from the annotated times, which
    on a real error recording also flags the steps whose prerequisite is
    missing. These tests need the run to have pointed at the instances they
    chose and at nothing else, so the order column is rewritten by hand and
    the execution column (the truth for missing steps) is left alone.
    """
    anchors = {verdict.step_id: verdict.anchor for verdict in verdicts.steps}
    steps = []
    for verdict in verdicts.steps:
        if verdict.execution is not Execution.EXECUTED:
            steps.append(
                replace(
                    verdict,
                    order=Order.NOT_APPLICABLE,
                    prerequisite_step_id=None,
                    prerequisite_anchor=None,
                )
            )
        elif verdict.step_id in violated:
            prerequisite_id = violated[verdict.step_id]
            steps.append(
                replace(
                    verdict,
                    order=Order.VIOLATION,
                    prerequisite_step_id=prerequisite_id,
                    prerequisite_anchor=anchors.get(prerequisite_id),
                )
            )
        else:
            steps.append(
                replace(
                    verdict,
                    order=Order.OK,
                    prerequisite_step_id=None,
                    prerequisite_anchor=None,
                )
            )
    return replace(verdicts, steps=tuple(steps))


def flag_one_violation(
    verdicts: RecordingVerdicts, step_id: int, prerequisite_id: int
) -> RecordingVerdicts:
    """Rewrite the order verdicts so exactly ``step_id`` is a violation."""
    return flag_violations(verdicts, {step_id: prerequisite_id})


def truncated_transcript(finish_reasons: tuple[str | None, ...]) -> dict:
    """A transcript of one turn per ``finish_reason``, as the record stores it.

    Only the fields ``count_length_turns`` reads are filled in: the counter
    walks the record's JSON, not a rebuilt ``Transcript``.
    """
    return {
        "prompt": "",
        "steps": [
            {
                "turn": {"index": index, "finish_reason": reason},
                "action": None,
                "final_answer": None,
                "parse_error": None,
            }
            for index, reason in enumerate(finish_reasons)
        ],
        "final_answer": None,
        "status": "final_answer",
        "usage": {},
    }


def make_record(
    recording: Recording,
    verdicts: RecordingVerdicts,
    video: dict,
    *,
    wall_s: float,
    usage: Usage,
    config_id: str = CONFIG_ID,
    n_answer_candidates: int = 1,
    transcript: dict | None = None,
) -> dict:
    """A run record of the shape ``task.run.run_recording`` writes."""
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "config": {
            "orchestrator_model": "fake-chat",
            "caption_variant": "default",
            "seed": 1,
            "code_version": "deadbee",
            "config_id": config_id,
        },
        "recording_id": recording.recording_id,
        "activity_id": recording.activity_id,
        "started_at": "2025-01-01T12:00:00+00:00",
        "finished_at": "2025-01-01T12:00:30+00:00",
        "wall_s": wall_s,
        "status": "ok",
        "error": None,
        "n_segments": video["n_segments"],
        "video": video,
        "question": "which steps were executed?",
        "final_answer_text": "{\"steps\": [{\"step_id\": 1}]}",
        "n_answer_candidates": n_answer_candidates,
        "verdicts": [verdict.to_dict() for verdict in verdicts.steps],
        "transcript": transcript,
        "usage": usage.to_dict(),
        "usage_total": usage.to_dict(),
    }


def build_store(
    root: Path, annotations: AnnotationSet, run_name: str = "agent"
) -> RunStore:
    """A two-record store: the normal recording perfect, the error one flagged.

    Shared with ``test_evaluate_cli``, which drives the script over it.
    """
    store = RunStore(root, run_name)

    normal = annotations.recordings[NORMAL_ID]
    verdicts, video = perfect_verdicts(annotations, normal)
    store.save(
        NORMAL_ID,
        make_record(
            normal,
            verdicts,
            video,
            wall_s=30.0,
            usage=Usage(llm_calls=4, tool_calls=3, tokens_in=1000, tokens_out=200),
        ),
    )

    error = annotations.recordings[ERROR_ID]
    verdicts, video = perfect_verdicts(annotations, error)
    store.save(
        ERROR_ID,
        make_record(
            error,
            flag_one_violation(verdicts, FLAGGED_STEP, FLAGGED_PREREQUISITE),
            video,
            wall_s=50.0,
            usage=Usage(llm_calls=6, tool_calls=5, tokens_in=2000, tokens_out=400),
            n_answer_candidates=2,
            transcript=truncated_transcript((None, "length", "stop")),
        ),
    )
    return store


@pytest.fixture(scope="module")
def store(tmp_path_factory, annotations: AnnotationSet) -> RunStore:
    return build_store(tmp_path_factory.mktemp("runs"), annotations)


@pytest.fixture(scope="module")
def recording_ids() -> list[str]:
    """The two recordings the run answered, plus the one it did not."""
    return [NORMAL_ID, ERROR_ID, MISSING_ID]


@pytest.fixture(scope="module")
def evaluation(
    store: RunStore, recording_ids: list[str], annotations: AnnotationSet
) -> Evaluation:
    return evaluate_run(
        store,
        recording_ids,
        annotations,
        n_bootstrap=N_BOOTSTRAP,
        seed=0,
        subset="dev",
    )


class TestEvaluateRun:
    def test_the_whole_measurement_is_plain_json(self, evaluation: Evaluation) -> None:
        payload = json.loads(json.dumps(evaluation.to_dict(), ensure_ascii=False))

        assert payload["run_name"] == "agent"
        assert payload["subset"] == "dev"
        assert payload["recordings"] == 3
        assert payload["n_bootstrap"] == N_BOOTSTRAP
        assert payload["localization"]["80-inf"]["n"] >= 0

    def test_a_recording_without_a_record_is_counted_not_dropped(
        self, evaluation: Evaluation, annotations: AnnotationSet
    ) -> None:
        assert evaluation.status_counts == {"ok": 2, MISSING_STATUS: 1}
        assert evaluation.instances == sum(
            len(annotations.recordings[recording_id].steps)
            for recording_id in (NORMAL_ID, ERROR_ID, MISSING_ID)
        )

    def test_the_instances_of_a_missing_record_are_all_indeterminate(
        self, evaluation: Evaluation, annotations: AnnotationSet
    ) -> None:
        absent = len(annotations.recordings[MISSING_ID].steps)

        assert evaluation.indeterminate["omission"] > 0
        assert evaluation.indeterminate["omission"] == pytest.approx(
            absent / evaluation.instances
        )

    def test_only_the_tagged_instance_is_scored_as_an_order_error(
        self, evaluation: Evaluation
    ) -> None:
        order = evaluation.category["order"]

        assert (order.tp, order.fp, order.fn) == (1, 0, 0)
        assert order.precision == 1.0 and order.recall == 1.0

    def test_the_two_omitted_steps_are_recovered(
        self, evaluation: Evaluation
    ) -> None:
        omission = evaluation.category["omission"]

        assert (omission.tp, omission.fp, omission.fn) == (2, 0, 0)

    def test_the_intervals_cover_the_three_rates_of_both_categories(
        self, evaluation: Evaluation
    ) -> None:
        for category in ("order", "omission"):
            for metric in ("precision", "recall", "f1"):
                interval = evaluation.category_ci[category][metric]
                assert interval.n_resamples <= N_BOOTSTRAP
                assert interval.point == pytest.approx(
                    getattr(evaluation.category[category], metric)
                )

    def test_the_video_with_the_order_error_is_the_only_positive_video(
        self, evaluation: Evaluation
    ) -> None:
        assert evaluation.per_video["order"].tp == 1
        assert evaluation.per_video["order"].fp == 0

    def test_eda_is_pooled_over_the_recordings_that_have_a_grid(
        self, evaluation: Evaluation
    ) -> None:
        assert evaluation.eda.recordings == 2
        assert evaluation.eda_all_normal.recordings == 2
        assert evaluation.eda.total == evaluation.eda_all_normal.total > 0

    def test_cost_averages_only_the_records_that_exist(
        self, evaluation: Evaluation
    ) -> None:
        cost = evaluation.cost

        assert cost.recordings == 2
        assert cost.mean_llm_calls == 5.0
        assert cost.total_tokens_in == 3000
        assert cost.mean_wall_s == 40.0
        assert cost.median_wall_s == 40.0

    def test_the_mandatory_baselines_ride_along(self, evaluation: Evaluation) -> None:
        assert evaluation.baselines["all_normal"]["order"].tp == 0
        assert evaluation.baselines["all_error"]["order"].fn == 0
        assert evaluation.baselines["all_error"]["omission"].recall == 1.0

    def test_an_unknown_recording_is_a_contract_violation(
        self, store: RunStore, annotations: AnnotationSet
    ) -> None:
        with pytest.raises(ValueError):
            evaluate_run(store, ["9_99"], annotations, n_bootstrap=0)

    def test_the_truncated_turns_and_the_answers_written_are_counted(
        self, evaluation: Evaluation
    ) -> None:
        assert evaluation.length_turns == 1
        assert evaluation.answer_candidates == 3

    def test_the_configuration_of_the_records_travels_with_the_report(
        self, evaluation: Evaluation
    ) -> None:
        assert evaluation.config_ids == (CONFIG_ID,)

    def test_nothing_is_excluded_from_recordings_without_an_optional_step(
        self, evaluation: Evaluation
    ) -> None:
        assert evaluation.excluded_optional == 0


@pytest.fixture(scope="module")
def scored_optional(store: RunStore, annotations: AnnotationSet) -> Evaluation:
    """18_3 scored with no record for it.

    The exclusion of unperformed optional steps is a property of the
    annotation, so it
    happens whether or not the system answered about the recording.
    """
    return evaluate_run(
        store, [OPTIONAL_ID], annotations, n_bootstrap=0, subset="dev"
    )


@pytest.fixture(scope="module")
def mixed_store(tmp_path_factory, annotations: AnnotationSet) -> RunStore:
    """A store whose error record claims a different configuration."""
    store = build_store(tmp_path_factory.mktemp("mixed"), annotations, "mixed")
    error = annotations.recordings[ERROR_ID]
    verdicts, video = perfect_verdicts(annotations, error)
    store.save(
        ERROR_ID,
        make_record(
            error,
            verdicts,
            video,
            wall_s=50.0,
            usage=Usage(llm_calls=6),
            config_id="fedcba9876543210",
        ),
    )
    return store


class TestTheOptionalStepIsCountedAndDropped:
    """The unperformed optional step of 18_3, end to end."""

    def test_the_dropped_instance_is_reported_not_hidden(
        self, scored_optional: Evaluation, annotations: AnnotationSet
    ) -> None:
        assert scored_optional.excluded_optional == 1
        assert scored_optional.instances == (
            len(annotations.recordings[OPTIONAL_ID].steps) - 1
        )

    def test_the_footer_of_the_results_row_says_how_many_were_dropped(
        self, scored_optional: Evaluation
    ) -> None:
        assert render_results_row(scored_optional, "agent").splitlines()[4].startswith(
            "1 instance(s)"
        )

    def test_the_baselines_of_the_same_report_drop_it_too(
        self, scored_optional: Evaluation, annotations: AnnotationSet
    ) -> None:
        kept = len(annotations.recordings[OPTIONAL_ID].steps) - 1
        for name in ("all_normal", "all_error"):
            prf = scored_optional.baselines[name]["omission"]

            assert prf.tp + prf.fp + prf.fn + prf.tn == kept


class TestRecordsWrittenBeforeTheCounters:
    """A record without the newer counters must stay readable (schema growth)."""

    def test_a_record_without_the_answer_counter_is_read_as_zero(
        self, tmp_path: Path, annotations: AnnotationSet
    ) -> None:
        store = build_store(tmp_path / "old", annotations, "old")
        record = store.load(NORMAL_ID)
        del record["n_answer_candidates"]
        store.save(NORMAL_ID, record)

        scored = evaluate_run(
            store, [NORMAL_ID, ERROR_ID], annotations, n_bootstrap=0
        )

        # Only the error record still declares its two candidates.
        assert scored.answer_candidates == 2


class TestOneConfigurationPerRun:
    """A results row is one configuration, or it is an average of two."""

    def test_two_configurations_in_one_run_are_refused(
        self, mixed_store: RunStore, annotations: AnnotationSet
    ) -> None:
        with pytest.raises(ValueError) as raised:
            evaluate_run(
                mixed_store, [NORMAL_ID, ERROR_ID], annotations, n_bootstrap=0
            )

        message = str(raised.value)
        assert CONFIG_ID in message and "fedcba9876543210" in message
        assert NORMAL_ID in message and ERROR_ID in message

    def test_the_flag_downgrades_the_refusal_to_a_warning_on_stderr(
        self, mixed_store: RunStore, annotations: AnnotationSet, capsys
    ) -> None:
        scored = evaluate_run(
            mixed_store,
            [NORMAL_ID, ERROR_ID],
            annotations,
            n_bootstrap=0,
            allow_mixed_config=True,
        )

        assert scored.config_ids == ("0123456789abcdef", "fedcba9876543210")
        assert "mixes 2 configurations" in capsys.readouterr().err

    def test_a_run_of_one_configuration_says_nothing(
        self, store: RunStore, recording_ids: list[str], annotations: AnnotationSet,
        capsys,
    ) -> None:
        evaluate_run(store, recording_ids, annotations, n_bootstrap=0)

        assert capsys.readouterr().err == ""


class TestCountLengthTurns:
    def test_a_record_without_a_transcript_counts_nothing(self) -> None:
        assert count_length_turns({"transcript": None}) == 0
        assert count_length_turns({}) == 0

    def test_every_truncated_turn_of_the_episode_counts(self) -> None:
        record = {
            "transcript": truncated_transcript(("length", "stop", "length", None))
        }

        assert count_length_turns(record) == 2

    def test_a_sub_agent_episode_is_walked_too(self) -> None:
        nested = truncated_transcript(("length",))
        outer = truncated_transcript(("stop", "length"))
        outer["steps"][0]["action"] = {"sub_transcript": nested}

        assert count_length_turns({"transcript": outer}) == 2


REPLICA_CONFIG = {
    "orchestrator_model": "qwen",
    "caption_variant": "default",
    "seed": 1,
    "code_version": "aaaaaaa",
    "config_id": "1111111111111111",
}
"""The configuration one replica of the agent stamps on its records."""


class TestCheckReplicaConfigs:
    """``--write-noise-floor`` claims two replicas; the records can say."""

    def test_a_pair_that_differs_only_in_the_seed_is_a_replica(self) -> None:
        other = {**REPLICA_CONFIG, "seed": 2, "config_id": "2222222222222222"}

        assert "seed" not in check_replica_configs(REPLICA_CONFIG, other)
        assert "orchestrator_model" in check_replica_configs(REPLICA_CONFIG, other)

    def test_a_new_code_version_between_the_two_executions_is_still_a_replica(
        self,
    ) -> None:
        other = {**REPLICA_CONFIG, "code_version": "bbbbbbb"}

        check_replica_configs(REPLICA_CONFIG, other)

    def test_a_different_caption_variant_is_not_a_replica(self) -> None:
        other = {**REPLICA_CONFIG, "caption_variant": "objects"}

        with pytest.raises(ValueError) as raised:
            check_replica_configs(REPLICA_CONFIG, other)

        assert "caption_variant" in str(raised.value)

    def test_a_key_only_one_side_has_is_a_difference(self) -> None:
        with pytest.raises(ValueError) as raised:
            check_replica_configs(REPLICA_CONFIG, {**REPLICA_CONFIG, "thinking": True})

        assert "thinking" in str(raised.value)


class TestRenderers:
    def test_the_results_row_is_its_header_one_row_and_two_footers(
        self, evaluation: Evaluation
    ) -> None:
        lines = render_results_row(evaluation, "agent").splitlines()

        assert lines[0] == TABLE_B_HEADER
        assert len(lines) == 6
        assert lines[2].startswith("| agent | dev |")
        assert lines[3] == ""
        assert lines[4] == results_row_footnote(evaluation)
        assert lines[4].startswith("0 instance(s)")
        assert lines[5] == order_source_footnote(evaluation)

    def test_the_row_has_one_cell_per_header_column(
        self, evaluation: Evaluation
    ) -> None:
        lines = render_results_row(evaluation, "agent").splitlines()

        assert len(lines[2].split("|")) == len(TABLE_B_HEADER.split("|"))
        assert len(lines[1].split("|")) == len(TABLE_B_HEADER.split("|"))

    def test_the_status_counts_are_columns_of_the_row(
        self, evaluation: Evaluation
    ) -> None:
        """Without them a row alone could not tell 60 answers from 40."""
        header = [cell.strip() for cell in TABLE_B_HEADER.split("|")]
        row = render_results_row(evaluation, "agent").splitlines()[2]
        cells = [cell.strip() for cell in row.split("|")]
        read = dict(zip(header, cells))

        assert [read[status] for status in RUN_STATUSES] == ["2", "0", "0", "0", "1"]
        assert read["length_turns"] == "1"
        assert read["answer_candidates"] == "3"

    def test_the_csv_row_keeps_the_fractions(self, evaluation: Evaluation) -> None:
        lines = render_results_row_csv(evaluation, "agent").splitlines()
        header, row = list(csv.reader(lines))

        assert lines[0] == TABLE_B_CSV_HEADER
        assert len(lines) == 2
        assert len(row) == len(header)
        assert row[:2] == ["agent", "dev"]

    def test_the_csv_carries_the_counters_the_markdown_shows(
        self, evaluation: Evaluation
    ) -> None:
        lines = render_results_row_csv(evaluation, "agent").splitlines()
        read = dict(zip(*csv.reader(lines)))

        assert read["excluded_optional"] == "0"
        assert [read[f"status_{status}"] for status in RUN_STATUSES] == [
            "2",
            "0",
            "0",
            "0",
            "1",
        ]
        assert read["length_turns"] == "1"
        assert read["answer_candidates"] == "3"

    def test_the_literature_row_carries_the_all_normal_baseline_under_the_system(
        self, evaluation: Evaluation
    ) -> None:
        lines = render_literature_row(evaluation, "agent").splitlines()

        assert lines[0] == "| Method | Type | Oracle | Subset | EDA | Note |"
        assert len(lines) == 4
        assert lines[2].startswith("| **agent** |")
        assert lines[3].startswith("| All normal |")

    def test_localization_has_one_row_per_duration_bin(
        self, evaluation: Evaluation
    ) -> None:
        lines = render_localization(evaluation).splitlines()

        assert lines[0].startswith("| Duration bin (s) |")
        assert len(lines) == 2 + len(DURATION_BINS)
        assert lines[-1].startswith("| 80-inf |")

    def test_the_localization_csv_has_the_same_rows(
        self, evaluation: Evaluation
    ) -> None:
        lines = render_localization_csv(evaluation).splitlines()

        assert lines[0] == "bin,n,anchored,mean_iou,hit_at_half"
        assert len(lines) == 1 + len(DURATION_BINS)

    def test_the_open_ended_bin_is_named_by_its_bounds(self) -> None:
        assert bin_label((80, float("inf"))) == "80-inf"
        assert bin_label((0, 10)) == "0-10"


@pytest.fixture(scope="module")
def comparison(
    store: RunStore, recording_ids: list[str], annotations: AnnotationSet
) -> dict:
    """A run compared with itself: the difference is known to be exactly zero."""
    return compare_runs(
        store,
        store,
        recording_ids,
        annotations,
        n_bootstrap=N_BOOTSTRAP,
        seed=0,
    )


class TestCompareRuns:
    def test_a_run_does_not_differ_from_itself(self, comparison: dict) -> None:
        for metrics in comparison["differences"].values():
            for interval in metrics.values():
                assert interval["point"] == 0.0
                assert interval["low"] == 0.0 and interval["high"] == 0.0

    def test_no_instance_is_discordant_with_itself(self, comparison: dict) -> None:
        for cells in comparison["mcnemar"].values():
            assert (cells["b"], cells["c"]) == (0, 0)
            assert cells["p"] == 1.0

    def test_both_flat_tables_are_reported_and_identical(
        self, comparison: dict, evaluation: Evaluation
    ) -> None:
        assert comparison["flat_a"] == comparison["flat_b"]
        assert set(comparison["flat_a"]) == set(flat_metrics(evaluation))

    def test_the_absolute_difference_of_a_run_with_itself_is_zero(
        self, comparison: dict
    ) -> None:
        values = comparison["absolute_difference"]

        assert set(values) == set(comparison["flat_a"])
        assert all(value in (0.0, None) for value in values.values())
        assert any(value == 0.0 for value in values.values())

    def test_no_floor_is_assumed_when_none_is_given(self, comparison: dict) -> None:
        assert comparison["noise_floor"] is None

    def test_the_floor_is_the_one_supplied_not_the_difference_measured(
        self, store: RunStore, recording_ids: list[str], annotations: AnnotationSet
    ) -> None:
        floor = {"order.recall": 0.25, "omission.recall": None}

        compared = compare_runs(
            store,
            store,
            recording_ids,
            annotations,
            n_bootstrap=0,
            seed=0,
            noise_floor=floor,
        )

        assert compared["noise_floor"] == floor
        assert compared["absolute_difference"]["order.recall"] == 0.0


RATE_NAMES = {"precision", "recall", "f1"}


def difference_rows(text: str) -> list[str]:
    """The rows of the differences table; the McNemar table below has no metric."""
    rows = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) > 2 and cells[2] in RATE_NAMES:
            rows.append(line)
    return rows


def save_run(
    store: RunStore,
    annotations: AnnotationSet,
    recording_id: str,
    violated: dict[int, int],
) -> None:
    """Overwrite one record with a run that flags exactly ``violated``."""
    recording = annotations.recordings[recording_id]
    verdicts, video = perfect_verdicts(annotations, recording)
    store.save(
        recording_id,
        make_record(
            recording,
            flag_violations(verdicts, violated),
            video,
            wall_s=30.0,
            usage=Usage(llm_calls=4),
        ),
    )


@pytest.fixture(scope="module")
def crossed_comparison(
    tmp_path_factory, recording_ids: list[str], annotations: AnnotationSet
) -> dict:
    """Two runs that each add one false positive of order, on different videos.

    Both recover the tagged violation of ``ERROR_ID`` and both flag exactly
    one untagged step, so their precision is the same and the point
    difference is zero — while a resample that draws one of the two
    recordings and not the other sees a large difference in either
    direction. That is why the floor also uses the interval width: a single
    |A − B| can be small by luck while the noise is not.
    """
    root = tmp_path_factory.mktemp("crossed")
    first = build_store(root, annotations, "fp-normal")
    save_run(first, annotations, NORMAL_ID, dict([UNTAGGED_PAIR_NORMAL]))
    second = build_store(root, annotations, "fp-error")
    save_run(
        second,
        annotations,
        ERROR_ID,
        dict([(FLAGGED_STEP, FLAGGED_PREREQUISITE), UNTAGGED_PAIR_ERROR]),
    )
    return compare_runs(
        first, second, recording_ids, annotations, n_bootstrap=N_BOOTSTRAP, seed=0
    )


class TestNoiseFloorEstimate:
    """The floor is the larger of |A − B| and half the CI width.

    A single absolute difference is one draw of a noisy quantity; half the
    width of the paired interval estimates the same spread out of 2 000
    resamples, and the floor takes whichever is larger so a lucky pair of
    replicas cannot certify a floor of nothing.
    """

    def test_half_the_interval_wins_when_the_replicas_agreed_by_luck(
        self,
    ) -> None:
        hand_made = {
            "absolute_difference": {"order.recall": 0.01},
            "differences": {
                "order": {"recall": {"point": 0.01, "low": -0.20, "high": 0.20}}
            },
        }

        detail = noise_floor_detail(hand_made)

        assert detail["order.recall"] == {
            "abs_difference": 0.01,
            "half_ci_width": pytest.approx(0.20),
            "floor": pytest.approx(0.20),
        }
        assert noise_floor_values(detail) == {"order.recall": pytest.approx(0.20)}

    def test_the_absolute_difference_wins_when_it_is_the_larger(self) -> None:
        hand_made = {
            "absolute_difference": {"order.recall": 0.30},
            "differences": {
                "order": {"recall": {"point": 0.30, "low": 0.25, "high": 0.35}}
            },
        }

        detail = noise_floor_detail(hand_made)

        assert detail["order.recall"]["half_ci_width"] == pytest.approx(0.05)
        assert detail["order.recall"]["floor"] == pytest.approx(0.30)

    def test_a_metric_without_an_interval_falls_back_to_the_difference(
        self,
    ) -> None:
        hand_made = {
            "absolute_difference": {"eda.accuracy": 0.02},
            "differences": {},
        }

        detail = noise_floor_detail(hand_made)

        assert detail["eda.accuracy"] == {
            "abs_difference": 0.02,
            "half_ci_width": None,
            "floor": 0.02,
        }

    def test_undefined_on_both_sides_leaves_the_floor_undefined(self) -> None:
        hand_made = {
            "absolute_difference": {"order.precision": None},
            "differences": {
                "order": {
                    "precision": {"point": None, "low": None, "high": None}
                }
            },
        }

        detail = noise_floor_detail(hand_made)

        assert detail["order.precision"] == {
            "abs_difference": None,
            "half_ci_width": None,
            "floor": None,
        }

    def test_two_runs_that_disagree_in_both_directions_get_the_wider_floor(
        self, crossed_comparison: dict
    ) -> None:
        detail = noise_floor_detail(crossed_comparison)["order.precision"]

        assert detail["abs_difference"] == pytest.approx(0.0)
        assert detail["half_ci_width"] > 0.1
        assert detail["floor"] == detail["half_ci_width"]

    def test_every_floor_is_the_larger_of_the_two_estimates(
        self, crossed_comparison: dict
    ) -> None:
        for key, entry in noise_floor_detail(crossed_comparison).items():
            estimates = [
                value
                for value in (entry["abs_difference"], entry["half_ci_width"])
                if value is not None
            ]

            assert entry["floor"] == (max(estimates) if estimates else None), key

    def test_it_covers_every_metric_of_the_comparison(
        self, comparison: dict
    ) -> None:
        detail = noise_floor_detail(comparison)

        assert set(detail) == set(comparison["absolute_difference"])
        # A run against itself: no disagreement and no spread, hence no floor
        # above zero anywhere.
        assert all(
            entry["floor"] in (0.0, None) for entry in detail.values()
        )


class TestRenderComparison:
    def test_the_header_says_the_floor_can_only_veto(self, comparison: dict) -> None:
        """Exceeding the floor is necessary, not sufficient."""
        text = render_comparison(comparison)

        assert FLOOR_NOTE in text.splitlines()
        assert "necessary condition, not a sufficient one" in FLOOR_NOTE
        assert "paired difference is what confirms" in FLOOR_NOTE

    def test_without_a_floor_nothing_may_be_called_detected(
        self, comparison: dict
    ) -> None:
        text = render_comparison(comparison)

        assert NO_FLOOR_NOTE in text
        rows = difference_rows(text)
        assert len(rows) == 6
        assert all(row.endswith("| — |") for row in rows)

    def test_with_a_floor_every_row_gets_a_verdict(self, comparison: dict) -> None:
        floor = {key: 0.01 for key in comparison["flat_a"]}

        text = render_comparison({**comparison, "noise_floor": floor})

        assert NO_FLOOR_NOTE not in text
        rows = difference_rows(text)
        assert len(rows) == 6
        # Identical runs: a zero difference never clears a floor.
        assert all(row.endswith("| no |") for row in rows)

    def test_a_difference_above_the_floor_with_an_interval_off_zero_is_detected(
        self,
    ) -> None:
        hand_made = {
            "run_a": "variant",
            "run_b": "agent",
            "recordings": 60,
            "differences": {
                "order": {
                    "recall": {"point": 0.20, "low": 0.05, "high": 0.35},
                    "precision": {"point": 0.20, "low": -0.05, "high": 0.35},
                    "f1": {"point": 0.005, "low": 0.001, "high": 0.02},
                }
            },
            "mcnemar": {"order": {"b": 7, "c": 1, "p": 0.0703}},
            "noise_floor": {"order.recall": 0.03, "order.precision": 0.03,
                            "order.f1": 0.03},
        }

        rows = difference_rows(render_comparison(hand_made))

        # Above the floor and off zero; above the floor but straddling zero;
        # off zero but below the floor.
        assert rows[0].endswith("| yes |")
        assert rows[1].endswith("| no |")
        assert rows[2].endswith("| no |")

    def test_a_metric_the_floor_does_not_name_has_no_verdict(self) -> None:
        hand_made = {
            "run_a": "variant",
            "run_b": "agent",
            "recordings": 60,
            "differences": {
                "order": {"recall": {"point": 0.20, "low": 0.05, "high": 0.35}}
            },
            "mcnemar": {},
            "noise_floor": {"omission.recall": 0.03},
        }

        rows = difference_rows(render_comparison(hand_made))

        assert rows[0].endswith("| — |")


class TestReplicaProvenanceKeys:
    def test_template_source_may_differ_or_be_absent(self):
        older = dict(REPLICA_CONFIG)
        newer = dict(
            REPLICA_CONFIG,
            prompt_template_source="package",
            question_template_source="/tmp/task_question_v1.txt",
        )
        # a record without provenance keys against one with them
        assert "question_template_source" not in check_replica_configs(older, newer)

    def test_a_different_template_digest_is_still_refused(self):
        other = dict(REPLICA_CONFIG, question_template_sha256="0" * 64)
        with pytest.raises(ValueError, match="question_template_sha256"):
            check_replica_configs(REPLICA_CONFIG, other)


ANCHORS_POLICY = OrderPolicy(source="anchors")
"""Computed order: transitive closure, both ends of a displacement, end margin."""


@pytest.fixture(scope="module")
def scored_from_anchors(
    store: RunStore, recording_ids: list[str], annotations: AnnotationSet
) -> Evaluation:
    """The same records, with the order column derived from their own anchors."""
    return evaluate_run(
        store,
        recording_ids,
        annotations,
        n_bootstrap=N_BOOTSTRAP,
        seed=0,
        subset="dev",
        order_policy=ANCHORS_POLICY,
    )


class TestTheOrderReadingIsPartOfTheReport:
    def test_the_default_is_the_column_the_model_wrote(
        self, evaluation: Evaluation
    ) -> None:
        """The judged order is the default; the computed order is opt-in."""
        assert evaluation.order_policy == LLM_ORDER.to_dict()
        assert evaluation.to_dict()["order_policy"] == {
            "source": "llm",
            "rule": "transitive",
            "edges": "both",
            "margin": "end",
        }

    def test_the_policy_travels_in_the_json(
        self, scored_from_anchors: Evaluation
    ) -> None:
        payload = json.loads(
            json.dumps(scored_from_anchors.to_dict(), ensure_ascii=False)
        )

        assert payload["order_policy"] == {
            "source": "anchors",
            "rule": "transitive",
            "edges": "both",
            "margin": "end",
        }

    def test_the_footer_of_the_results_row_names_the_reading(
        self, evaluation: Evaluation, scored_from_anchors: Evaluation
    ) -> None:
        assert (
            render_results_row(evaluation, "agent").splitlines()[5]
            == "Order verdict: judged order (written by the LLM)."
        )
        assert render_results_row(scored_from_anchors, "agent").splitlines()[5] == (
            "Order verdict: computed order, derived by the program from the "
            "anchors (transitive, both, margin end)."
        )

    def test_the_csv_carries_the_reading_as_its_last_column(
        self, evaluation: Evaluation, scored_from_anchors: Evaluation
    ) -> None:
        for scored, label in (
            (evaluation, "llm"),
            (scored_from_anchors, "anchors:transitive,both,end"),
        ):
            lines = render_results_row_csv(scored, "agent").splitlines()
            header, row = list(csv.reader(lines))
            read = dict(zip(header, row))

            assert header[-1] == "order_source"
            assert read["order_source"] == label

    def test_the_two_readings_disagree_about_order_and_nothing_else(
        self, evaluation: Evaluation, scored_from_anchors: Evaluation
    ) -> None:
        """The reading rewrites one column: execution, anchors and cost are the same."""
        assert scored_from_anchors.instances == evaluation.instances
        assert scored_from_anchors.status_counts == evaluation.status_counts
        assert (
            scored_from_anchors.category["omission"].to_dict()
            == evaluation.category["omission"].to_dict()
        )
        assert scored_from_anchors.cost.to_dict() == evaluation.cost.to_dict()
        assert scored_from_anchors.localization == evaluation.localization

    def test_the_run_that_flagged_one_step_by_hand_is_reread_from_its_anchors(
        self, scored_from_anchors: Evaluation, evaluation: Evaluation
    ) -> None:
        """The fixture's citation was written by hand; the anchors say otherwise.

        ``build_store`` flags step 261 of the error recording and nothing
        else, with anchors taken from the annotation — which is an ordered
        timeline. Read off those anchors the recording has no order violation
        at all, so the order recall of the computed order is 0 here while
        the hand-written column scores the tagged instance.
        """
        assert evaluation.category["order"].tp == 1
        assert scored_from_anchors.category["order"].tp == 0
        assert scored_from_anchors.category["order"].fp == 0

    def test_the_comparison_records_the_reading_both_sides_used(
        self, store: RunStore, recording_ids: list[str], annotations: AnnotationSet
    ) -> None:
        comparison = compare_runs(
            store,
            store,
            recording_ids,
            annotations,
            n_bootstrap=N_BOOTSTRAP,
            order_policy=ANCHORS_POLICY,
        )

        assert comparison["order_policy"] == ANCHORS_POLICY.to_dict()
        assert (
            "Order verdict: anchors:transitive,both,end"
            in render_comparison(comparison).splitlines()
        )

    def test_a_comparison_without_a_policy_is_read_as_the_llm_one(self) -> None:
        """A comparison without the key is read as the judged order."""
        hand_made = {
            "run_a": "variant",
            "run_b": "agent",
            "recordings": 60,
            "differences": {},
            "mcnemar": {},
            "noise_floor": None,
        }

        assert (
            "Order verdict: llm" in render_comparison(hand_made).splitlines()
        )

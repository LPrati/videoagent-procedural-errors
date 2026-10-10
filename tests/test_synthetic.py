"""Synthetic harness: the metric code checked against hand-known answers.

Runs over the real annotations, on the first five normal dev recordings of
distinct recipes that can host the cases. Nothing here calls a model: the
cases carry the verdicts a perfect system would emit, and what is being
checked is that the metrics read those verdicts as perfect.
"""

from pathlib import Path

import numpy as np
import pytest

from video_agent.eval.baselines import all_error, all_normal
from video_agent.eval.matching import match_recording
from video_agent.eval.metrics import (
    RESPECTS,
    VIOLATES,
    category_prf,
    eda,
    eda_all_normal,
    localization,
    pairwise_order_accuracy,
)
from video_agent.eval.synthetic import (
    CASE_LETTERS,
    CASE_NAMES,
    HashingTextEmbedder,
    Case,
    count_matching,
    derive_order_truth,
    eta,
    first_starts,
    is_order_clean,
    is_synthetic_video,
    make_cases,
    score_case,
    synthetic_temporal_memory,
    synthetic_video,
)
from video_agent.media import Video
from video_agent.memory.preprocess import get_video_segments
from video_agent.memory.store import segment_index_at
from video_agent.task.annotations import (
    MISSING_TAG,
    ORDER_TAG,
    AnnotationSet,
    TaskGraph,
    load_split,
)
from video_agent.task.verdicts import Execution, Order

from tests._paths import DEV_SPLIT, load_annotations

N_RECORDINGS = 5
POSITIVE_CASES = ("omission", "linked_swap", "omission_and_swap")
CLEAN_CASES = ("normal", "unlinked_swap", "repeated_step")


@pytest.fixture(scope="module")
def annotations() -> AnnotationSet:
    return load_annotations()


@pytest.fixture(scope="module")
def base_recordings(annotations: AnnotationSet) -> list[str]:
    """The first five normal dev recordings that can host the seven cases.

    Distinct recipes (the dev split holds one normal per recipe), every step
    timed, and an annotated order the task graph accepts — two of the 24
    normals perform a step before a prerequisite without being tagged, and
    case (a) would then have to expect an error.
    """
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
    """Every (recording, case) of the harness, built once."""
    built: dict[tuple[str, str], Case] = {}
    for recording_id in base_recordings:
        recording = annotations.recordings[recording_id]
        sop, graph = annotations.for_recording(recording_id)
        for case in make_cases(recording, sop, graph):
            built[(recording_id, case.name)] = case
    return built


def graph_of(annotations: AnnotationSet, recording_id: str) -> TaskGraph:
    return annotations.for_recording(recording_id)[1]


class TestSelection:
    def test_five_recipes_of_the_dev_split_can_host_the_cases(
        self, base_recordings: list[str]
    ) -> None:
        assert len(base_recordings) == N_RECORDINGS
        assert len(set(base_recordings)) == N_RECORDINGS

    def test_a_normal_recording_whose_order_breaks_the_graph_is_excluded(
        self, annotations: AnnotationSet, base_recordings: list[str]
    ) -> None:
        # 5_2 is annotated as normal but performs step 63 before a
        # prerequisite; the truth rule flags it, so it is not a base.
        recording = annotations.recordings["5_2"]

        assert not is_order_clean(recording, graph_of(annotations, "5_2"))
        assert "5_2" not in base_recordings


class TestHashingTextEmbedder:
    def test_rows_are_unit_vectors_of_the_requested_width(self) -> None:
        embedder = HashingTextEmbedder(dim=64)

        matrix = embedder.embed(["pour the milk", "whisk the eggs"])

        assert matrix.shape == (2, 64)
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

    def test_the_same_text_always_hashes_to_the_same_vector(self) -> None:
        embedder = HashingTextEmbedder(dim=32)

        assert np.array_equal(
            embedder.embed(["pour the milk"]), embedder.encode_text(["pour the milk"])
        )

    def test_word_overlap_is_similarity(self) -> None:
        embedder = HashingTextEmbedder(dim=512)

        rows = embedder.embed(["pour the milk", "pour the milk slowly", "wash hands"])

        assert float(rows[0] @ rows[1]) > float(rows[0] @ rows[2])

    def test_text_without_words_embeds_to_zero(self) -> None:
        assert not HashingTextEmbedder(dim=8).embed(["..."]).any()

    def test_it_refuses_to_pretend_it_has_seen_the_video(self) -> None:
        with pytest.raises(ValueError):
            HashingTextEmbedder().encode_video([])


class TestSyntheticVideo:
    def test_the_grid_runs_one_segment_past_the_last_annotated_step(
        self, annotations: AnnotationSet, base_recordings: list[str]
    ) -> None:
        recording = annotations.recordings[base_recordings[0]]
        video = synthetic_video(recording)
        segments = get_video_segments(video)
        last_end = max(step.end_s for step in recording.steps)

        assert is_synthetic_video(video)
        assert video.path.name == recording.recording_id
        assert segments[-1].end_s >= last_end
        assert segments[-2].end_s < last_end + 2.0

    def test_a_real_video_is_not_taken_for_a_synthetic_one(self) -> None:
        assert not is_synthetic_video(
            Video(path=Path("videos/1_25_360p.mp4"), fps=30.0, frame_count=600)
        )


class TestSyntheticMemory:
    def test_each_segment_is_captioned_with_the_step_being_performed(
        self, annotations: AnnotationSet, base_recordings: list[str], cases
    ) -> None:
        embedder = HashingTextEmbedder()
        for recording_id in base_recordings:
            case = cases[(recording_id, "normal")]
            sop, _ = annotations.for_recording(recording_id)
            descriptions = dict(sop.steps)
            memory = synthetic_temporal_memory(
                case.recording,
                sop,
                video=synthetic_video(case.recording),
                text_embedder=embedder,
                aligner=embedder,
            )
            for step in case.recording.steps:
                middle = (step.start_s + step.end_s) / 2.0
                index = segment_index_at(memory.segments, middle)
                assert memory.captions[index] == descriptions[step.step_id]

    def test_the_gaps_between_steps_say_nothing(
        self, annotations: AnnotationSet, base_recordings: list[str], cases
    ) -> None:
        embedder = HashingTextEmbedder()
        recording_id = base_recordings[0]
        sop, _ = annotations.for_recording(recording_id)
        case = cases[(recording_id, "normal")]

        memory = synthetic_temporal_memory(
            case.recording,
            sop,
            video=synthetic_video(case.recording),
            text_embedder=embedder,
            aligner=embedder,
        )

        assert memory.captions[-1] == "The person is standing in the kitchen."
        assert len(memory.captions) == len(memory.segments)
        assert memory.caption_embeddings.shape[0] == len(memory.segments)
        assert memory.visual_embeddings.shape[0] == len(memory.segments)

    def test_an_omitted_step_is_nowhere_in_the_memory(
        self, annotations: AnnotationSet, base_recordings: list[str], cases
    ) -> None:
        embedder = HashingTextEmbedder()
        recording_id = base_recordings[0]
        sop, _ = annotations.for_recording(recording_id)
        case = cases[(recording_id, "omission")]
        omitted = next(
            step.step_id for step in case.recording.steps if MISSING_TAG in step.tags
        )

        memory = synthetic_temporal_memory(
            case.recording,
            sop,
            video=synthetic_video(case.recording),
            text_embedder=embedder,
            aligner=embedder,
        )

        assert dict(sop.steps)[omitted] not in memory.captions


class TestCases:
    def test_every_recording_yields_the_seven_cases(
        self, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            names = [
                name for (rid, name) in cases if rid == recording_id
            ]
            assert tuple(names) == CASE_NAMES
        assert tuple(CASE_LETTERS) == ("a", "b", "c", "d", "e", "f", "g")

    def test_the_same_seed_builds_the_same_cases(
        self, annotations: AnnotationSet, base_recordings: list[str]
    ) -> None:
        recording_id = base_recordings[0]
        recording = annotations.recordings[recording_id]
        sop, graph = annotations.for_recording(recording_id)

        first = make_cases(recording, sop, graph)
        again = make_cases(recording, sop, graph)
        other = make_cases(recording, sop, graph, seed=1)

        assert [case.recording for case in first] == [case.recording for case in again]
        assert [case.note for case in first] != [case.note for case in other]

    def test_the_omission_case_removes_one_step_and_tags_it(
        self, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            case = cases[(recording_id, "omission")]
            omitted = [
                step for step in case.recording.steps if MISSING_TAG in step.tags
            ]

            assert len(omitted) == 1
            assert omitted[0].start_s is None and omitted[0].end_s is None

    def test_swapping_two_linked_steps_tags_the_dependent(
        self, annotations: AnnotationSet, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            case = cases[(recording_id, "linked_swap")]
            graph = graph_of(annotations, recording_id)
            tagged = {
                step.step_id for step in case.recording.steps if ORDER_TAG in step.tags
            }

            assert tagged
            assert tagged == derive_order_truth(first_starts(case.recording), graph)
            assert not any(MISSING_TAG in step.tags for step in case.recording.steps)

    def test_swapping_two_independent_steps_is_not_an_error(
        self, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            case = cases[(recording_id, "unlinked_swap")]

            assert not any(step.tags for step in case.recording.steps)

    def test_the_combined_case_injects_both_errors(
        self, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            case = cases[(recording_id, "omission_and_swap")]
            missing = [s for s in case.recording.steps if MISSING_TAG in s.tags]
            out_of_order = [s for s in case.recording.steps if ORDER_TAG in s.tags]

            assert len(missing) == 1
            assert len(out_of_order) >= 1
            assert missing[0].step_id not in {s.step_id for s in out_of_order}

    def test_a_repeated_step_has_two_occurrences_and_no_tag(
        self, annotations: AnnotationSet, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            base = annotations.recordings[recording_id]
            case = cases[(recording_id, "repeated_step")]
            counts: dict[int, int] = {}
            for step in case.recording.steps:
                counts[step.step_id] = counts.get(step.step_id, 0) + 1
            repeated = [
                step_id
                for step_id, count in counts.items()
                if count > sum(1 for s in base.steps if s.step_id == step_id)
            ]

            assert len(repeated) == 1
            assert counts[repeated[0]] == 2
            assert len(case.recording.steps) == len(base.steps) + 1
            assert not any(step.tags for step in case.recording.steps)

    def test_the_shuffled_case_has_no_expected_answer_and_no_tags(
        self, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            case = cases[(recording_id, "shuffled")]

            assert case.expected is None
            assert not any(step.tags for step in case.recording.steps)
            assert "η" in case.note

    def test_a_recording_with_nothing_to_perturb_is_refused(
        self, annotations: AnnotationSet, base_recordings: list[str]
    ) -> None:
        recording_id = base_recordings[0]
        recording = annotations.recordings[recording_id]
        sop, graph = annotations.for_recording(recording_id)
        empty = TaskGraph(
            activity_id=graph.activity_id,
            prerequisites={step: frozenset() for step in graph.prerequisites},
            edges=frozenset(),
        )

        with pytest.raises(ValueError):
            make_cases(recording, sop, empty)


class TestPerfectVerdictsScorePerfectly:
    def test_the_injected_errors_are_recovered_exactly(
        self, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            for name in POSITIVE_CASES:
                case = cases[(recording_id, name)]
                matched = match_recording(case.recording, case.expected)
                for category in ("order", "omission"):
                    prf = category_prf(matched, category)
                    if prf.support == 0:
                        assert prf.fp == 0
                        continue
                    assert (prf.precision, prf.recall, prf.f1) == (1.0, 1.0, 1.0)
                    assert (prf.fp, prf.fn) == (0, 0)

    def test_a_clean_case_yields_no_positives_at_all(
        self, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            for name in CLEAN_CASES:
                case = cases[(recording_id, name)]
                matched = match_recording(case.recording, case.expected)
                for category in ("order", "omission"):
                    prf = category_prf(matched, category)

                    assert (prf.tp, prf.fp, prf.fn) == (0, 0, 0)
                    assert prf.precision is None and prf.recall is None

    def test_the_pairwise_order_is_right_in_both_strata(
        self, annotations: AnnotationSet, base_recordings: list[str], cases
    ) -> None:
        violates_seen = 0
        for recording_id in base_recordings:
            graph = graph_of(annotations, recording_id)
            for name in CASE_NAMES:
                case = cases[(recording_id, name)]
                if case.expected is None:
                    continue
                stats = pairwise_order_accuracy(case.recording, case.expected, graph)
                for stratum in (RESPECTS, VIOLATES):
                    assert stats[stratum].determinate == stats[stratum].n
                    if stats[stratum].n:
                        assert stats[stratum].accuracy == 1.0
                violates_seen += stats[VIOLATES].n
        assert violates_seen > 0

    def test_the_anchors_are_the_annotated_intervals_up_to_the_grid(
        self, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            for name in CASE_NAMES:
                case = cases[(recording_id, name)]
                if case.expected is None:
                    continue
                matched = match_recording(case.recording, case.expected)
                bins = localization(matched)
                anchored = sum(stats.anchored for stats in bins.values())
                weighted = sum(
                    stats.mean_iou * stats.anchored
                    for stats in bins.values()
                    if stats.anchored
                )

                assert anchored > 0
                # 2-second segments at 30 fps: rounding an interval out to the
                # grid costs under a tenth of the union, and never a hit.
                assert weighted / anchored >= 0.9
                assert all(
                    stats.hit_at_half == 1.0
                    for stats in bins.values()
                    if stats.anchored
                )

    def test_every_segment_of_the_video_is_classified_correctly(
        self, base_recordings: list[str], cases
    ) -> None:
        for recording_id in base_recordings:
            for name in CASE_NAMES:
                case = cases[(recording_id, name)]
                if case.expected is None:
                    continue
                segments = get_video_segments(synthetic_video(case.recording))
                correct, total = eda(case.recording, case.expected, segments)
                normal_correct, _ = eda_all_normal(case.recording, segments)

                # The expected anchor of a violation spans exactly the segments
                # the annotated interval touches, so the predicted error
                # segments are the labelled ones — EDA is 1 and beats the
                # "everything normal" baseline whenever the case has an error.
                assert (correct, total) == (total, total)
                assert normal_correct <= correct


class TestBaselinesOnTheCases:
    def test_all_normal_never_finds_the_omission(
        self, base_recordings: list[str], cases, annotations: AnnotationSet
    ) -> None:
        for recording_id in base_recordings:
            case = cases[(recording_id, "omission")]
            sop, _ = annotations.for_recording(recording_id)
            matched = match_recording(
                case.recording, all_normal(recording_id, sop)
            )
            prf = category_prf(matched, "omission")

            assert prf.recall == 0.0
            assert prf.precision is None

    def test_all_error_finds_everything_at_the_price_of_the_prevalence(
        self, base_recordings: list[str], cases, annotations: AnnotationSet
    ) -> None:
        for recording_id in base_recordings:
            case = cases[(recording_id, "omission_and_swap")]
            sop, _ = annotations.for_recording(recording_id)
            matched = match_recording(case.recording, all_error(recording_id, sop))
            for category in ("order", "omission"):
                prf = category_prf(matched, category)

                assert prf.recall == 1.0
                assert prf.precision == prf.prevalence
                assert 0.0 < prf.precision < 1.0


class TestScoring:
    def test_the_expected_verdicts_score_a_perfect_case(
        self, base_recordings: list[str], cases
    ) -> None:
        case = cases[(base_recordings[0], "linked_swap")]

        score = score_case(case, case.expected)

        assert score["passed"] is True
        assert score["matched"] == score["total"] == len(case.expected.steps)
        assert score["mismatches"] == []

    def test_a_wrong_verdict_is_reported_with_both_sides(
        self, base_recordings: list[str], cases
    ) -> None:
        case = cases[(base_recordings[0], "omission")]
        wrong = all_normal(case.recording.recording_id, _sop_of(case))

        score = score_case(case, wrong)

        assert score["passed"] is False
        assert score["matched"] < score["total"]
        mismatch = score["mismatches"][0]
        assert mismatch["got_execution"] == Execution.EXECUTED.value
        assert mismatch["expected_execution"] in {
            Execution.NOT_EXECUTED.value,
            Execution.EXECUTED.value,
        }
        assert Order.VIOLATION.value in {
            entry["expected_order"] for entry in score["mismatches"]
        }

    def test_the_shuffled_case_is_not_scored_against_an_expectation(
        self, base_recordings: list[str], cases
    ) -> None:
        case = cases[(base_recordings[0], "shuffled")]
        normal = cases[(base_recordings[0], "normal")]

        score = score_case(case, normal.expected)

        assert score["scored"] is False
        assert score["matched"] is None

    def test_eta_is_the_ratio_of_correct_verdicts(self) -> None:
        assert eta(5, 10) == 0.5
        assert eta(0, 4) == 0.0
        assert eta(3, 0) is None

    def test_counting_matches_ignores_the_anchors(
        self, base_recordings: list[str], cases
    ) -> None:
        normal = cases[(base_recordings[0], "normal")]
        swapped = cases[(base_recordings[0], "linked_swap")]

        matched, total = count_matching(normal.expected, normal.expected)
        against_other, _ = count_matching(normal.expected, swapped.expected)

        assert matched == total
        assert against_other < total


def _sop_of(case: Case):
    """The SOP a case's expected verdicts were written against."""
    from video_agent.task.annotations import Sop

    return Sop(
        activity_id=case.recording.activity_id,
        recipe_name="",
        steps=tuple(
            (verdict.step_id, "") for verdict in case.expected.steps
        ),
    )


class TestDeriveOrderTruth:
    GRAPH = TaskGraph(
        activity_id=1,
        prerequisites={1: frozenset(), 2: frozenset({1}), 3: frozenset({2})},
        edges=frozenset({(1, 2), (2, 3)}),
    )

    def test_an_ordered_timeline_violates_nothing(self) -> None:
        assert derive_order_truth({1: 0.0, 2: 10.0, 3: 20.0}, self.GRAPH) == set()

    def test_a_prerequisite_performed_later_is_a_violation(self) -> None:
        assert derive_order_truth({1: 30.0, 2: 10.0, 3: 20.0}, self.GRAPH) == {2}

    def test_a_prerequisite_never_performed_is_a_violation(self) -> None:
        assert derive_order_truth({1: None, 2: 10.0, 3: 20.0}, self.GRAPH) == {2}

    def test_a_step_that_did_not_happen_has_no_order_to_violate(self) -> None:
        assert derive_order_truth({1: 30.0, 2: None, 3: 20.0}, self.GRAPH) == {3}

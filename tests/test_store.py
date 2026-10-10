from pathlib import Path

import numpy as np
import pytest

from tests.fakes import object_record
from video_agent.media import Video
from video_agent.memory.artifacts import VideoArtifacts
from video_agent.memory.store import (
    DEFAULT_ALIGNER_SLUG,
    DEFAULT_DETECTOR_SLUG,
    OBJECT_MEMORY_FILE,
    VIDEO_META_FILE,
    VISUAL_EMBEDDINGS_FILE,
    caption_embeddings_file,
    captions_file,
    crop_embedding_widths,
    detections_file,
    load_object_memory,
    load_temporal_memory,
    load_video_meta,
    object_memory_file,
    object_memory_slugs,
    object_stage,
    object_suffix,
    segment_grid,
    segment_index_at,
    segment_interval,
    tracklet_embeddings_file,
    tracklets_file,
    visual_embed_stage,
    visual_embeddings_file,
)
from video_agent.memory.types import ObjectMemory

VARIANT = "qwen3vl"


class TestVisualEmbeddingsFile:
    """Which aligner's segment embeddings a name points at."""

    def test_the_default_aligner_keeps_the_unsuffixed_name(self) -> None:
        # Existing artifacts use the unsuffixed name; a suffix here would
        # orphan them.
        assert visual_embeddings_file(None) == VISUAL_EMBEDDINGS_FILE
        assert visual_embeddings_file() == VISUAL_EMBEDDINGS_FILE
        assert visual_embeddings_file(DEFAULT_ALIGNER_SLUG) == VISUAL_EMBEDDINGS_FILE

    def test_another_aligner_carries_its_slug(self) -> None:
        assert (
            visual_embeddings_file("pe-core-l")
            == "segment_visual_embeddings.pe-core-l.npy"
        )
        assert (
            visual_embeddings_file("viclip-l-14")
            == "segment_visual_embeddings.viclip-l-14.npy"
        )

    def test_two_candidates_never_share_a_file(self) -> None:
        names = {
            visual_embeddings_file(slug)
            for slug in (None, "pe-core-l", "pe-core-g", "siglip2-so400m-mean")
        }

        assert len(names) == 4

    def test_the_lineage_key_is_suffixed_the_same_way(self) -> None:
        # A second candidate must record its model and width without erasing
        # the first's entry in meta.json.
        assert visual_embed_stage(None) == "visual_embed"
        assert visual_embed_stage(DEFAULT_ALIGNER_SLUG) == "visual_embed"
        assert visual_embed_stage("pe-core-l") == "visual_embed.pe-core-l"


@pytest.fixture
def artifacts(tmp_path: Path) -> VideoArtifacts:
    return VideoArtifacts(tmp_path, "synthetic")


def write_memory(
    artifacts: VideoArtifacts,
    captions: list[str],
    caption_rows: list[list[float]],
    visual_rows: list[list[float]],
    visual_name: str = VISUAL_EMBEDDINGS_FILE,
) -> None:
    artifacts.save_json(captions_file(VARIANT), captions)
    artifacts.save_array(caption_embeddings_file(VARIANT), np.array(caption_rows))
    artifacts.save_array(visual_name, np.array(visual_rows))


def test_load_temporal_memory_normalizes_both_channels(
    artifacts: VideoArtifacts, synthetic_video: Video
) -> None:
    write_memory(
        artifacts,
        ["#C C opens a jar", "#C C stirs the pan"],
        [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]],
        [[0.0, 2.0], [0.0, 0.0]],
    )

    memory = load_temporal_memory(artifacts, VARIANT, synthetic_video)

    assert memory.captions == ["#C C opens a jar", "#C C stirs the pan"]
    # 150 frames at 30 fps: the grid the artifacts were computed on.
    assert [(s.index, s.start_s, s.end_s) for s in memory.segments] == [
        (0, 0.0, 2.0),
        (1, 2.0, 4.0),
    ]
    assert memory.caption_embeddings[0] == pytest.approx([0.6, 0.8, 0.0])
    assert memory.visual_embeddings[0] == pytest.approx([0.0, 1.0])
    # A zero row has no direction to normalize and must stay zero.
    assert memory.visual_embeddings[1] == pytest.approx([0.0, 0.0])


def test_load_temporal_memory_reads_the_aligner_it_was_asked_for(
    artifacts: VideoArtifacts, synthetic_video: Video
) -> None:
    """One recording can hold one embeddings file per aligner."""
    write_memory(artifacts, ["a", "b"], [[1.0, 0.0]] * 2, [[2.0, 0.0]] * 2)
    artifacts.save_array(
        visual_embeddings_file("pe-core-l"), np.array([[0.0, 3.0]] * 2)
    )

    def visual(slug: str | None):
        memory = load_temporal_memory(
            artifacts, VARIANT, synthetic_video, aligner_slug=slug
        )
        return memory.visual_embeddings[0]

    assert visual("pe-core-l") == pytest.approx([0.0, 1.0])
    # X-CLIP and "no slug at all" name the same file, the
    # unsuffixed one every recording already has.
    assert visual(None) == pytest.approx([1.0, 0.0])
    assert visual(DEFAULT_ALIGNER_SLUG) == pytest.approx([1.0, 0.0])


def test_load_temporal_memory_never_substitutes_another_aligners_embeddings(
    artifacts: VideoArtifacts, synthetic_video: Video
) -> None:
    """The two matrices live in different spaces; a silent fallback scores noise."""
    write_memory(artifacts, ["a", "b"], [[1.0, 0.0]] * 2, [[2.0, 0.0]] * 2)

    with pytest.raises(FileNotFoundError) as error:
        load_temporal_memory(
            artifacts, VARIANT, synthetic_video, aligner_slug="pe-core-l"
        )

    message = str(error.value)
    assert "segment_visual_embeddings.pe-core-l.npy" in message
    assert "synthetic" in message  # which recording is missing it


def test_load_temporal_memory_can_keep_the_raw_embeddings(
    artifacts: VideoArtifacts, synthetic_video: Video
) -> None:
    write_memory(artifacts, ["a", "b"], [[3.0, 4.0]] * 2, [[0.0, 2.0]] * 2)

    memory = load_temporal_memory(artifacts, VARIANT, synthetic_video, normalize=False)

    assert memory.caption_embeddings[0] == pytest.approx([3.0, 4.0])
    assert memory.visual_embeddings[0] == pytest.approx([0.0, 2.0])


def test_load_temporal_memory_rejects_embeddings_of_another_run(
    artifacts: VideoArtifacts, synthetic_video: Video
) -> None:
    write_memory(artifacts, ["a", "b"], [[1.0], [1.0], [1.0]], [[1.0], [1.0]])

    with pytest.raises(ValueError):
        load_temporal_memory(artifacts, VARIANT, synthetic_video)


def test_load_temporal_memory_rejects_a_grid_the_video_cannot_have(
    artifacts: VideoArtifacts, synthetic_video: Video
) -> None:
    # Five captions cannot come from a 5-second video sliced into 2-second segments.
    write_memory(artifacts, list("abcde"), [[1.0]] * 5, [[1.0]] * 5)

    with pytest.raises(ValueError):
        load_temporal_memory(artifacts, VARIANT, synthetic_video)


def test_segment_grid_truncates_to_the_artifacts(synthetic_video: Video) -> None:
    assert len(segment_grid(synthetic_video, 1)) == 1
    assert len(segment_grid(synthetic_video, 2)) == 2
    with pytest.raises(ValueError):
        segment_grid(synthetic_video, 3)


def test_video_meta_is_probed_once_then_reused(
    artifacts: VideoArtifacts, synthetic_video: Video
) -> None:
    with pytest.raises(FileNotFoundError):
        load_video_meta(artifacts)

    probed = load_video_meta(artifacts, synthetic_video.path)
    assert artifacts.has(VIDEO_META_FILE)
    assert round(probed.fps) == 30
    assert probed.frame_count == 150

    cached = load_video_meta(artifacts)
    assert (cached.fps, cached.frame_count) == (probed.fps, probed.frame_count)
    assert cached.path == synthetic_video.path

    # The dataset moved: metadata survives, the path is overridden.
    moved = load_video_meta(artifacts, Path("/elsewhere/synthetic.mp4"))
    assert moved.path == Path("/elsewhere/synthetic.mp4")
    assert (moved.fps, moved.frame_count) == (probed.fps, probed.frame_count)


def test_crop_embedding_widths_come_from_the_embed_stage(
    artifacts: VideoArtifacts,
) -> None:
    with pytest.raises(ValueError):
        crop_embedding_widths(artifacts)

    artifacts.record_stage(
        "embed",
        artifact="object_memory.pkl",
        model="siglip+dinov2",
        params={"widths": [3, 5]},
    )
    assert crop_embedding_widths(artifacts) == (3, 5)




OTHER_SLUG = "other-detector"


class TestObjectFiles:
    """Which detector's object artifacts a name points at."""

    def test_the_default_detector_keeps_the_unsuffixed_names(self) -> None:
        # The default detector's artifacts are unsuffixed; a suffix here
        # would orphan four files per recording.
        for slug in (None, DEFAULT_DETECTOR_SLUG):
            assert object_suffix(slug) == ""
            assert detections_file(slug) == "detections.pkl"
            assert tracklets_file(slug) == "tracklets.pkl"
            assert tracklet_embeddings_file(slug) == "tracklet_embeddings.npy"
            assert object_memory_file(slug) == OBJECT_MEMORY_FILE
        assert object_memory_file() == OBJECT_MEMORY_FILE

    def test_another_detector_suffixes_all_four_artifacts(self) -> None:
        assert detections_file(OTHER_SLUG) == f"detections.{OTHER_SLUG}.pkl"
        assert tracklets_file(OTHER_SLUG) == f"tracklets.{OTHER_SLUG}.pkl"
        assert (
            tracklet_embeddings_file(OTHER_SLUG)
            == f"tracklet_embeddings.{OTHER_SLUG}.npy"
        )
        assert object_memory_file(OTHER_SLUG) == f"object_memory.{OTHER_SLUG}.pkl"

    def test_two_detectors_never_share_a_file(self) -> None:
        # The weights are the same and the words are not, so the memories
        # are different and the names have to be too.
        names = {
            object_memory_file(slug)
            for slug in (None, "other", "other-detector", "third-detector")
        }

        assert len(names) == 4

    def test_the_lineage_keys_are_suffixed_the_same_way(self) -> None:
        assert object_stage("detect") == "detect"
        assert object_stage("assemble", DEFAULT_DETECTOR_SLUG) == "assemble"
        assert object_stage("embed", OTHER_SLUG) == f"embed.{OTHER_SLUG}"

    def test_a_stage_that_is_not_one_of_the_four_is_a_typo(self) -> None:
        # A typo would record lineage under a key nothing reads and lose the
        # provenance of an artifact that was in fact written.
        with pytest.raises(ValueError, match="detect"):
            object_stage("detection", OTHER_SLUG)


def object_memory(category: str = "cup") -> ObjectMemory:
    return ObjectMemory(records=[object_record(1, category, (0,), [1.0, 0.0])])


class TestLoadObjectMemory:
    def test_the_unsuffixed_pickle_loads_under_both_of_its_names(
        self, artifacts: VideoArtifacts
    ) -> None:
        """``None`` and the default slug name the same existing file."""
        artifacts.save_pickle(OBJECT_MEMORY_FILE, object_memory())

        for slug in (None, DEFAULT_DETECTOR_SLUG):
            memory = load_object_memory(artifacts, slug)
            assert [record.category for record in memory.records] == ["cup"]

    def test_a_suffixed_memory_is_read_only_when_it_is_asked_for(
        self, artifacts: VideoArtifacts
    ) -> None:
        artifacts.save_pickle(OBJECT_MEMORY_FILE, object_memory("bowl"))
        artifacts.save_pickle(object_memory_file(OTHER_SLUG), object_memory("jelly"))

        assert load_object_memory(artifacts).records[0].category == "bowl"
        assert (
            load_object_memory(artifacts, OTHER_SLUG).records[0].category == "jelly"
        )

    def test_a_missing_memory_is_never_answered_with_another_detectors(
        self, artifacts: VideoArtifacts
    ) -> None:
        # "bowl" would be a plausible-looking answer to a question about a
        # word list that does not contain it.
        artifacts.save_pickle(OBJECT_MEMORY_FILE, object_memory("bowl"))

        with pytest.raises(FileNotFoundError) as error:
            load_object_memory(artifacts, OTHER_SLUG)

        message = str(error.value)
        assert f"object_memory.{OTHER_SLUG}.pkl" in message
        assert "synthetic" in message  # which recording is missing it

    def test_a_pickle_of_something_else_is_refused(
        self, artifacts: VideoArtifacts
    ) -> None:
        artifacts.save_pickle(OBJECT_MEMORY_FILE, {"records": []})

        with pytest.raises(ValueError, match="ObjectMemory"):
            load_object_memory(artifacts)


def test_crop_embedding_widths_are_read_per_detector(
    artifacts: VideoArtifacts,
) -> None:
    """Each memory was embedded by its own run of the embed stage."""
    artifacts.record_stage(
        "embed", artifact="tracklet_embeddings.npy", params={"widths": [3, 5]}
    )
    artifacts.record_stage(
        object_stage("embed", OTHER_SLUG),
        artifact=tracklet_embeddings_file(OTHER_SLUG),
        params={"widths": [7, 11]},
    )

    assert crop_embedding_widths(artifacts) == (3, 5)
    assert crop_embedding_widths(artifacts, DEFAULT_DETECTOR_SLUG) == (3, 5)
    assert crop_embedding_widths(artifacts, OTHER_SLUG) == (7, 11)
    with pytest.raises(ValueError, match="embed.third"):
        crop_embedding_widths(artifacts, "third")


class TestObjectMemorySlugs:
    def test_an_empty_directory_has_no_memories(
        self, artifacts: VideoArtifacts
    ) -> None:
        assert object_memory_slugs(artifacts) == []

    def test_the_unsuffixed_memory_comes_first_and_is_named_none(
        self, artifacts: VideoArtifacts
    ) -> None:
        # ``None`` is what a caller passes back to load it, so that is what
        # the listing says rather than the detector's name.
        artifacts.save_pickle(object_memory_file("third-detector"), object_memory())
        artifacts.save_pickle(object_memory_file(OTHER_SLUG), object_memory())
        artifacts.save_pickle(OBJECT_MEMORY_FILE, object_memory())

        assert object_memory_slugs(artifacts) == [
            None,
            OTHER_SLUG,
            "third-detector",
        ]

    def test_the_other_object_artifacts_are_not_memories(
        self, artifacts: VideoArtifacts
    ) -> None:
        artifacts.save_pickle(detections_file(OTHER_SLUG), {})
        artifacts.save_pickle(tracklets_file(OTHER_SLUG), [])

        assert object_memory_slugs(artifacts) == []



def test_segment_index_at_covers_half_open_intervals(synthetic_video: Video) -> None:
    segments = segment_grid(synthetic_video, 2)

    assert segment_index_at(segments, 0.0) == 0
    assert segment_index_at(segments, 1.999) == 0
    assert segment_index_at(segments, 2.0) == 1
    assert segment_index_at(segments, 3.999) == 1
    assert segment_index_at(segments, 4.0) is None  # past the last segment
    assert segment_index_at(segments, -0.5) is None


def test_segment_interval_clamps_to_the_grid(synthetic_video: Video) -> None:
    segments = segment_grid(synthetic_video, 2)

    assert segment_interval(segments, 0, 1) == (0.0, 4.0)
    assert segment_interval(segments, -3, 9) == (0.0, 4.0)
    assert segment_interval(segments, 1, 1) == (2.0, 4.0)
    with pytest.raises(ValueError):
        segment_interval(segments, 1, 0)
    with pytest.raises(ValueError):
        segment_interval([], 0, 0)


def test_segment_interval_spans_a_grid_that_is_not_in_time_order(
    synthetic_video: Video,
) -> None:
    """A shuffled memory holds the same intervals under other names.

    Reading the first and the last row of such a grid would return a span
    that ends before it starts; the span of a range of rows is the earliest
    start and the latest end in it, which on the ordered grid above is the
    same thing.
    """
    from video_agent.memory.ablation import SegmentPermutation, shuffle_segments

    shuffled = shuffle_segments(
        segment_grid(synthetic_video, 2), SegmentPermutation((1, 0), seed=1, ordinal=0)
    )

    assert segment_interval(shuffled, 0, 0) == (2.0, 4.0)
    assert segment_interval(shuffled, 1, 1) == (0.0, 2.0)
    assert segment_interval(shuffled, 0, 1) == (0.0, 4.0)

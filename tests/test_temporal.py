from typing import Sequence

import cv2
import numpy as np
import pytest

from video_agent.media import Clip, Video, load_clips
from video_agent.memory.preprocess import generate_temporal_memory, get_video_segments
from video_agent.memory.temporal import (
    context_frame_count,
    context_position,
    generate_segment_caption_embeddings,
    generate_segment_captions,
    generate_segment_captions_chunked,
    generate_segment_visual_embeddings,
    generate_segment_visual_embeddings_chunked,
)


class FakeCaptioner:
    def caption(self, clips: Sequence[Clip]) -> list[str]:
        return [f"clip with {len(clip.frames)} frames" for clip in clips]


class FakeTextEmbedder:
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.full((len(texts), 4), 1.0)


class FakeAligner:
    def encode_video(self, clips: Sequence[Clip]) -> np.ndarray:
        return np.full((len(clips), 8), 2.0)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return np.full((len(texts), 8), 3.0)


class MisalignedCaptioner:
    def caption(self, clips: Sequence[Clip]) -> list[str]:
        return ["only one"]


class MisalignedTextEmbedder:
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts) + 1, 4))


def test_steps_validate_alignment_with_inputs() -> None:
    clips = [Clip(frames=()), Clip(frames=())]
    with pytest.raises(ValueError):
        generate_segment_captions(MisalignedCaptioner(), clips)
    with pytest.raises(ValueError):
        generate_segment_caption_embeddings(MisalignedTextEmbedder(), ["a"])


def test_generate_temporal_memory_end_to_end(synthetic_video: Video) -> None:
    segments = get_video_segments(synthetic_video)
    memory = generate_temporal_memory(
        synthetic_video,
        segments,
        captioner=FakeCaptioner(),
        caption_embedder=FakeTextEmbedder(),
        aligner=FakeAligner(),
    )
    assert memory.segments == segments
    # 5 fps sampling over 2-second segments: 10 frames per clip (reference).
    assert memory.captions == ["clip with 10 frames"] * len(segments)
    assert memory.caption_embeddings.shape == (len(segments), 4)
    assert memory.visual_embeddings.shape == (len(segments), 8)


class ContentCaptioner:
    """Captions depend on decoded pixels; batch sizes recorded to prove bounding."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def caption(self, clips: Sequence[Clip]) -> list[str]:
        self.batch_sizes.append(len(clips))
        return [
            f"{clip.frames[0].index}:{float(clip.frames[0].data[:, :, 0].mean()):.2f}"
            for clip in clips
        ]


class ContentAligner:
    """Embeddings depend on decoded pixels; batch sizes recorded."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode_video(self, clips: Sequence[Clip]) -> np.ndarray:
        self.batch_sizes.append(len(clips))
        return np.array(
            [
                [float(clip.frames[0].data[:, :, 0].mean()), float(len(clip.frames))]
                for clip in clips
            ]
        )

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros((len(texts), 2))


def test_chunked_captions_match_in_memory(synthetic_video: Video) -> None:
    segments = get_video_segments(synthetic_video)
    intervals = [(s.start_s, s.end_s) for s in segments]
    clips = load_clips(synthetic_video, intervals, sample_fps=5.0)
    in_memory = generate_segment_captions(ContentCaptioner(), clips)

    captioner = ContentCaptioner()
    chunked = generate_segment_captions_chunked(
        captioner, synthetic_video, intervals, clips_per_batch=1
    )
    assert chunked == in_memory
    assert captioner.batch_sizes == [1] * len(segments)


def test_chunked_visual_embeddings_match_in_memory(synthetic_video: Video) -> None:
    segments = get_video_segments(synthetic_video)
    intervals = [(s.start_s, s.end_s) for s in segments]
    clips = load_clips(synthetic_video, intervals, sample_fps=5.0)
    in_memory = generate_segment_visual_embeddings(ContentAligner(), clips)

    aligner = ContentAligner()
    chunked = generate_segment_visual_embeddings_chunked(
        aligner, synthetic_video, intervals, clips_per_batch=1
    )
    assert chunked.shape == in_memory.shape
    assert chunked == pytest.approx(in_memory)
    assert aligner.batch_sizes == [1] * len(segments)


def test_chunked_steps_validate_alignment(synthetic_video: Video) -> None:
    intervals = [(0.0, 2.0), (2.0, 4.0)]
    with pytest.raises(ValueError):
        generate_segment_captions_chunked(
            MisalignedCaptioner(), synthetic_video, intervals, clips_per_batch=2
        )


# --------------------------------------------------------------------------
# The context window: one decode, a widened clip, one prompt per position
# --------------------------------------------------------------------------


class PromptedCaptioner:
    """Records what each clip carried and what it was asked, in clip order."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, str | None]] = []
        self.indices: list[tuple[int, ...]] = []

    def caption(
        self, clips: Sequence[Clip], prompts: Sequence[str] | None = None
    ) -> list[str]:
        texts = list(prompts) if prompts is not None else [None] * len(clips)
        assert len(texts) == len(clips)
        for clip, text in zip(clips, texts):
            self.seen.append((len(clip.frames), text))
            self.indices.append(tuple(frame.index for frame in clip.frames))
        return [f"clip with {len(clip.frames)} frames" for clip in clips]


@pytest.fixture(scope="module")
def ten_second_video(tmp_path_factory: pytest.TempPathFactory) -> Video:
    """A 10-second video: five 2-second segments, enough for a k=1 window."""
    path = tmp_path_factory.mktemp("media") / "context.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48)
    )
    for index in range(300):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = index % 256
        writer.write(frame)
    writer.release()
    return Video(path=path, fps=30.0, frame_count=300)


FIVE_INTERVALS = [(2.0 * i, 2.0 * (i + 1)) for i in range(5)]


class TestContextPosition:
    def test_the_ends_of_the_recording_get_their_own_instruction(self) -> None:
        assert context_position(0, 5, 1) == "first"
        assert context_position(4, 5, 1) == "last"
        assert context_position(2, 5, 1) == "middle"

    def test_a_segment_short_of_context_on_both_sides_is_a_first(self) -> None:
        assert context_position(0, 1, 1) == "first"

    def test_without_context_every_segment_is_a_middle(self) -> None:
        assert context_position(3, 5, 0) == "middle"

    def test_the_frame_count_is_the_window_times_the_segment(self) -> None:
        counts = [context_frame_count(i, 5, 1, 10) for i in range(5)]

        assert counts == [20, 30, 30, 30, 20]


class TestContextClips:
    def test_the_widened_clip_carries_its_two_neighbours(
        self, ten_second_video: Video
    ) -> None:
        captioner = PromptedCaptioner()

        captions = generate_segment_captions_chunked(
            captioner,
            ten_second_video,
            FIVE_INTERVALS,
            clips_per_batch=2,
            context_segments=1,
        )

        assert [frames for frames, _ in captioner.seen] == [20, 30, 30, 30, 20]
        assert captions == [f"clip with {n} frames" for n in (20, 30, 30, 30, 20)]

    def test_the_frames_are_the_base_clips_in_time_order(
        self, ten_second_video: Video
    ) -> None:
        base = generate_segment_captions_chunked(
            (recorder := PromptedCaptioner()), ten_second_video, FIVE_INTERVALS
        )
        assert len(base) == 5
        plain = list(recorder.indices)

        widened = PromptedCaptioner()
        generate_segment_captions_chunked(
            widened, ten_second_video, FIVE_INTERVALS, context_segments=1
        )

        # The middle clip is exactly its three base clips, concatenated.
        assert widened.indices[2] == plain[1] + plain[2] + plain[3]
        assert widened.indices[0] == plain[0] + plain[1]
        assert widened.indices[4] == plain[3] + plain[4]
        for indices in widened.indices:
            assert list(indices) == sorted(indices)
            assert len(set(indices)) == len(indices)

    def test_each_position_is_asked_its_own_question(
        self, ten_second_video: Video
    ) -> None:
        captioner = PromptedCaptioner()
        prompts = [
            f"ask about {context_position(i, 5, 1)}" for i in range(5)
        ]

        generate_segment_captions_chunked(
            captioner,
            ten_second_video,
            FIVE_INTERVALS,
            clips_per_batch=2,
            context_segments=1,
            prompts=prompts,
        )

        assert [text for _, text in captioner.seen] == [
            "ask about first",
            "ask about middle",
            "ask about middle",
            "ask about middle",
            "ask about last",
        ]

    def test_a_recording_shorter_than_the_window_is_one_clip(
        self, ten_second_video: Video
    ) -> None:
        captioner = PromptedCaptioner()

        generate_segment_captions_chunked(
            captioner, ten_second_video, FIVE_INTERVALS[:2], context_segments=3
        )

        assert [frames for frames, _ in captioner.seen] == [20, 20]

    def test_without_context_the_captions_match_the_plain_path(
        self, ten_second_video: Video
    ) -> None:
        # ``FakeCaptioner`` takes no ``prompts`` at all — a captioner without
        # per-clip prompts — and the default call has to keep working on it.
        clips = load_clips(ten_second_video, FIVE_INTERVALS, sample_fps=5.0)
        in_memory = generate_segment_captions(FakeCaptioner(), clips)

        chunked = generate_segment_captions_chunked(
            FakeCaptioner(), ten_second_video, FIVE_INTERVALS, context_segments=0
        )

        assert chunked == in_memory == ["clip with 10 frames"] * 5

    def test_a_prompt_list_of_the_wrong_length_is_refused(
        self, ten_second_video: Video
    ) -> None:
        with pytest.raises(ValueError, match="prompts"):
            generate_segment_captions_chunked(
                PromptedCaptioner(),
                ten_second_video,
                FIVE_INTERVALS,
                prompts=["one", "two"],
            )

    def test_a_negative_window_is_refused(self, ten_second_video: Video) -> None:
        with pytest.raises(ValueError, match="context_segments"):
            generate_segment_captions_chunked(
                PromptedCaptioner(),
                ten_second_video,
                FIVE_INTERVALS,
                context_segments=-1,
            )

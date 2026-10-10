"""Temporal-memory construction steps: caption and embed each segment."""

from collections import deque
from typing import Iterable, Iterator, Sequence

import numpy as np

from video_agent.media import Clip, Frame, Video, iter_clips
from video_agent.models.interfaces import Captioner, TextEmbedder, VideoTextAligner


def _validate_embeddings(embeddings: np.ndarray, expected_rows: int) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"expected ({expected_rows}, D) embeddings, got {embeddings.shape}"
        )
    return embeddings


def generate_segment_captions(
    captioner: Captioner, clips: Sequence[Clip]
) -> list[str]:
    """Return one caption per clip, aligned by index."""
    captions = captioner.caption(clips)
    if len(captions) != len(clips):
        raise ValueError(f"expected {len(clips)} captions, got {len(captions)}")
    return captions


def generate_segment_caption_embeddings(
    embedder: TextEmbedder, captions: Sequence[str]
) -> np.ndarray:
    """Return caption embeddings of shape (len(captions), D)."""
    return _validate_embeddings(embedder.embed(captions), len(captions))


def generate_segment_visual_embeddings(
    aligner: VideoTextAligner, clips: Sequence[Clip]
) -> np.ndarray:
    """Return clip embeddings of shape (len(clips), D) in the aligner's space."""
    return _validate_embeddings(aligner.encode_video(clips), len(clips))


def _batched(clips: Iterable[Clip], batch_size: int) -> Iterator[list[Clip]]:
    """Group a stream of clips into lists of at most ``batch_size``."""
    batch: list[Clip] = []
    for clip in clips:
        batch.append(clip)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_clip_batches(
    video: Video,
    intervals: Sequence[tuple[float, float]],
    *,
    sample_fps: float,
    batch_size: int,
) -> Iterator[list[Clip]]:
    """Group streamed clips into lists of at most ``batch_size``."""
    return _batched(iter_clips(video, intervals, sample_fps=sample_fps), batch_size)


def context_position(i: int, n: int, k: int) -> str:
    """Which instruction segment ``i`` of ``n`` gets with ``k`` neighbours a side.

    A clip cut at the start of a recording carries no left-hand context and
    one cut at the end no right-hand one, so the sentence "describe the
    middle two seconds" would be wrong for both; each gets a prompt naming
    the part of the clip it actually is. A segment missing context on *both*
    sides (a recording shorter than the window) is captioned as a ``first``
    one: the described seconds are the clip's opening either way.
    """
    if n <= 0:
        raise ValueError(f"a recording has no segments to position: n={n}")
    if not 0 <= i < n:
        raise ValueError(f"segment {i} is outside the {n} segments")
    if k < 0:
        raise ValueError(f"context_segments must be >= 0, got {k}")
    missing_left = i - k < 0
    missing_right = i + k > n - 1
    if missing_left:
        return "first"
    if missing_right:
        return "last"
    return "middle"


def context_frame_count(i: int, n: int, k: int, frames_per_segment: int) -> int:
    """How many frames the context clip of segment ``i`` carries.

    The arithmetic of the window, spelled out so a test can assert on it
    without decoding a video: 30 frames in the middle and 20 at the ends for
    ``k = 1`` and ten frames a segment.
    """
    first = max(0, i - k)
    last = min(n - 1, i + k)
    return (last - first + 1) * int(frames_per_segment)


def _concatenate(clips: Iterable[Clip]) -> Clip:
    """One clip out of consecutive ones, frames kept in time order."""
    frames: list[Frame] = []
    for clip in clips:
        frames.extend(clip.frames)
    return Clip(frames=tuple(frames))


def _iter_context_clips(
    batches: Iterable[Sequence[Clip]], *, context_segments: int
) -> Iterator[Clip]:
    """Turn a stream of base clips into the stream of context clips.

    The window of segment ``i`` is ``i−k … i+k`` cut at the ends of the
    recording, so emitting it needs the clip ``k`` positions *ahead* of it —
    but the video is decoded once, forward, and a second pass over a 20-minute
    recording costs more than the captioning does. A deque of ``2k+1`` clips
    holds exactly what an emission can still need: as soon as clip ``j``
    arrives, segment ``j−k`` has both its sides in the buffer and is emitted,
    and the ``k`` last segments are emitted from what is left when the stream
    ends. Nothing but those ``2k+1`` decoded clips is ever held: an emitted
    clip carries *references* to frames the buffer already decoded, so the
    window and not the batch is what bounds the memory.
    """
    k = int(context_segments)
    window: deque[Clip] = deque(maxlen=2 * k + 1)
    last = -1  # index of the base clip most recently appended
    emitted = 0
    for batch in batches:
        for clip in batch:
            window.append(clip)
            last += 1
            while emitted <= last - k:
                yield _window_clip(window, last, emitted, k)
                emitted += 1
    while emitted <= last:
        yield _window_clip(window, last, emitted, k)
        emitted += 1


def _window_clip(window: Sequence[Clip], last: int, i: int, k: int) -> Clip:
    """The clip of segment ``i`` out of a buffer ending at base clip ``last``."""
    lowest = last - len(window) + 1
    first_wanted = max(lowest, i - k)
    last_wanted = min(last, i + k)
    return _concatenate(
        window[index - lowest] for index in range(first_wanted, last_wanted + 1)
    )


def generate_segment_captions_chunked(
    captioner: Captioner,
    video: Video,
    intervals: Sequence[tuple[float, float]],
    *,
    clip_sample_fps: float = 5.0,
    clips_per_batch: int = 8,
    context_segments: int = 0,
    prompts: Sequence[str] | None = None,
) -> list[str]:
    """Memory-bounded ``generate_segment_captions`` reading from disk.

    Streams clips in one sequential pass and captions them in batches of
    ``clips_per_batch``, so only one batch of decoded clips is ever in
    memory — materializing every clip of a long video costs GBs. Captions
    are per-clip, so results match the in-memory function.

    ``context_segments = k`` widens what the captioner sees without widening
    what it is asked about: the clip sent for interval ``i`` carries the
    frames of intervals ``i−k … i+k``, cut at the ends of the recording, and
    ``prompts[i]`` says which part of it the sentence is about. The video is
    still decoded exactly once (see ``_iter_context_clips``), and with
    ``k = 0`` and no prompts the calls are the ones this function has always
    made — the same clips, in the same batches, with the captioner's own
    instruction.
    """
    if context_segments < 0:
        raise ValueError(f"context_segments must be >= 0, got {context_segments}")
    if prompts is not None and len(prompts) != len(intervals):
        raise ValueError(f"{len(intervals)} intervals != {len(prompts)} prompts")
    base = _iter_clip_batches(
        video, intervals, sample_fps=clip_sample_fps, batch_size=clips_per_batch
    )
    batches: Iterable[Sequence[Clip]] = (
        base
        if context_segments == 0
        else _batched(
            _iter_context_clips(base, context_segments=context_segments),
            clips_per_batch,
        )
    )
    captions: list[str] = []
    for batch in batches:
        if prompts is None:
            batch_captions = captioner.caption(batch)
        else:
            batch_captions = captioner.caption(
                batch, prompts=list(prompts[len(captions) : len(captions) + len(batch)])
            )
        if len(batch_captions) != len(batch):
            raise ValueError(
                f"expected {len(batch)} captions, got {len(batch_captions)}"
            )
        captions.extend(batch_captions)
    return captions


def generate_segment_visual_embeddings_chunked(
    aligner: VideoTextAligner,
    video: Video,
    intervals: Sequence[tuple[float, float]],
    *,
    clip_sample_fps: float = 5.0,
    clips_per_batch: int = 8,
) -> np.ndarray:
    """Memory-bounded ``generate_segment_visual_embeddings`` reading from disk.

    Streams clips in one sequential pass and encodes them in batches of
    ``clips_per_batch``. Embeddings are per-clip, so results match the
    in-memory function.
    """
    chunks: list[np.ndarray] = []
    for batch in _iter_clip_batches(
        video, intervals, sample_fps=clip_sample_fps, batch_size=clips_per_batch
    ):
        chunks.append(_validate_embeddings(aligner.encode_video(batch), len(batch)))
    if not chunks:
        return np.zeros((0, 0))
    return np.concatenate(chunks, axis=0)

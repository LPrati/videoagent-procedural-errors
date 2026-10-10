"""Raw media primitives: videos, frames, clips.

Lowest layer of the package; has no knowledge of models or memories.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeAlias

import cv2
import numpy as np

Image: TypeAlias = np.ndarray
"""RGB image of shape (H, W, 3)."""


@dataclass(frozen=True)
class Video:
    """A video file and its decoding metadata."""

    path: Path
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        """Duration of the video in seconds."""
        return self.frame_count / self.fps


@dataclass(frozen=True)
class Frame:
    """A single decoded frame."""

    index: int  # position in the source video, not in a sampled sequence
    time_s: float
    data: Image


@dataclass(frozen=True)
class Clip:
    """Frames sampled from a contiguous interval of a video."""

    frames: tuple[Frame, ...]


def open_video(path: Path) -> Video:
    """Probe a video file for its decoding metadata."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return Video(path=path, fps=fps, frame_count=frame_count)


def _read_rgb(capture: cv2.VideoCapture) -> Image | None:
    success, frame = capture.read()
    if not success:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _retrieve_rgb(capture: cv2.VideoCapture) -> Image | None:
    """Retrieve after a successful ``grab()`` (skips no decode, only copy)."""
    success, frame = capture.retrieve()
    if not success:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _sample_interval(video_fps: float, sample_fps: float) -> int:
    # Rounded fps mirrors the reference implementation, which computes the
    # stride from the container's nominal (integer) frame rate.
    return max(1, math.ceil(round(video_fps) / sample_fps))


def load_frames(
    video: Video,
    sample_fps: float,
    frame_range: tuple[int, int] | None = None,
) -> list[Frame]:
    """Decode frames at ``sample_fps``, preserving source-video indices.

    ``frame_range`` restricts decoding to source indices ``[start, end)``.
    The sampling grid is anchored at frame 0 regardless of the window, so
    windowed calls concatenate to exactly the whole-video result.
    """
    interval = _sample_interval(video.fps, sample_fps)
    start, end = frame_range if frame_range is not None else (0, video.frame_count)
    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video.path}")
    if start > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[Frame] = []
    for index in range(start, end):
        # grab() decodes without the retrieve/convert copy; only sampled
        # frames pay for retrieval.
        if not capture.grab():
            break
        if index % interval == 0:
            data = _retrieve_rgb(capture)
            if data is None:
                break
            frames.append(Frame(index=index, time_s=index / video.fps, data=data))
    capture.release()
    return frames


_SEEK_GAP = 1000
"""Frames: seek only for jumps longer than this.

A seek costs a disk head move plus an h264 re-decode from the previous
keyframe, so grabbing forward wins for anything shorter, especially on
storage where random access is slow.
"""


def iter_frames_at(video: Video, indices: Iterable[int]) -> Iterator[Frame]:
    """Yield the given source-frame indices in one mostly-sequential pass.

    Opens the video once and scans forward, so callers can stream over
    thousands of scattered frames without either seeking per frame or
    holding them all in memory.
    """
    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video.path}")
    try:
        position = 0  # next frame index the capture will grab
        for index in sorted(set(indices)):
            if index < position or index - position > _SEEK_GAP:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                position = index
            ok = True
            while position <= index and ok:
                ok = capture.grab()
                position += 1
            if not ok:
                return
            data = _retrieve_rgb(capture)
            if data is None:
                return
            yield Frame(index=index, time_s=index / video.fps, data=data)
    finally:
        capture.release()


def load_frames_at(video: Video, indices: Iterable[int]) -> list[Frame]:
    """Decode exactly the given source-frame indices, sorted ascending."""
    return list(iter_frames_at(video, indices))


def iter_clips(
    video: Video,
    intervals: Sequence[tuple[float, float]],
    sample_fps: float | None = None,
) -> Iterator[Clip]:
    """Yield one clip per ``(start_s, end_s)`` interval, aligned by index.

    Streaming counterpart of ``load_clips``: only the clip currently being
    assembled is in memory, so callers can process the intervals of a long
    video in bounded batches. ``sample_fps`` subsamples frames within each
    interval; ``None`` keeps the native frame rate.
    """
    stride = 1 if sample_fps is None else _sample_interval(video.fps, sample_fps)
    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video.path}")
    try:
        position = 0  # next frame index the capture will decode
        for start_s, end_s in intervals:
            start = int(round(start_s * video.fps))
            end = min(int(round(end_s * video.fps)), video.frame_count)
            if start != position:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                position = start
            frames: list[Frame] = []
            for index in range(start, end):
                data = _read_rgb(capture)
                if data is None:
                    break
                position += 1
                if (index - start) % stride == 0:
                    frames.append(
                        Frame(index=index, time_s=index / video.fps, data=data)
                    )
            yield Clip(frames=tuple(frames))
    finally:
        capture.release()


def load_clips(
    video: Video,
    intervals: Sequence[tuple[float, float]],
    sample_fps: float | None = None,
) -> list[Clip]:
    """Decode one clip per ``(start_s, end_s)`` interval, aligned by index.

    All returned frames are materialized in memory at once; callers
    processing long videos should prefer ``iter_clips``.
    """
    return list(iter_clips(video, intervals, sample_fps))

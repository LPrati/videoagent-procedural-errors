from pathlib import Path

import pytest

from video_agent.media import Video
from video_agent.memory.preprocess import get_video_segments


def video(fps: float, frame_count: int) -> Video:
    return Video(path=Path("unused.mp4"), fps=fps, frame_count=frame_count)


def test_grid_covers_whole_segments_and_drops_the_tail() -> None:
    segments = get_video_segments(video(fps=30, frame_count=150 + 17))
    assert len(segments) == 2
    assert segments[0].frame_range == (0, 60)
    assert segments[1].frame_range == (60, 120)
    assert segments[1].index == 1
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 2.0


def test_grid_uses_rounded_fps_for_frames_and_true_fps_for_times() -> None:
    segments = get_video_segments(video(fps=29.97, frame_count=120))
    assert segments[0].frame_range == (0, 60)  # round(29.97) * 2
    assert segments[0].end_s == pytest.approx(60 / 29.97)


def test_grid_rejects_degenerate_segment() -> None:
    with pytest.raises(ValueError):
        get_video_segments(video(fps=30, frame_count=100), seconds_per_segment=0)

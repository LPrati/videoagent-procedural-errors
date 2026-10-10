"""Shared fixtures: a tiny synthetic video, and the captions of a perfect run."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from video_agent.eval.synthetic import IDLE_CAPTION
from video_agent.media import Video

FPS = 30
FRAME_COUNT = 150  # 5 seconds
WIDTH, HEIGHT = 64, 48


def frame_for_index(index: int) -> np.ndarray:
    """RGB frame whose red channel stores the frame index."""
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = index
    return frame


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Video:
    path: Path = tmp_path_factory.mktemp("media") / "synthetic.mp4"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )
    for index in range(FRAME_COUNT):
        writer.write(cv2.cvtColor(frame_for_index(index), cv2.COLOR_RGB2BGR))
    writer.release()
    return Video(path=path, fps=FPS, frame_count=FRAME_COUNT)


def synthesize_captions(recording, seconds_per_segment: float = 2.0) -> list[str]:
    """One caption per segment: each step's own description, inside its interval.

    The memory a perfect captioner would leave behind, reduced to what the
    localization tests read. Every timed step claims one segment it overlaps — the one
    holding its midpoint, or the nearest free one — so a ranking that works
    at all must put that segment first, and every other segment says nothing
    (``IDLE_CAPTION``) rather than describing the neighbouring step.
    """
    steps = [step for step in recording.steps if step.start_s is not None]
    count = math.ceil(max(step.end_s for step in steps) / seconds_per_segment) + 1
    captions = [IDLE_CAPTION] * count
    claimed: set[int] = set()
    for step in sorted(steps, key=lambda step: (step.start_s, step.step_id)):
        overlapping = [
            index
            for index in range(count)
            if index * seconds_per_segment < step.end_s
            and (index + 1) * seconds_per_segment > step.start_s
        ]
        middle = int((step.start_s + step.end_s) / 2 // seconds_per_segment)
        free = [index for index in overlapping if index not in claimed]
        chosen = min(free, key=lambda index: (abs(index - middle), index))
        claimed.add(chosen)
        captions[chosen] = step.description
    return captions

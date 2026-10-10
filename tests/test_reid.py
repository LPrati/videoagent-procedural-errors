import numpy as np
import pytest

from video_agent.memory.reid import (
    GreedyReIDMerger,
    ScoreChannel,
    reference_channels,
    sigmoid_calibration,
)
from video_agent.models.types import Detection, Tracklet


def tracklet(track_id: int, frame_indices: list[int]) -> Tracklet:
    return Tracklet(
        track_id=track_id,
        detections=tuple(
            Detection(
                frame_index=i, box_xyxy=(0.0, 0.0, 1.0, 1.0), category="cup", score=1.0
            )
            for i in frame_indices
        ),
    )


def unit(angle_rad: float) -> list[float]:
    return [np.cos(angle_rad), np.sin(angle_rad)]


SINGLE_CHANNEL = [ScoreChannel(dims=slice(0, 2), weight=1.0)]


def test_identical_disjoint_tracklets_merge() -> None:
    merger = GreedyReIDMerger(SINGLE_CHANNEL)
    mapping = merger.merge(
        [tracklet(10, [0, 1]), tracklet(20, [5, 6])],
        np.array([unit(0.0), unit(0.0)]),
    )
    assert mapping == {10: 0, 20: 0}


def test_cooccurring_tracklets_never_merge() -> None:
    merger = GreedyReIDMerger(SINGLE_CHANNEL)
    mapping = merger.merge(
        [tracklet(10, [0, 1]), tracklet(20, [1, 2])],
        np.array([unit(0.0), unit(0.0)]),
    )
    assert mapping == {10: 0, 20: 1}


def test_join_requires_all_above_floor_and_one_above_anchor() -> None:
    # cos(a, b) = 1.0; candidate c: cos vs a = cos(0.95) ~ 0.58 (>= 0.5, < 0.62),
    # cos vs c anchor b at cos(0.6) ~ 0.83 (>= 0.62) -> joins.
    merger = GreedyReIDMerger(SINGLE_CHANNEL)
    joined = merger.merge(
        [tracklet(1, [0]), tracklet(2, [1]), tracklet(3, [2])],
        np.array([unit(0.0), unit(0.35), unit(0.95)]),
    )
    assert joined == {1: 0, 2: 0, 3: 0}

    # Without an anchor >= 0.62 (all cosines in [0.5, 0.62)), no join happens.
    lonely = merger.merge(
        [tracklet(1, [0]), tracklet(2, [1])],
        np.array([unit(0.0), unit(1.0)]),  # cos(1.0) ~ 0.54
    )
    assert lonely == {1: 0, 2: 1}


def test_groups_are_numbered_by_first_appearance() -> None:
    merger = GreedyReIDMerger(SINGLE_CHANNEL)
    mapping = merger.merge(
        # Input order deliberately not appearance order.
        [tracklet(7, [50]), tracklet(8, [0]), tracklet(9, [10])],
        np.array([unit(0.0), unit(np.pi / 2), unit(np.pi)]),  # all orthogonal/opposed
    )
    assert mapping == {8: 0, 9: 1, 7: 2}


def test_reference_channels_reproduce_paper_score() -> None:
    channels = reference_channels(clip_dims=2, dinov2_dims=2)
    merger = GreedyReIDMerger(channels)
    # CLIP cosine exactly x0=0.925 -> sigmoid 0.5; DINOv2 cosine 1.0 (raw).
    clip_angle = np.arccos(0.925)
    embeddings = np.array(
        [unit(0.0) + unit(0.0), unit(clip_angle) + unit(0.0)]
    )
    scores = merger._pairwise_scores(embeddings)
    assert scores[0, 1] == pytest.approx(0.15 * 0.5 + 0.85 * 1.0)


def test_sigmoid_calibration_matches_reference_formula() -> None:
    calibrate = sigmoid_calibration(x0=0.925, slope=20.0)
    cosine = np.array(0.95)
    assert calibrate(cosine) == pytest.approx(1 / (1 + np.exp(-20 * (0.95 - 0.925))))

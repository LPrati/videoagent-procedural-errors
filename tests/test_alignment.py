import numpy as np
import pytest

from video_agent.memory.preprocess import assemble_object_memory
from video_agent.memory.types import Segment
from video_agent.models.types import Detection, Tracklet


def segment(index: int, frames_per_segment: int = 60) -> Segment:
    start = index * frames_per_segment
    return Segment(
        index=index,
        start_s=start / 30,
        end_s=(start + frames_per_segment) / 30,
        frame_range=(start, start + frames_per_segment),
    )


def tracklet(track_id: int, frame_indices: list[int], categories: list[str]) -> Tracklet:
    return Tracklet(
        track_id=track_id,
        detections=tuple(
            Detection(
                frame_index=f, box_xyxy=(0.0, 0.0, 1.0, 1.0), category=c, score=1.0
            )
            for f, c in zip(frame_indices, categories, strict=True)
        ),
    )


def test_alignment_groups_tracklets_into_records() -> None:
    segments = [segment(0), segment(1), segment(2)]
    tracklets = [
        tracklet(10, [0, 30], ["cup", "cup"]),
        tracklet(20, [70], ["bowl"]),
        tracklet(30, [130, 150], ["cup", "vase"]),  # same object as track 10
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 0.0]])

    memory = assemble_object_memory(
        segments, tracklets, {10: 0, 20: 1, 30: 0}, embeddings
    )

    assert [r.object_id for r in memory.records] == [0, 1]
    first, second = memory.records
    assert first.segment_indices == (0, 2)
    assert first.category == "cup"  # earliest tracklet names the object
    assert first.feature == pytest.approx([2.0, 0.0])  # mean of tracklet features
    assert [t.track_id for t in first.tracklets] == [10, 30]
    assert second.segment_indices == (1,)
    assert second.category == "bowl"


def test_detections_beyond_the_grid_are_dropped() -> None:
    segments = [segment(0)]
    tracklets = [tracklet(10, [10, 500], ["cup", "cup"])]
    memory = assemble_object_memory(
        segments, tracklets, {10: 0}, np.ones((1, 2))
    )
    assert memory.records[0].segment_indices == (0,)


def test_tracklet_category_is_majority_vote() -> None:
    segments = [segment(0)]
    tracklets = [tracklet(10, [0, 1, 2], ["cup", "vase", "vase"])]
    memory = assemble_object_memory(
        segments, tracklets, {10: 0}, np.ones((1, 2))
    )
    assert memory.records[0].category == "vase"

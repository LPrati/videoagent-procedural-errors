from typing import Sequence

import numpy as np
import pytest

from video_agent.media import Frame, Image, Video, load_frames
from video_agent.memory.object import (
    generate_object_visual_embeddings,
    generate_object_visual_embeddings_chunked,
)
from video_agent.models.types import Detection, Tracklet


class MeanColorEmbedder:
    """Embeds a crop as its mean RGB color — enough to check crop provenance."""

    def embed(self, images: Sequence[Image]) -> np.ndarray:
        return np.stack([image.reshape(-1, 3).mean(axis=0) for image in images])


def solid_frame(index: int, value: int) -> Frame:
    data = np.full((48, 64, 3), value, dtype=np.uint8)
    return Frame(index=index, time_s=index / 30, data=data)


def detection(frame_index: int, box=(8.0, 8.0, 32.0, 32.0)) -> Detection:
    return Detection(frame_index=frame_index, box_xyxy=box, category="cup", score=0.9)


def test_embeddings_average_sampled_crops_per_tracklet() -> None:
    frames = [solid_frame(0, 10), solid_frame(1, 20), solid_frame(2, 90)]
    tracklets = [
        Tracklet(track_id=1, detections=(detection(0), detection(1))),
        Tracklet(track_id=2, detections=(detection(2),)),
    ]
    embeddings = generate_object_visual_embeddings(
        MeanColorEmbedder(), frames, tracklets
    )
    assert embeddings.shape == (2, 3)
    assert embeddings[0] == pytest.approx([15.0, 15.0, 15.0])
    assert embeddings[1] == pytest.approx([90.0, 90.0, 90.0])


def test_sampling_caps_crops_and_is_deterministic() -> None:
    frames = [solid_frame(i, i) for i in range(20)]
    tracklets = [Tracklet(track_id=1, detections=tuple(detection(i) for i in range(20)))]

    first = generate_object_visual_embeddings(
        MeanColorEmbedder(), frames, tracklets, samples_per_tracklet=5, seed=0
    )
    second = generate_object_visual_embeddings(
        MeanColorEmbedder(), frames, tracklets, samples_per_tracklet=5, seed=0
    )
    other_seed = generate_object_visual_embeddings(
        MeanColorEmbedder(), frames, tracklets, samples_per_tracklet=5, seed=7
    )
    assert np.array_equal(first, second)
    # 5 of 20 solid frames sampled: the mean identifies which ones.
    assert not np.array_equal(first, other_seed)


def test_out_of_bounds_boxes_are_clamped() -> None:
    frames = [solid_frame(0, 50)]
    tracklets = [
        Tracklet(
            track_id=1,
            detections=(detection(0, box=(-10.0, -10.0, 1000.0, 1000.0)),),
        )
    ]
    embeddings = generate_object_visual_embeddings(
        MeanColorEmbedder(), frames, tracklets
    )
    assert embeddings[0] == pytest.approx([50.0, 50.0, 50.0])


def test_chunked_matches_in_memory(synthetic_video: Video) -> None:
    tracklets = [
        Tracklet(track_id=1, detections=tuple(detection(i) for i in range(0, 20))),
        Tracklet(track_id=2, detections=tuple(detection(i) for i in range(50, 62))),
    ]
    frames = load_frames(synthetic_video, sample_fps=1000)  # every frame

    in_memory = generate_object_visual_embeddings(MeanColorEmbedder(), frames, tracklets)
    chunked = generate_object_visual_embeddings_chunked(
        MeanColorEmbedder(), synthetic_video, tracklets, crops_per_batch=2
    )

    assert chunked.shape == in_memory.shape
    assert chunked == pytest.approx(in_memory, rel=1e-5)


def test_chunked_reports_undecodable_tracklets(synthetic_video: Video) -> None:
    beyond_end = Tracklet(track_id=1, detections=(detection(10_000),))
    with pytest.raises(ValueError, match="no decodable crop"):
        generate_object_visual_embeddings_chunked(
            MeanColorEmbedder(), synthetic_video, [beyond_end]
        )


def test_empty_tracklet_is_rejected() -> None:
    with pytest.raises(ValueError):
        generate_object_visual_embeddings(
            MeanColorEmbedder(), [], [Tracklet(track_id=1, detections=())]
        )

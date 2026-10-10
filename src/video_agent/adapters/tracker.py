"""Tracklet association with ByteTrack (roboflow trackers implementation)."""

from typing import Sequence

import numpy as np

from video_agent.media import Frame
from video_agent.models.types import Detection, Tracklet


def build_category_vocabulary(
    detections: Sequence[Sequence[Detection]],
) -> dict[str, int]:
    """Stable category -> class id mapping over all frames' detections."""
    return {
        category: class_id
        for class_id, category in enumerate(
            sorted({d.category for frame in detections for d in frame})
        )
    }


class ByteTrackTracker:
    """Tracker running ByteTrack over per-frame detections.

    Motion-only (frames are ignored), like the reference. ``tracker_kwargs``
    are forwarded to ``trackers.ByteTrackTracker`` (thresholds, buffer size,
    frame rate); a fresh tracker is created per ``track`` call.
    """

    def __init__(self, **tracker_kwargs) -> None:
        self.tracker_kwargs = tracker_kwargs

    def track(
        self, frames: Sequence[Frame], detections: Sequence[Sequence[Detection]]
    ) -> list[Tracklet]:
        """Return tracklets associated from per-frame detections."""
        import supervision as sv
        import trackers

        if len(frames) != len(detections):
            raise ValueError("frames and detections must be aligned")

        vocabulary = build_category_vocabulary(detections)
        categories = {class_id: category for category, class_id in vocabulary.items()}
        tracker = trackers.ByteTrackTracker(**self.tracker_kwargs)

        collected: dict[int, list[Detection]] = {}
        for frame, frame_detections in zip(frames, detections):
            if frame_detections:
                batch = sv.Detections(
                    xyxy=np.array([d.box_xyxy for d in frame_detections], dtype=np.float32),
                    confidence=np.array([d.score for d in frame_detections], dtype=np.float32),
                    class_id=np.array(
                        [vocabulary[d.category] for d in frame_detections], dtype=int
                    ),
                )
            else:
                batch = sv.Detections.empty()  # still ages lost-track buffers
            tracked = tracker.update(batch)
            if tracked.tracker_id is None:
                continue
            for box, class_id, score, track_id in zip(
                tracked.xyxy, tracked.class_id, tracked.confidence, tracked.tracker_id
            ):
                if track_id is None or int(track_id) < 0:
                    continue
                collected.setdefault(int(track_id), []).append(
                    Detection(
                        frame_index=frame.index,
                        box_xyxy=tuple(float(v) for v in box),
                        category=categories[int(class_id)],
                        score=float(score),
                    )
                )

        return [
            Tracklet(track_id=track_id, detections=tuple(track))
            for track_id, track in sorted(collected.items())
        ]

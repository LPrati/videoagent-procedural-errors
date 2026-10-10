"""Builds the temporal and object memories from a source video.

Pipeline layout follows VideoAgent (Fan et al., ECCV 2024, arXiv:2403.11481).
"""

from bisect import bisect_right
from collections import Counter
from typing import Mapping, Sequence

import numpy as np

from video_agent.media import Video, load_frames
from video_agent.memory.object import (
    generate_detection_info,
    generate_object_visual_embeddings,
    generate_tracking_info,
    postprocess_tracking_with_reid,
)
from video_agent.memory.temporal import (
    generate_segment_caption_embeddings,
    generate_segment_captions_chunked,
    generate_segment_visual_embeddings_chunked,
)
from video_agent.memory.types import (
    ObjectMemory,
    ObjectRecord,
    Segment,
    TemporalMemory,
)
from video_agent.models.interfaces import (
    Captioner,
    ImageEmbedder,
    ObjectDetector,
    ReIDMerger,
    TextEmbedder,
    Tracker,
    VideoTextAligner,
)
from video_agent.models.types import Tracklet


SECONDS_PER_SEGMENT = 2.0  # reference value; captions and features share this grid


def get_video_segments(
    video: Video, seconds_per_segment: float = SECONDS_PER_SEGMENT
) -> list[Segment]:
    """Return the segment grid; generate once per video and pass to both builders.

    Frame ranges are the ground truth: each segment spans exactly
    ``round(fps) * seconds_per_segment`` frames and a trailing partial segment
    is dropped, mirroring the reference implementation. Times are derived from
    the actual (fractional) fps.
    """
    frames_per_segment = int(round(video.fps) * seconds_per_segment)
    if frames_per_segment <= 0:
        raise ValueError(f"segment shorter than one frame at {video.fps} fps")
    count = video.frame_count // frames_per_segment
    segments = []
    for index in range(count):
        start = index * frames_per_segment
        end = start + frames_per_segment
        segments.append(
            Segment(
                index=index,
                start_s=start / video.fps,
                end_s=end / video.fps,
                frame_range=(start, end),
            )
        )
    return segments


def generate_temporal_memory(
    video: Video,
    segments: Sequence[Segment],
    *,
    captioner: Captioner,
    caption_embedder: TextEmbedder,
    aligner: VideoTextAligner,
    clip_sample_fps: float = 5.0,
    clips_per_batch: int = 8,
) -> TemporalMemory:
    """Build the temporal memory for ``segments``.

    Clips are streamed from disk in batches of ``clips_per_batch`` — one
    sequential pass for the captioner and another for the aligner — so memory
    is bounded by one batch instead of every clip in the video. The 5.0
    ``clip_sample_fps`` default yields 10 frames per 2-second segment — the
    reference aligner sampling; captioners subsample further as needed.
    """
    intervals = [(s.start_s, s.end_s) for s in segments]

    captions = generate_segment_captions_chunked(
        captioner,
        video,
        intervals,
        clip_sample_fps=clip_sample_fps,
        clips_per_batch=clips_per_batch,
    )
    caption_embeddings = generate_segment_caption_embeddings(
        caption_embedder, captions
    )
    visual_embeddings = generate_segment_visual_embeddings_chunked(
        aligner,
        video,
        intervals,
        clip_sample_fps=clip_sample_fps,
        clips_per_batch=clips_per_batch,
    )

    return TemporalMemory(
        segments=list(segments),
        captions=captions,
        caption_embeddings=caption_embeddings,
        visual_embeddings=visual_embeddings,
    )


def _tracklet_category(tracklet: Tracklet) -> str:
    """Majority vote over detection categories; ties go to the earliest seen."""
    counts = Counter(d.category for d in tracklet.detections)
    return counts.most_common(1)[0][0]


def _segment_index_by_frame(
    segments: Sequence[Segment], frame_index: int
) -> int | None:
    starts = [s.frame_range[0] for s in segments]
    position = bisect_right(starts, frame_index) - 1
    if position >= 0 and frame_index < segments[position].frame_range[1]:
        return segments[position].index
    return None  # frame beyond the grid (dropped trailing partial segment)


def assemble_object_memory(
    segments: Sequence[Segment],
    tracklets: Sequence[Tracklet],
    track_to_object: Mapping[int, int],
    tracklet_embeddings: np.ndarray,
) -> ObjectMemory:
    """Group tracklets by object id and map their detections to segment indices."""
    members: dict[int, list[int]] = {}
    for row, tracklet in enumerate(tracklets):
        members.setdefault(track_to_object[tracklet.track_id], []).append(row)

    records = []
    for object_id in sorted(members):
        rows = sorted(
            members[object_id],
            key=lambda row: min(d.frame_index for d in tracklets[row].detections),
        )
        segment_indices = sorted(
            {
                index
                for row in rows
                for d in tracklets[row].detections
                if (index := _segment_index_by_frame(segments, d.frame_index))
                is not None
            }
        )
        records.append(
            ObjectRecord(
                object_id=object_id,
                # Reference behavior: the earliest tracklet names the object.
                category=_tracklet_category(tracklets[rows[0]]),
                segment_indices=tuple(segment_indices),
                feature=tracklet_embeddings[rows].mean(axis=0),
                tracklets=tuple(tracklets[row] for row in rows),
            )
        )
    return ObjectMemory(records=records)


def generate_object_memory(
    video: Video,
    segments: Sequence[Segment],
    *,
    detector: ObjectDetector,
    tracker: Tracker,
    crop_embedder: ImageEmbedder,
    reid: ReIDMerger,
    detection_fps: float,
) -> ObjectMemory:
    """Build the object memory: detect, track, embed, re-identify, align."""
    frames = load_frames(video, sample_fps=detection_fps)

    detection_info = generate_detection_info(detector, frames)
    tracking_info = generate_tracking_info(tracker, frames, detection_info)
    tracklet_embeddings = generate_object_visual_embeddings(
        crop_embedder, frames, tracking_info
    )
    track_to_object = postprocess_tracking_with_reid(
        reid, tracking_info, tracklet_embeddings
    )

    return assemble_object_memory(
        segments, tracking_info, track_to_object, tracklet_embeddings
    )

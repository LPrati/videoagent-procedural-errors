"""Memory entities: built during preprocessing, queried at inference."""

from dataclasses import dataclass

import numpy as np

from video_agent.models.types import Tracklet


@dataclass(frozen=True)
class Segment:
    """Time interval of the source video; the index shared by both memories."""

    index: int
    start_s: float
    end_s: float
    frame_range: tuple[int, int]  # [start, end) in source-video frame indices


@dataclass
class TemporalMemory:
    """Columnar per-segment store: row ``i`` of every field describes ``segments[i]``."""

    segments: list[Segment]
    captions: list[str]
    caption_embeddings: np.ndarray  # (N, D), TextEmbedder space
    visual_embeddings: np.ndarray  # (N, D), VideoTextAligner space


@dataclass(frozen=True)
class ObjectRecord:
    """One re-identified object and the segments where it appears."""

    object_id: int
    category: str
    segment_indices: tuple[int, ...]
    feature: np.ndarray  # representative ImageEmbedder embedding
    tracklets: tuple[Tracklet, ...]


@dataclass
class ObjectMemory:
    """Store of re-identified objects across the whole video."""

    records: list[ObjectRecord]

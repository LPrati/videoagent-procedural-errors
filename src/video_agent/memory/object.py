"""Object-memory construction steps: detect, track, embed and re-identify."""

import random
from typing import Sequence

import cv2
import numpy as np

from video_agent.media import Frame, Image, Video, iter_frames_at
from video_agent.models.interfaces import (
    ImageEmbedder,
    ObjectDetector,
    ReIDMerger,
    Tracker,
)
from video_agent.models.types import Detection, Tracklet

CROP_SIZE = 224  # reference: crops are squashed to 224x224 before embedding
SAMPLES_PER_TRACKLET = 5  # reference: at most 5 crops represent a tracklet


def generate_detection_info(
    detector: ObjectDetector, frames: Sequence[Frame]
) -> list[list[Detection]]:
    """Return per-frame detections, aligned with ``frames``."""
    detections = detector.detect(frames)
    if len(detections) != len(frames):
        raise ValueError(
            f"expected detections for {len(frames)} frames, got {len(detections)}"
        )
    return detections


def generate_tracking_info(
    tracker: Tracker,
    frames: Sequence[Frame],
    detections: Sequence[Sequence[Detection]],
) -> list[Tracklet]:
    """Return tracklets associated from per-frame detections."""
    return tracker.track(frames, detections)


def _crop(image: Image, box_xyxy: tuple[float, float, float, float]) -> Image:
    height, width = image.shape[:2]
    x1 = min(max(int(box_xyxy[0]), 0), width - 1)
    y1 = min(max(int(box_xyxy[1]), 0), height - 1)
    x2 = min(max(int(box_xyxy[2]), x1 + 1), width)
    y2 = min(max(int(box_xyxy[3]), y1 + 1), height)
    return cv2.resize(image[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))


def sample_tracklet_detections(
    tracklets: Sequence[Tracklet],
    *,
    samples_per_tracklet: int = SAMPLES_PER_TRACKLET,
    seed: int = 0,
) -> list[list[Detection]]:
    """Pick the detections whose crops will represent each tracklet.

    Deterministic for a given seed, so callers may pre-compute the sample to
    decode only the needed frames, then embed with the same parameters.
    """
    rng = random.Random(seed)
    sampled = []
    for tracklet in tracklets:
        if not tracklet.detections:
            raise ValueError(f"tracklet {tracklet.track_id} has no detections")
        sampled.append(
            rng.sample(
                tracklet.detections,
                min(len(tracklet.detections), samples_per_tracklet),
            )
        )
    return sampled


def generate_object_visual_embeddings(
    embedder: ImageEmbedder,
    frames: Sequence[Frame],
    tracklets: Sequence[Tracklet],
    *,
    samples_per_tracklet: int = SAMPLES_PER_TRACKLET,
    seed: int = 0,
) -> np.ndarray:
    """Return one embedding per tracklet, shape (len(tracklets), D).

    For each tracklet, up to ``samples_per_tracklet`` detections are sampled
    at random, their boxes cropped and squashed to a fixed square, and the
    crop embeddings averaged — the reference representation of a tracklet.
    """
    frame_by_index = {frame.index: frame for frame in frames}
    sampled = sample_tracklet_detections(
        tracklets, samples_per_tracklet=samples_per_tracklet, seed=seed
    )

    crops: list[Image] = []
    counts: list[int] = []
    for chosen in sampled:
        for detection in chosen:
            image = frame_by_index[detection.frame_index].data
            crops.append(_crop(image, detection.box_xyxy))
        counts.append(len(chosen))

    if not crops:
        return np.zeros((0, 0))

    embeddings = embedder.embed(crops)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(crops):
        raise ValueError(f"expected ({len(crops)}, D) embeddings, got {embeddings.shape}")

    averaged = np.empty((len(tracklets), embeddings.shape[1]), dtype=embeddings.dtype)
    offset = 0
    for row, count in enumerate(counts):
        averaged[row] = embeddings[offset : offset + count].mean(axis=0)
        offset += count
    return averaged


def generate_object_visual_embeddings_chunked(
    embedder: ImageEmbedder,
    video: Video,
    tracklets: Sequence[Tracklet],
    *,
    samples_per_tracklet: int = SAMPLES_PER_TRACKLET,
    seed: int = 0,
    crops_per_batch: int = 256,
) -> np.ndarray:
    """Memory-bounded ``generate_object_visual_embeddings`` reading from disk.

    Streams the sampled frames in a single sequential pass, cropping each on
    arrival and accumulating per-tracklet embedding sums, so only one batch
    of crops is ever in memory — a video with thousands of tracklets would
    otherwise hold GBs of frames. Results match the in-memory function for
    the same seed.
    """
    sampled = sample_tracklet_detections(
        tracklets, samples_per_tracklet=samples_per_tracklet, seed=seed
    )
    by_frame: dict[int, list[tuple[int, Detection]]] = {}
    for row, chosen in enumerate(sampled):
        for detection in chosen:
            by_frame.setdefault(detection.frame_index, []).append((row, detection))

    totals: np.ndarray | None = None
    counts = np.zeros(len(tracklets), dtype=np.int64)
    crops: list[Image] = []
    rows: list[int] = []

    def flush() -> None:
        nonlocal totals
        if not crops:
            return
        embeddings = embedder.embed(crops)
        if totals is None:
            totals = np.zeros((len(tracklets), embeddings.shape[1]), dtype=np.float64)
        np.add.at(totals, rows, embeddings)
        np.add.at(counts, rows, 1)
        crops.clear()
        rows.clear()

    for frame in iter_frames_at(video, by_frame):
        for row, detection in by_frame[frame.index]:
            crops.append(_crop(frame.data, detection.box_xyxy))
            rows.append(row)
        if len(crops) >= crops_per_batch:
            flush()
    flush()

    if totals is None or not counts.all():
        missing = len(tracklets) if totals is None else int((counts == 0).sum())
        raise ValueError(f"{missing} tracklets got no decodable crop")
    return (totals / counts[:, None]).astype(np.float32)


def postprocess_tracking_with_reid(
    reid: ReIDMerger,
    tracklets: Sequence[Tracklet],
    tracklet_embeddings: np.ndarray,
) -> dict[int, int]:
    """Return a ``track_id -> object_id`` mapping merging co-identical tracklets."""
    if tracklet_embeddings.shape[0] != len(tracklets):
        raise ValueError(
            f"expected {len(tracklets)} embeddings, got {tracklet_embeddings.shape[0]}"
        )
    return reid.merge(tracklets, tracklet_embeddings)

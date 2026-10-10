"""Greedy tracklet re-identification, ported from the reference ``reid.py``.

The reference scores a pair of tracklets with a weighted ensemble of cosine
similarities over two embedding spaces (CLIP passed through a steep sigmoid,
DINOv2 raw), forbids merging tracklets that ever co-occur in a frame, and
greedily grows object groups in order of first appearance.

Thresholds and weights were calibrated for the reference embedders; treat
them as configuration to recalibrate when the embedding models change.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from video_agent.models.types import Tracklet

Calibration = Callable[[np.ndarray], np.ndarray]


def sigmoid_calibration(x0: float, slope: float) -> Calibration:
    """Map cosine similarity through ``1 / (1 + exp(-slope * (cos - x0)))``."""

    def calibrate(cosines: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-slope * (cosines - x0)))

    return calibrate


@dataclass(frozen=True)
class ScoreChannel:
    """One embedding subspace contributing to the pair score."""

    dims: slice  # columns of the embedding matrix holding this channel
    weight: float
    calibration: Calibration | None = None  # applied to the cosine; None = raw


def reference_channels(clip_dims: int, dinov2_dims: int) -> tuple[ScoreChannel, ...]:
    """The paper's ensemble over embeddings laid out as [CLIP | DINOv2]."""
    return (
        ScoreChannel(
            dims=slice(0, clip_dims),
            weight=0.15,
            calibration=sigmoid_calibration(x0=0.925, slope=20.0),
        ),
        ScoreChannel(
            dims=slice(clip_dims, clip_dims + dinov2_dims),
            weight=0.85,
        ),
    )


class GreedyReIDMerger:
    """Merges tracklets of the same physical object under a global object id."""

    def __init__(
        self,
        channels: Sequence[ScoreChannel],
        *,
        all_threshold: float = 0.5,
        any_threshold: float = 0.62,
    ) -> None:
        if not channels:
            raise ValueError("at least one score channel is required")
        self.channels = tuple(channels)
        self.all_threshold = all_threshold
        self.any_threshold = any_threshold

    def _pairwise_scores(self, embeddings: np.ndarray) -> np.ndarray:
        scores = np.zeros((embeddings.shape[0], embeddings.shape[0]))
        for channel in self.channels:
            part = embeddings[:, channel.dims].astype(np.float64)
            norms = np.linalg.norm(part, axis=1, keepdims=True)
            normalized = part / np.maximum(norms, np.finfo(np.float64).tiny)
            cosines = normalized @ normalized.T
            if channel.calibration is not None:
                cosines = channel.calibration(cosines)
            scores += channel.weight * cosines
        return scores

    @staticmethod
    def _cooccurrence(tracklets: Sequence[Tracklet]) -> np.ndarray:
        """Boolean (n, n) matrix: True where two tracklets share a frame."""
        frame_to_rows: dict[int, list[int]] = {}
        for row, tracklet in enumerate(tracklets):
            for detection in tracklet.detections:
                frame_to_rows.setdefault(detection.frame_index, []).append(row)
        together = np.zeros((len(tracklets), len(tracklets)), dtype=bool)
        for rows in frame_to_rows.values():
            for a in rows:
                for b in rows:
                    together[a, b] = True
        return together

    def merge(
        self, tracklets: Sequence[Tracklet], embeddings: np.ndarray
    ) -> dict[int, int]:
        """Return a ``Tracklet.track_id -> object_id`` mapping."""
        if embeddings.shape[0] != len(tracklets):
            raise ValueError(
                f"expected {len(tracklets)} embeddings, got {embeddings.shape[0]}"
            )
        track_ids = [t.track_id for t in tracklets]
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("tracklets must have unique track ids")

        scores = self._pairwise_scores(embeddings)
        scores[self._cooccurrence(tracklets)] = 0.0  # hard constraint

        # Greedy grouping in order of first appearance: a tracklet joins the
        # first group it scores >= all_threshold against every member of and
        # >= any_threshold against at least one member of.
        order = sorted(
            range(len(tracklets)),
            key=lambda row: min(d.frame_index for d in tracklets[row].detections),
        )
        groups: list[list[int]] = []
        row_to_object: dict[int, int] = {}
        for row in order:
            for object_id, members in enumerate(groups):
                member_scores = scores[row, members]
                if (
                    np.all(member_scores >= self.all_threshold)
                    and np.max(member_scores) >= self.any_threshold
                ):
                    members.append(row)
                    row_to_object[row] = object_id
                    break
            else:
                row_to_object[row] = len(groups)
                groups.append([row])

        return {track_ids[row]: object_id for row, object_id in row_to_object.items()}

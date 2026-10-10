"""Load memories back from per-stage artifacts (see ``memory.artifacts``).

Preprocessing writes one file per stage; inference needs the entities of
``memory.types`` again. Two things are not in the artifacts and are handled
here: the segment grid (recomputed from the video's fps and frame count,
persisted once as ``video_meta.json`` so evaluation machines without the
videos can still rebuild it) and L2 normalization of the embedding matrices
(caption embeddings already come normalized from the checkpoint; some
aligners' visual embeddings, X-CLIP's for example, do not).
"""

from bisect import bisect_right
from pathlib import Path
from typing import Sequence

import numpy as np

from video_agent.media import Video, open_video
from video_agent.memory.artifacts import VideoArtifacts
from video_agent.memory.preprocess import SECONDS_PER_SEGMENT, get_video_segments
from video_agent.memory.types import ObjectMemory, Segment, TemporalMemory

VIDEO_META_FILE = "video_meta.json"
VISUAL_EMBEDDINGS_FILE = "segment_visual_embeddings.npy"
OBJECT_MEMORY_FILE = "object_memory.pkl"

OBJECT_STAGES = ("detect", "track", "embed", "assemble")
"""The object pipeline's stages, in the order one feeds the next.

Named here because ``object_stage`` refuses anything else: a typo in a
stage name would write a lineage entry nobody reads and silently lose the
provenance of an artifact that was in fact produced.
"""

DEFAULT_ALIGNER_SLUG = "xclip-base-patch16"
"""Slug of ``adapters.aligner.DEFAULT_ALIGNER``, spelled here on purpose.

The store names artifacts and must not import an adapter to do it; a test
ties the two spellings together so they cannot drift. This is the aligner
whose embeddings keep the unsuffixed file name — not the aligner the agent
uses by default (``agent.config.DEFAULT_ALIGNER_MODEL``, whose file carries
its slug).
"""

DEFAULT_DETECTOR_SLUG = "rtdetr-v2-r50"
"""Slug of ``adapters.detector.DEFAULT_DETECTOR``, spelled here on purpose.

Same rule as ``DEFAULT_ALIGNER_SLUG``: the store names artifacts and must not
import an adapter to do it, so a test ties the two spellings together. This
is the detector whose object memory keeps the unsuffixed file names — not
necessarily the detector a run reads (``agent.config.ToolConfig.object_memory_slug``).
"""


def captions_file(variant: str) -> str:
    return f"captions.{variant}.json"


def caption_embeddings_file(variant: str) -> str:
    return f"caption_embeddings.{variant}.npy"


def visual_embeddings_file(slug: str | None = None) -> str:
    """Where one aligner's segment embeddings live in a video's directory.

    Several aligners can be run over the same recordings, so their
    embeddings have to coexist in one artifact root. Every aligner
    except X-CLIP (unsuffixed artifact name) is suffixed with its slug;
    X-CLIP uses ``segment_visual_embeddings.npy``.
    """
    if slug is None or slug == DEFAULT_ALIGNER_SLUG:
        return VISUAL_EMBEDDINGS_FILE
    return f"segment_visual_embeddings.{slug}.npy"


def visual_embed_stage(slug: str | None = None) -> str:
    """Lineage key of the stage that wrote ``visual_embeddings_file(slug)``.

    One key per aligner, for the same reason the file has one name per
    aligner: a second candidate must record its model, frame count and width
    without erasing the first's.
    """
    if slug is None or slug == DEFAULT_ALIGNER_SLUG:
        return "visual_embed"
    return f"visual_embed.{slug}"


def load_video_meta(artifacts: VideoArtifacts, video_path: Path | None = None) -> Video:
    """Return the video's decoding metadata, probing and caching it on first use.

    ``video_path`` overrides the stored path (datasets move between machines)
    and is required the first time, when the file has to be probed.
    """
    if artifacts.has(VIDEO_META_FILE):
        meta = artifacts.load_json(VIDEO_META_FILE)
        path = Path(video_path) if video_path is not None else Path(meta["path"])
        return Video(path=path, fps=float(meta["fps"]), frame_count=int(meta["frame_count"]))
    if video_path is None:
        raise FileNotFoundError(
            f"{VIDEO_META_FILE} missing in {artifacts.directory} and no video to probe"
        )
    video = open_video(Path(video_path))
    artifacts.save_json(
        VIDEO_META_FILE,
        {"path": str(video.path), "fps": video.fps, "frame_count": video.frame_count},
    )
    return video


def segment_grid(
    video: Video, n_segments: int, seconds_per_segment: float = SECONDS_PER_SEGMENT
) -> list[Segment]:
    """Rebuild the grid the artifacts were computed on and check it has ``n_segments``."""
    segments = get_video_segments(video, seconds_per_segment)
    if len(segments) < n_segments:
        raise ValueError(
            f"grid has {len(segments)} segments but artifacts have {n_segments}; "
            f"video metadata (fps={video.fps}, frames={video.frame_count}) may not match"
        )
    return segments[:n_segments]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length (zero rows stay zero); returns float32."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def load_temporal_memory(
    artifacts: VideoArtifacts,
    caption_variant: str,
    video: Video,
    *,
    aligner_slug: str | None = None,
    normalize: bool = True,
) -> TemporalMemory:
    """Assemble the temporal memory from the caption and visual artifacts.

    ``aligner_slug`` says whose visual embeddings to read; ``None`` is the
    unsuffixed file (X-CLIP). The file is checked
    before it is opened, and a missing one is an error naming it rather than
    a fall back to another aligner's: two aligners' embeddings live in
    different spaces, so reading the wrong file would silently score a step
    description against vectors it shares no geometry with.

    ``normalize`` L2-normalizes both embedding matrices so cosine similarity
    reduces to a dot product downstream; it is idempotent for the caption
    channel and required for the visual one.
    """
    captions = artifacts.load_json(captions_file(caption_variant))
    caption_embeddings = artifacts.load_array(caption_embeddings_file(caption_variant))
    visual_name = visual_embeddings_file(aligner_slug)
    if not artifacts.has(visual_name):
        raise FileNotFoundError(
            f"{visual_name} is missing for {artifacts.directory.name} "
            f"(in {artifacts.directory}): run preprocess_temporal.py "
            "--stages visual_embed with this aligner, or run with the "
            "aligner whose embeddings this recording has"
        )
    visual_embeddings = artifacts.load_array(visual_name)
    n = len(captions)
    for name, matrix in (
        ("caption embeddings", caption_embeddings),
        ("visual embeddings", visual_embeddings),
    ):
        if matrix.ndim != 2 or matrix.shape[0] != n:
            raise ValueError(f"{name} shape {matrix.shape} != ({n}, D) for {n} captions")
    if normalize:
        caption_embeddings = l2_normalize(caption_embeddings)
        visual_embeddings = l2_normalize(visual_embeddings)
    return TemporalMemory(
        segments=segment_grid(video, n),
        captions=[str(c) for c in captions],
        caption_embeddings=caption_embeddings,
        visual_embeddings=visual_embeddings,
    )


def object_suffix(slug: str | None) -> str:
    """What a detector's slug adds to an object artifact's name, ``""`` for none.

    One place decides it because four file names and four lineage keys have
    to agree; the default detector (RT-DETRv2) adds nothing, so its memory keeps
    the unsuffixed file names.
    """
    if slug is None or slug == DEFAULT_DETECTOR_SLUG:
        return ""
    return f".{slug}"


def detections_file(slug: str | None = None) -> str:
    """Per-frame detections of one detector (the detect stage's artifact)."""
    return f"detections{object_suffix(slug)}.pkl"


def tracklets_file(slug: str | None = None) -> str:
    """Tracklets associated from one detector's detections."""
    return f"tracklets{object_suffix(slug)}.pkl"


def tracklet_embeddings_file(slug: str | None = None) -> str:
    """Crop embeddings of one detector's tracklets, one row per tracklet."""
    return f"tracklet_embeddings{object_suffix(slug)}.npy"


def object_memory_file(slug: str | None = None) -> str:
    """Where one detector's assembled object memory lives.

    Two detectors produce
    different objects with different names from the same video, so they are
    different memories and must not share a file (see
    ``adapters.detector.object_memory_slug``).
    """
    return f"object_memory{object_suffix(slug)}.pkl"


def object_stage(name: str, slug: str | None = None) -> str:
    """Lineage key of one object stage of one detector.

    Suffixed like the artifact it records, so a second detector writes its
    model and its threshold into ``meta.json`` without
    erasing the first's — the same rule ``visual_embed_stage`` follows for
    the aligners.
    """
    if name not in OBJECT_STAGES:
        raise ValueError(
            f"unknown object stage {name!r}; the stages are "
            + ", ".join(OBJECT_STAGES)
        )
    return f"{name}{object_suffix(slug)}"


def load_object_memory(
    artifacts: VideoArtifacts, slug: str | None = None
) -> ObjectMemory:
    """Read one detector's object memory; ``None`` is the unsuffixed file.

    Missing is an error naming both the file and the recording, never a fall
    back to another detector's memory: two detectors can name different
    categories, and an answer read from the wrong memory would look like a
    result.
    """
    name = object_memory_file(slug)
    if not artifacts.has(name):
        raise FileNotFoundError(
            f"{name} is missing for {artifacts.directory.name} "
            f"(in {artifacts.directory}): run preprocess_objects.py "
            "--detector-model with this detector, or run with the memory "
            "this recording has"
        )
    memory = artifacts.load_pickle(name)
    if not isinstance(memory, ObjectMemory):
        raise ValueError(f"{name} holds {type(memory).__name__}, not ObjectMemory")
    return memory


def crop_embedding_widths(
    artifacts: VideoArtifacts, slug: str | None = None
) -> tuple[int, ...]:
    """Column widths of ``ObjectRecord.feature`` per member embedder, from lineage.

    Per detector, because each one's tracklets were embedded by their own
    run of the embed stage; the widths decide which columns of the feature
    are text-aligned, so reading another memory's entry would slice the
    wrong half.
    """
    stage = object_stage("embed", slug)
    meta = artifacts.stage_meta(stage)
    if meta is None or "widths" not in meta.get("params", {}):
        raise ValueError(f"no {stage!r} stage widths recorded in {artifacts.directory}")
    return tuple(int(w) for w in meta["params"]["widths"])


def object_memory_slugs(artifacts: VideoArtifacts) -> list[str | None]:
    """Which detectors left an object memory here; ``None`` is RT-DETRv2's.

    Read off the file names rather than off ``meta.json`` so a directory
    copied without its lineage still answers, and so the answer is what can
    actually be loaded. ``None`` sorts first because it is the default
    detector's memory.
    """
    prefix, suffix = "object_memory", ".pkl"
    slugs: list[str | None] = []
    for path in sorted(artifacts.directory.glob(f"{prefix}*{suffix}")):
        middle = path.name[len(prefix) : -len(suffix)]
        if middle == "":
            slugs.append(None)
        elif middle.startswith("."):
            slugs.append(middle[1:])
    return sorted(slugs, key=lambda slug: (slug is not None, slug or ""))


def segment_index_at(segments: Sequence[Segment], time_s: float) -> int | None:
    """Index of the segment containing ``time_s``; ``None`` beyond the grid."""
    position = bisect_right([s.start_s for s in segments], time_s) - 1
    if position < 0 or time_s >= segments[position].end_s:
        return None
    return segments[position].index


def segment_interval(
    segments: Sequence[Segment], start_index: int, end_index: int
) -> tuple[float, float]:
    """Time span ``[start, end)`` covered by segments ``start_index..end_index``, clamped.

    The span is the earliest start and the latest end of the rows in the
    range, which on the ordered grid every real memory has is exactly
    ``segments[start].start_s`` and ``segments[end].end_s`` — the rows are
    consecutive intervals. It is spelled as a minimum and a maximum for the
    one grid that is not ordered: a shuffled memory (``memory.ablation``)
    holds the same intervals under other names, and reading its first and
    last row would return a span that ends before it starts.
    """
    if not segments:
        raise ValueError("empty segment grid")
    last = len(segments) - 1
    start_index = min(max(int(start_index), 0), last)
    end_index = min(max(int(end_index), 0), last)
    if end_index < start_index:
        raise ValueError(f"end segment {end_index} precedes start segment {start_index}")
    window = segments[start_index : end_index + 1]
    return min(s.start_s for s in window), max(s.end_s for s in window)

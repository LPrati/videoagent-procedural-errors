"""Batch object-memory preprocessing with per-stage artifacts and resume.

Stages per video (each skipped when its artifact already exists):
    detect   -> detections[.slug].pkl        (GPU, chunked decoding; the slow one)
    track    -> tracklets[.slug].pkl         (CPU, ByteTrack over saved detections)
    embed    -> tracklet_embeddings[.slug].npy (GPU, decodes only sampled frames)
    assemble -> object_memory[.slug].pkl     (CPU, re-ID + segment alignment)

The pipeline is RT-DETRv2 (COCO classes) + ByteTrack + SigLIP 2/DINOv2 crop
embeddings with greedy re-identification. Its artifacts carry unsuffixed
names; a detector added to ``adapters.detector``'s registry would suffix its
artifacts with its slug, so several object memories can live side by side in
one artifact root.

Usage:
    uv run python scripts/preprocess_objects.py --videos-dir VIDEOS_DIR \\
        --out PREPROCESS_DIR
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

from video_agent.adapters.detector import (
    DEFAULT_DETECTOR,
    build_detector,
    detector_spec,
    known_detectors,
    object_memory_slug,
)
from video_agent.lineage import code_version, library_versions
from video_agent.media import Frame, load_frames, open_video
from video_agent.memory.artifacts import VideoArtifacts
from video_agent.memory.object import (
    SAMPLES_PER_TRACKLET,
    generate_object_visual_embeddings_chunked,
)
from video_agent.memory.preprocess import assemble_object_memory, get_video_segments
from video_agent.memory.reid import GreedyReIDMerger, reference_channels
from video_agent.memory.store import (
    OBJECT_STAGES,
    detections_file,
    object_memory_file,
    object_stage,
    tracklet_embeddings_file,
    tracklets_file,
)

DETECTION_FPS = 15.0
"""Default detection rate, as the reproduction sampled: two frames a second
is not enough for ByteTrack to associate a hand-held object, and the whole
frame budget of a run is linear in this number, so it is a flag."""

CROP_SEED = 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument("--videos-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--window-s",
        type=float,
        default=60.0,
        help="detection decoding window; bounds RAM (frames are materialized)",
    )
    parser.add_argument("--embed-chunk", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--shard",
        help="I/N: process only videos where index %% N == I (parallel workers)",
    )
    parser.add_argument(
        "--force-stages",
        default="",
        help="comma-separated stages to recompute even if cached "
        "(detect,track,embed,assemble); downstream stages recompute too",
    )
    parser.add_argument(
        "--detector-model",
        default=DEFAULT_DETECTOR,
        help="detector of the detect stage (see adapters/detector.py for the "
        "registry)",
    )
    parser.add_argument(
        "--detection-fps",
        type=float,
        default=DETECTION_FPS,
        help="frames per second handed to the detector and to the tracker",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        help="detection score cut; defaults to the detector's own (0.25 for "
        "RT-DETRv2)",
    )
    parser.add_argument(
        "--track-activation-threshold",
        type=float,
        help="ByteTrack score a detection needs to start a track; defaults to "
        "the trackers library's own (0.7)",
    )
    parser.add_argument(
        "--track-high-conf-threshold",
        type=float,
        help="ByteTrack score above which a detection joins the first "
        "(high-confidence) association pass; defaults to the library's own (0.6)",
    )
    parser.add_argument("--detector-batch", type=int, default=16)
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--dinov2-model", default="facebook/dinov2-base")
    args = parser.parse_args(argv)

    # Resolve the detector and the slug of its memory here, so a typo costs
    # milliseconds instead of a checkpoint download and a decoded video.
    try:
        detector_spec(args.detector_model)
    except ValueError as error:
        parser.error(f"{error} (known detectors: {', '.join(known_detectors())})")
    args.object_slug = object_memory_slug(args.detector_model)
    return args


class LazyModels:
    """Load each model on first use so skipped stages cost nothing."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._detector = None
        self._crop_embedder = None

    @property
    def detector(self):
        if self._detector is None:
            self._detector = build_detector(
                self.args.detector_model,
                device=self.args.device,
                batch_size=self.args.detector_batch,
                score_threshold=self.args.score_threshold,
            )
        return self._detector

    @property
    def crop_embedder(self):
        if self._crop_embedder is None:
            from video_agent.adapters.image_embedder import (
                ConcatImageEmbedder,
                Dinov2ImageEmbedder,
                SiglipImageEmbedder,
            )

            self._crop_embedder = ConcatImageEmbedder(
                [
                    SiglipImageEmbedder(self.args.siglip_model, device=self.args.device),
                    Dinov2ImageEmbedder(self.args.dinov2_model, device=self.args.device),
                ]
            )
        return self._crop_embedder


def cuda_torch():
    """``torch`` when this process can measure GPU memory, else ``None``.

    Imported here and never at module level: the argument parser, the resume
    logic and the tests all run on machines with neither torch nor a GPU,
    and none of them should pay a model library to learn a file name.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        return torch if torch.cuda.is_available() else None
    except (RuntimeError, AssertionError):
        return None


def peak_vram_bytes(torch_module) -> int | None:
    """Largest allocation this process reached on the GPU, or ``None``."""
    if torch_module is None:
        return None
    return int(torch_module.cuda.max_memory_allocated())


def stage_detect(video, artifacts: VideoArtifacts, models: LazyModels, args, version, slug):
    """Detect over the sampled frames and save the detections.

    Decoding and detection are timed apart because they compete for very
    different resources: the detector's own throughput is what sizes a full
    run, and a number that includes OpenCV's decoding would credit the GPU
    with the CPU's work.
    """
    torch_module = cuda_torch()
    if torch_module is not None:
        torch_module.cuda.reset_peak_memory_stats()
    detector = models.detector

    window = int(round(video.fps) * args.window_s)
    frame_indices: list[int] = []
    per_frame: list[list] = []
    decode_s = 0.0
    detect_s = 0.0
    started = time.perf_counter()
    for start in range(0, video.frame_count, window):
        clock = time.perf_counter()
        frames = load_frames(
            video, args.detection_fps, frame_range=(start, start + window)
        )
        decode_s += time.perf_counter() - clock
        clock = time.perf_counter()
        detections = detector.detect(frames)
        detect_s += time.perf_counter() - clock
        frame_indices.extend(frame.index for frame in frames)
        per_frame.extend(detections)

    saved = {"frame_indices": frame_indices, "per_frame": per_frame}
    name = detections_file(slug)
    artifacts.save_pickle(name, saved)

    stage_s = time.perf_counter() - started
    n_frames = len(frame_indices)
    artifacts.record_stage(
        object_stage("detect", slug),
        artifact=name,
        model=args.detector_model,
        params={
            "detection_fps": args.detection_fps,
            **detector.lineage(),
            "timing": {
                "n_frames": n_frames,
                "decode_s": round(decode_s, 3),
                "detect_s": round(detect_s, 3),
                "stage_s": round(stage_s, 3),
                "frames_per_s_detect": (n_frames / detect_s) if detect_s else None,
                "frames_per_s_stage": (n_frames / stage_s) if stage_s else None,
                "peak_vram_bytes": peak_vram_bytes(torch_module),
            },
            "library_versions": library_versions(("torch", "transformers")),
        },
        code_version=version,
    )
    return sum(len(d) for d in per_frame)


def stage_track(video, artifacts: VideoArtifacts, args, version, slug):
    from video_agent.adapters.tracker import ByteTrackTracker

    saved = artifacts.load_pickle(detections_file(slug))
    # ByteTrack is motion-only: it never reads pixels, only frame identity,
    # so placeholder images stand in for the (huge) decoded frames.
    placeholder = np.zeros((1, 1, 3), dtype=np.uint8)
    frames = [
        Frame(index=index, time_s=index / video.fps, data=placeholder)
        for index in saved["frame_indices"]
    ]
    # frame_rate must match the rate the frames were sampled at: roboflow
    # trackers defaults to 30, which would size the lost-track buffer at
    # half the intended duration. The detect stage's own record is the
    # authority, since a resumed run may carry another flag.
    meta = artifacts.stage_meta(object_stage("detect", slug)) or {}
    frame_rate = (meta.get("params") or {}).get("detection_fps", args.detection_fps)
    # Only thresholds the caller set are forwarded, so an unset flag keeps
    # the library's own default; the record below still spells out the value
    # that ran, because "default" is not a number two memories can be
    # compared by.
    thresholds = {
        "track_activation_threshold": getattr(args, "track_activation_threshold", None),
        "high_conf_det_threshold": getattr(args, "track_high_conf_threshold", None),
    }
    tracker_kwargs = {key: value for key, value in thresholds.items() if value is not None}
    tracklets = ByteTrackTracker(frame_rate=frame_rate, **tracker_kwargs).track(
        frames, saved["per_frame"]
    )
    name = tracklets_file(slug)
    artifacts.save_pickle(name, tracklets)
    artifacts.record_stage(
        object_stage("track", slug),
        artifact=name,
        model="bytetrack (roboflow trackers)",
        params={"frame_rate": frame_rate, **resolved_tracker_thresholds(tracker_kwargs)},
        code_version=version,
    )
    return len(tracklets)


def resolved_tracker_thresholds(tracker_kwargs: dict) -> dict:
    """The ByteTrack thresholds that actually ran, defaults filled in.

    Read off the library's signature rather than copied here, so the record
    stays true if ``trackers`` changes its defaults under us.
    """
    import inspect

    import trackers

    defaults = inspect.signature(trackers.ByteTrackTracker.__init__).parameters
    return {
        key: float(tracker_kwargs.get(key, defaults[key].default))
        for key in ("track_activation_threshold", "high_conf_det_threshold")
    }


def stage_embed(video, artifacts: VideoArtifacts, models: LazyModels, args, version, slug):
    tracklets = artifacts.load_pickle(tracklets_file(slug))
    embeddings = generate_object_visual_embeddings_chunked(
        models.crop_embedder,
        video,
        tracklets,
        seed=CROP_SEED,
        crops_per_batch=args.embed_chunk,
    )
    name = tracklet_embeddings_file(slug)
    artifacts.save_array(name, embeddings)
    artifacts.record_stage(
        object_stage("embed", slug),
        artifact=name,
        model=f"{args.siglip_model} + {args.dinov2_model}",
        params={
            "widths": list(models.crop_embedder.widths),
            "samples_per_tracklet": SAMPLES_PER_TRACKLET,
            "seed": CROP_SEED,
        },
        code_version=version,
    )
    return embeddings.shape


def stage_assemble(video, artifacts: VideoArtifacts, version, slug):
    tracklets = artifacts.load_pickle(tracklets_file(slug))
    embeddings = artifacts.load_array(tracklet_embeddings_file(slug))
    widths = artifacts.stage_meta(object_stage("embed", slug))["params"]["widths"]
    merger = GreedyReIDMerger(reference_channels(*widths))
    track_to_object = merger.merge(tracklets, embeddings)
    memory = assemble_object_memory(
        get_video_segments(video), tracklets, track_to_object, embeddings
    )
    name = object_memory_file(slug)
    artifacts.save_pickle(name, memory)
    artifacts.record_stage(
        object_stage("assemble", slug),
        artifact=name,
        model="greedy re-ID (reference channels and thresholds)",
        params={"widths": widths},
        code_version=version,
    )
    return len(memory.records)


def main() -> None:
    args = parse_args()
    videos = list(args.videos)
    if args.videos_dir:
        videos.extend(sorted(args.videos_dir.glob("*.mp4")))
    if args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        videos = videos[index::total]
    if args.limit:
        videos = videos[: args.limit]
    # A forced stage invalidates everything downstream of it.
    order = list(OBJECT_STAGES)
    forced = {stage for stage in args.force_stages.split(",") if stage}
    if unknown := forced - set(order):
        raise SystemExit(f"unknown stages: {sorted(unknown)}")
    if forced:
        forced.update(order[min(order.index(stage) for stage in forced) :])
    version = code_version()
    slug = args.object_slug
    models = LazyModels(args)
    print(
        f"{len(videos)} videos -> {args.out} (code {version}, "
        f"object memory {slug})",
        flush=True,
    )

    failures = 0
    for position, path in enumerate(videos, 1):
        started = time.monotonic()
        try:
            video = open_video(path)
            artifacts = VideoArtifacts(args.out, path.stem)
            done = []
            if "detect" in forced or not artifacts.has(detections_file(slug)):
                count = stage_detect(video, artifacts, models, args, version, slug)
                done.append(f"detect={count}")
            if "track" in forced or not artifacts.has(tracklets_file(slug)):
                count = stage_track(video, artifacts, args, version, slug)
                done.append(f"track={count}")
            if "embed" in forced or not artifacts.has(tracklet_embeddings_file(slug)):
                shape = stage_embed(video, artifacts, models, args, version, slug)
                done.append(f"embed={shape}")
            if "assemble" in forced or not artifacts.has(object_memory_file(slug)):
                count = stage_assemble(video, artifacts, version, slug)
                done.append(f"objects={count}")
            elapsed = time.monotonic() - started
            status = ", ".join(done) if done else "all stages cached"
            print(f"[{position}/{len(videos)}] {path.stem}: {status} "
                  f"({elapsed:.0f}s)", flush=True)
        except Exception:
            failures += 1
            print(f"[{position}/{len(videos)}] {path.stem}: FAILED", flush=True)
            traceback.print_exc()
    print(f"done, {failures} failures", flush=True)


if __name__ == "__main__":
    main()

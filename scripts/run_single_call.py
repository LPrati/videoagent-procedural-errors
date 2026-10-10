#!/usr/bin/env python3
"""Single-call systems: the whole video, or all of its captions, in one request.

The no-agent, no-memory, no-tools comparison systems. Each recording becomes
at most two chat completions against the served model: "was the recipe
followed?" (``global.json``) and the agent's per-step question (``run.json``).
The records have the shape ``video_agent.task.run.make_record`` writes, so
``scripts/evaluate.py`` scores these runs exactly as it scores the agent's.

Two systems (``--arm``):

* ``captions``: every caption of the temporal memory, one per line as
  ``[segment, start-end s] caption``, in a single text request.
* ``video``: the whole recording resampled to ``--fps`` at ``--width`` x
  ``--height`` and sent as one video input, within ``--visual-token-budget``
  visual tokens; a longer recording is sampled at a proportionally lower
  rate. The model answers in seconds, which are mapped onto the 2-second
  segment grid.

Three details are handled here rather than in the package:

* **The clock.** The model reads the timestamps the processor derives from
  the container, so the resampled MP4 must declare the rate its frames were
  actually taken at. The rate is quantized to what an MP4 carries exactly
  (``FPS_DECIMALS``), and that one number picks the timestamps, is written
  into the container and is rendered into the question. An MP4 declaring
  2 fps while carrying one frame per second would scale every time anchor
  by two with nothing in the record to show it.
* **Video before text.** Both questions share one video prefix, so the video
  parts come first in the user message and the question last: the second
  call reuses the server's prefix cache instead of re-encoding the video.
* **Seconds, not segments.** With the frames in front of it the model
  answers in seconds; ``spans_to_segments`` maps those onto the 2-second
  grid the evaluation scores on before ``verdicts.normalize`` sees them.

``--provider vllm`` (default) talks to the local vLLM server
(``serving/vllm/serve_single_call.sh``) with thinking disabled.
``--provider openai-compatible`` runs the captions system against a hosted
model behind an OpenAI-compatible endpoint (``--base-url``, ``--model``, the
API key in the variable named by ``--api-key-env``).

Usage, whole video in a single call::

    uv run python scripts/run_single_call.py --split data_splits/dev_83.json \\
        --arm video --fps 1 --width 448 --height 256 \\
        --visual-token-budget 56000 \\
        --artifacts-root PREPROCESS_DIR --videos-dir VIDEOS_DIR \\
        --out RUNS_DIR --run-name single-call-video \\
        --max-tokens 8192 --temperature 0.0 --seed 20260919

Captions in a single call::

    uv run python scripts/run_single_call.py --split data_splits/dev_83.json \\
        --arm captions --caption-variant abst_ctx \\
        --artifacts-root PREPROCESS_DIR --videos-dir VIDEOS_DIR \\
        --out RUNS_DIR --run-name single-call-captions \\
        --max-tokens 8192 --temperature 0.0 --seed 20260919

Exit codes: 0 when every recording was answered, 3 when the batch finished
with failed recordings (their ``error`` records are written), and the usual
1/2 for a crash or a usage error.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
from _env import load_dotenv

from video_agent.agent.transcript import LLMTurn, ReActStep, Transcript
from video_agent.lineage import code_version
from video_agent.media import Video
from video_agent.memory.artifacts import VideoArtifacts
from video_agent.memory.preprocess import SECONDS_PER_SEGMENT, get_video_segments
from video_agent.memory.store import captions_file, load_video_meta, segment_grid
from video_agent.memory.types import Segment
from video_agent.models.types import Usage
from video_agent.task.annotations import AnnotationSet, Sop, TaskGraph, load_split
# The four underscored names are private to the package and are imported on
# purpose. The question must list the steps and the prerequisites exactly as
# the agent's question does, so that the comparison is between two systems
# and not between two wordings; and the answer must be read by the same JSON
# scanner, so that "the last object" means the same thing in every run.
# ``build_question`` itself cannot be reused: its placeholder set is neither
# arm's.
from video_agent.task.prompt import (
    ANSWER_SCHEMA,
    _FENCE,
    _json_objects,
    _render_prerequisites,
    _render_steps,
    final_answer_candidates,
    parse_final_answer,
)
from video_agent.task.run import (
    CONFIG_ID_LENGTH,
    RUN_FILE,
    RUN_SCHEMA_VERSION,
    RunStore,
)
from video_agent.task.verdicts import normalize

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PROMPTS_DIR = REPO_ROOT / "prompts" / "single_call"
DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "data" / "captaincook4d" / "annotations"

SYSTEM = "single_call"
"""Stored under the configuration key ``"system"`` of every record."""
VIDEO_SUFFIX = "_360p"
"""The videos are the 360p renders, and artifacts are keyed by their stem."""

ARM_VIDEO = "video"
ARM_CAPTIONS = "captions"
ARMS = (ARM_VIDEO, ARM_CAPTIONS)

QUESTION_STEPS = "steps"
QUESTION_GLOBAL = "global"
QUESTION_ORDER = (QUESTION_GLOBAL, QUESTION_STEPS)
"""The global question runs first, always.

Both calls carry the same video prefix, so the order costs nothing, and the
global verdict is never written after the model has already given its
per-step verdicts.
"""

GLOBAL_FILE = "global.json"
GLOBAL_SCHEMA_VERSION = 1
"""The global answer lives beside ``run.json`` (``task.run.RUN_FILE``), in the
recording's own directory of the run."""

DEFAULT_VISUAL_TOKEN_BUDGET = 56_000
"""The most that fits on a 32 GB GPU with ``serving/vllm/serve_single_call.sh``."""
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_SEED = 20260919
DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_CAPTION_VARIANT = "abst_ctx"

PIXELS_PER_TOKEN_SIDE = 32
FRAMES_PER_TOKEN = 2
"""How the Qwen3-VL/3.5 processor packs pixels: one visual token per 32x32
patch of a *pair* of consecutive frames. Both numbers are the estimate's only
assumption, and ``input.est_visual_tokens`` is written beside the server's
own ``prompt_tokens`` so that the assumption can be checked against a run."""

FPS_DECIMALS = 3
"""Decimals of the frame rate an MP4 written by OpenCV carries back exactly.

Rates such as 0.5, 0.25, 0.123 and 0.873 read back identical through
``cv2.VideoCapture`` and through PyAV. The effective rate is floored onto this
grid *before* the timestamps are picked, so the rate in the container, the
rate the frames were taken at and the rate printed in the question are one
number and the model's clock is the wall clock."""

MIN_FPS = 10.0**-FPS_DECIMALS
"""Slowest rate this script will sample at; below it the container rounds."""

FOURCC = "mp4v"
"""Codec of the resampled clip, as ``adapters.chat_openai`` encodes video."""

NO_THINKING_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}
VIDEO_BODY: dict = {"mm_processor_kwargs": {"do_sample_frames": False}}
"""The two vLLM request-body switches of ``scripts/_wiring.py``, repeated here
so this script does not depend on the agent's wiring. Thinking is off because
the agent runs with it off; video frames are taken as sent because the
sampling happens here, and the server is started with
``--media-io-kwargs '{"video": {"num_frames": -1}}'`` so that it keeps every
frame the MP4 carries. Neither is sent with ``--provider openai-compatible``."""

PROVIDER_VLLM = "vllm"
PROVIDER_OPENAI_COMPATIBLE = "openai-compatible"
PROVIDERS = (PROVIDER_VLLM, PROVIDER_OPENAI_COMPATIBLE)
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

EXIT_OK = 0
EXIT_FAILED_RECORDINGS = 3
"""Exit code of a batch that finished with failed recordings.

``3`` and not ``1``: it lets a calling script distinguish a batch that
finished with failed recordings (whose ``error`` records are written and
counted by the evaluation) from a crash (``1``, an uncaught exception) or a
usage error (``2``, from ``argparse``).
"""

RETRY_WAIT_S = 5.0
RETRYABLE_ERRORS = frozenset(
    {"APIConnectionError", "APITimeoutError", "InternalServerError", "ConnectionError"}
)
"""Exception class names worth one more attempt; see ``is_retryable``."""

LENGTH_FINISH_REASON = "length"

STEPS_MARKER = "These are the steps of the recipe:"
"""Where a rendered question stops being its preamble.

The global question is the arm's own opening paragraph plus the recipe, so
that the model is told what it is looking at in exactly the words the other
call uses. Splitting on this line is what makes "the first paragraph" mean
the caption block too, in the arm that has one."""

PARTS_NOTE = (
    "The video is sent in {n} consecutive parts, each announced with the time "
    "span of the whole video it covers; every time you write in the answer "
    "must be in seconds from the start of the whole video, not from the start "
    "of a part."
)
"""Appended to the preamble under ``--video-parts-s``, so that the templates
read the same whether or not the video is split."""

PART_LABEL = "Part {k}: from {a} s to {b} s of the video."

VIDEO_TEMPLATE_FILE = "video_question.txt"
CAPTIONS_TEMPLATE_FILE = "captions_question.txt"
GLOBAL_TEMPLATE_FILE = "global_question.txt"

VIDEO_PLACEHOLDERS = (
    "{recipe_name}",
    "{fps}",
    "{duration_s}",
    "{steps}",
    "{prerequisites}",
    "{schema}",
)
CAPTIONS_PLACEHOLDERS = (
    "{recipe_name}",
    "{n_segments}",
    "{captions}",
    "{steps}",
    "{prerequisites}",
    "{schema}",
)
GLOBAL_PLACEHOLDERS = ("{preamble}", "{steps}", "{prerequisites}")

NO_CAPTION = "(no caption)"

SPAN_KEY = "span_s"
SEGMENTS_KEY = "segments"


def _video_answer_schema() -> str:
    """The packaged answer schema with its anchors written in seconds.

    Derived from ``ANSWER_SCHEMA`` instead of copied so the two cannot drift
    in the fields that matter (the verdicts), and checked so they cannot
    drift silently in the field that changes: if the package ever rewords its
    anchor, the replacement stops matching and this raises at import rather
    than sending the model a schema that asks for segment ids it cannot see.
    """
    old = '"segments": [<first>, <last>] or null'
    new = '"span_s": [<start>, <end>] or null'
    if ANSWER_SCHEMA.count(old) != 2:
        raise ValueError(
            f"task.prompt.ANSWER_SCHEMA no longer carries {old!r} twice; "
            "the video arm's schema has to be rewritten by hand"
        )
    return ANSWER_SCHEMA.replace(old, new)


VIDEO_ANSWER_SCHEMA = _video_answer_schema()


# --------------------------------------------------------------------------
# Sampling plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplePlan:
    """Which frames of a recording travel to the model, and what they cost.

    ``fps_requested`` is the arm's nominal rate and ``fps_effective`` the one
    that was actually used: they differ when the recording is long enough
    that the nominal rate would blow the visual-token budget, and both are
    recorded because a recording sampled more sparsely than its arm's
    nominal rate has to be identifiable when the results are read.
    """

    duration_s: float
    fps_requested: float
    fps_effective: float
    n_frames: int
    timestamps_s: tuple[float, ...]
    tokens_per_pair: int
    est_visual_tokens: int
    budget: int


def tokens_per_pair(width: int, height: int) -> int:
    """Visual tokens one pair of frames costs at ``width`` x ``height``."""
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size must be positive, got {width}x{height}")
    return math.ceil(width / PIXELS_PER_TOKEN_SIDE) * math.ceil(
        height / PIXELS_PER_TOKEN_SIDE
    )


def quantize_fps(fps: float) -> float:
    """Round ``fps`` *down* onto the grid an MP4 carries back exactly.

    Down and not to nearest: the rate is what decides how many frames are
    sent, and rounding up could push a plan that just fits over its budget.
    """
    scale = 10**FPS_DECIMALS
    # The epsilon absorbs the binary representation of a decimal rate: 0.5 and
    # 4.0 are exact, 0.1 * 1000 is 99.99999999999999.
    return math.floor(fps * scale + 1e-6) / scale


def frames_until(duration_s: float, fps: float) -> int:
    """How many frames at ``t = k / fps`` fall strictly inside ``duration_s``.

    Even, because the processor merges consecutive frames two by two: an odd
    last frame would be padded by the model's processor and cost a whole pair
    anyway, so it is dropped here where the record can say so.
    """
    if duration_s <= 0 or fps <= 0:
        return 0
    count = math.ceil(duration_s * fps - 1e-9)
    return count - (count % FRAMES_PER_TOKEN)


def estimate_visual_tokens(n_frames: int, per_pair: int) -> int:
    return math.ceil(n_frames / FRAMES_PER_TOKEN) * per_pair


def sample_plan(
    duration_s: float,
    fps: float,
    width: int,
    height: int,
    budget: int = DEFAULT_VISUAL_TOKEN_BUDGET,
) -> SamplePlan:
    """Decide the frames of one recording, thinning the rate to fit ``budget``.

    Pure: it never opens the video. The whole plan follows from the duration,
    the arm's rate and the frame size, which makes the budget arithmetic
    testable without a GPU or a dataset.

    A recording over budget is not skipped and not truncated: a truncated
    video would answer the per-step question about the half it was given and
    look like a model that missed the end. It is sampled thinner, in
    proportion, and ``fps_effective`` says by how much.
    """
    if fps <= 0:
        raise ValueError(f"--fps must be positive, got {fps}")
    if budget <= 0:
        raise ValueError(f"--visual-token-budget must be positive, got {budget}")
    per_pair = tokens_per_pair(width, height)

    effective = quantize_fps(fps)
    n_frames = frames_until(duration_s, effective)
    estimate = estimate_visual_tokens(n_frames, per_pair)
    if estimate > budget:
        effective = quantize_fps(effective * budget / estimate)
        # One rounding of the frame count can still leave a single pair over
        # the line; step down the grid until it does not. Bounded: the loop
        # only ever runs while a pair is over budget, and each step removes
        # at least one frame from a finite plan.
        while effective >= MIN_FPS:
            n_frames = frames_until(duration_s, effective)
            estimate = estimate_visual_tokens(n_frames, per_pair)
            if estimate <= budget:
                break
            effective = quantize_fps(effective - MIN_FPS)
        else:
            raise ValueError(
                f"a {duration_s:.0f}s recording at {width}x{height} "
                f"({per_pair} tokens per frame pair) cannot fit a budget of "
                f"{budget} visual tokens"
            )
    if n_frames < FRAMES_PER_TOKEN:
        raise ValueError(
            f"a {duration_s:.0f}s recording at {fps:g} fps yields {n_frames} "
            "frames; a video input needs at least two"
        )
    return SamplePlan(
        duration_s=float(duration_s),
        fps_requested=float(fps),
        fps_effective=effective,
        n_frames=n_frames,
        timestamps_s=tuple(index / effective for index in range(n_frames)),
        tokens_per_pair=per_pair,
        est_visual_tokens=estimate,
        budget=int(budget),
    )


@dataclass(frozen=True)
class VideoPart:
    """One ``video_url`` item: a contiguous slice of the sampling plan."""

    index: int  # 1-based, as the label prints it
    start_s: float
    end_s: float
    timestamps_s: tuple[float, ...]

    def label(self) -> str:
        """The text item announcing this part, in absolute video time."""
        return (
            PART_LABEL.replace("{k}", str(self.index))
            .replace("{a}", f"{self.start_s:g}")
            .replace("{b}", f"{self.end_s:g}")
        )


def split_into_parts(plan: SamplePlan, parts_s: float | None) -> tuple[VideoPart, ...]:
    """Cut the plan into the video items the request will carry.

    ``None`` (the default) is one item with the whole recording in it. A
    number is the optional alternative: the same frames at the same rate,
    split into slices of ``parts_s`` seconds, one video item each, for a
    server that will not take the whole recording as a single input. Each
    slice keeps an even frame count for the same reason the whole plan does.
    """
    if parts_s is None:
        return (
            VideoPart(
                index=1,
                start_s=0.0,
                end_s=plan.duration_s,
                timestamps_s=plan.timestamps_s,
            ),
        )
    if parts_s <= 0:
        raise ValueError(f"--video-parts-s must be positive, got {parts_s}")
    grouped: dict[int, list[float]] = {}
    for timestamp in plan.timestamps_s:
        grouped.setdefault(int(timestamp // parts_s), []).append(timestamp)
    parts: list[VideoPart] = []
    for slot in sorted(grouped):
        stamps = grouped[slot]
        keep = len(stamps) - (len(stamps) % FRAMES_PER_TOKEN)
        if keep < FRAMES_PER_TOKEN:
            continue
        parts.append(
            VideoPart(
                index=len(parts) + 1,
                start_s=slot * parts_s,
                end_s=min((slot + 1) * parts_s, plan.duration_s),
                timestamps_s=tuple(stamps[:keep]),
            )
        )
    if not parts:
        raise ValueError(
            f"splitting {plan.n_frames} frames into parts of {parts_s:g} s "
            "leaves no part with two frames"
        )
    return tuple(parts)


# --------------------------------------------------------------------------
# Decoding and re-encoding
# --------------------------------------------------------------------------


@dataclass
class EncodedPart:
    """One part of a recording, resampled and written to a temporary MP4."""

    part: VideoPart
    path: Path
    n_frames: int
    size_bytes: int


@dataclass
class EncodedVideo:
    """Everything the request sends as video, and what it is expected to cost."""

    parts: tuple[EncodedPart, ...]
    fps: float
    n_frames: int
    est_visual_tokens: int
    size_bytes: int


def source_frame_index(time_s: float, video: Video) -> int:
    """Which frame of the source video stands for ``time_s``."""
    return min(max(int(round(time_s * video.fps)), 0), max(video.frame_count - 1, 0))


def encode_parts(
    video: Video,
    parts: Sequence[VideoPart],
    *,
    width: int,
    height: int,
    fps: float,
    directory: Path,
) -> EncodedVideo:
    """Resample the recording into one MP4 per part, in one pass over the file.

    The pass is sequential (``grab()`` to skip, ``retrieve()`` only on a
    frame that is kept) because seeking per frame can cost more than decoding
    everything (``media.iter_frames_at`` makes the same
    trade). Frames are never converted to RGB: they come out
    of the decoder in the writer's own colour order, and this script has no
    use for them in between.

    Frames are written in pairs so that a part whose tail could not be
    decoded still holds an even number of them, and the frame counts in the
    record are the frames that were really sent.
    """
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video.path}")
    position = 0  # index of the frame the next grab() consumes
    retrieved_index = -1
    frame = None

    def frame_at(index: int):
        """The resampled frame at ``index``, or ``None`` past the end."""
        nonlocal position, retrieved_index, frame
        while position <= index:
            if not capture.grab():
                return None
            position += 1
        if retrieved_index != index:
            success, raw = capture.retrieve()
            if not success or raw is None:
                return None
            # A fresh array per call, so the pair held back below is never
            # overwritten by the next frame.
            frame = cv2.resize(raw, (width, height), interpolation=cv2.INTER_AREA)
            retrieved_index = index
        return frame

    encoded: list[EncodedPart] = []
    try:
        for part in parts:
            path = Path(directory) / f"part{part.index:03d}.mp4"
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise ValueError(f"cannot open an MP4 writer for {width}x{height}")
            written = 0
            pending = None
            try:
                for timestamp in part.timestamps_s:
                    current = frame_at(source_frame_index(timestamp, video))
                    if current is None:
                        break
                    if pending is None:
                        pending = current
                        continue
                    writer.write(pending)
                    writer.write(current)
                    written += FRAMES_PER_TOKEN
                    pending = None
            finally:
                writer.release()
            encoded.append(
                EncodedPart(
                    part=part,
                    path=path,
                    n_frames=written,
                    size_bytes=path.stat().st_size if path.exists() else 0,
                )
            )
    finally:
        capture.release()

    total_frames = sum(entry.n_frames for entry in encoded)
    if total_frames < FRAMES_PER_TOKEN:
        raise ValueError(f"no frames could be decoded from {video.path}")
    per_pair = tokens_per_pair(width, height)
    return EncodedVideo(
        parts=tuple(encoded),
        fps=fps,
        n_frames=total_frames,
        est_visual_tokens=sum(
            estimate_visual_tokens(entry.n_frames, per_pair) for entry in encoded
        ),
        size_bytes=sum(entry.size_bytes for entry in encoded),
    )


def video_data_url(path: Path) -> str:
    """Inline an MP4 as a ``data:`` URL, as ``adapters.chat_openai`` does."""
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{encoded}"


def build_video_content(encoded: EncodedVideo, question: str) -> list[dict]:
    """The user message of a video call: **video first, question last**.

    The package's ``to_openai_message`` puts the text first, which is right
    for a caption request and wrong here: the two questions of a recording
    share the video, and only a common *prefix* is a cache hit on the server.
    Several parts are announced by a text item each, so the absolute time of
    every part is in the context right before its frames.
    """
    content: list[dict] = []
    labelled = len(encoded.parts) > 1
    for entry in encoded.parts:
        if labelled:
            content.append({"type": "text", "text": entry.part.label()})
        content.append(
            {
                "type": "video_url",
                "video_url": {"url": video_data_url(entry.path)},
            }
        )
    content.append({"type": "text", "text": question})
    return content


def build_text_content(question: str) -> list[dict]:
    """The user message of a captions call: the question and nothing else."""
    return [{"type": "text", "text": question}]


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


def read_template(path: Path, placeholders: Sequence[str]) -> str:
    """Read a question template, refusing one that could not be rendered.

    Rendering is ``str.replace`` (the templates quote a JSON schema, so every
    brace would be a format field), and ``replace`` fails silently: a typo in
    a placeholder would send the model a question with no steps in it and the
    run would read as a bad model rather than as a bad file. Same check
    ``scripts/run_task.py`` makes, for the same reason.
    """
    text = Path(path).read_text(encoding="utf-8")
    missing = [name for name in placeholders if name not in text]
    if missing:
        raise ValueError(f"{path} is missing the placeholder(s) {', '.join(missing)}")
    return text


def template_sha256(text: str) -> str:
    """Digest identifying a wording in a record (``agent.prompts.prompt_sha256``)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_template_path(arm: str) -> Path:
    """The per-step template an arm uses when none is given on the command line."""
    name = VIDEO_TEMPLATE_FILE if arm == ARM_VIDEO else CAPTIONS_TEMPLATE_FILE
    return PROMPTS_DIR / name


def template_label(path: Path) -> str:
    """How a template file is named in a configuration.

    Relative to the repository when it lives inside it, so that the same file
    names the same configuration in clones at different absolute paths;
    absolute otherwise. The digest beside it
    is what identifies the wording, the label is what lets a reader find it.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def is_default_template(arm: str, path: Path) -> bool:
    return Path(path).resolve() == default_template_path(arm).resolve()


def format_fps(fps: float) -> str:
    """The rate as the question spells it: ``1``, ``0.5``, ``0.873``."""
    return f"{fps:g}"


def render_captions(segments: Sequence[Segment], captions: Sequence[str]) -> str:
    """One line per segment: ``[id, start-end s] caption``.

    Newlines inside a caption are flattened, because one caption per line is
    what tells the model where a segment ends; a segment whose caption is
    missing or blank says so rather than closing the line early, so that the
    ids stay in step with the grid.
    """
    lines: list[str] = []
    for segment in segments:
        caption = captions[segment.index] if segment.index < len(captions) else ""
        text = " ".join(str(caption).split()) or NO_CAPTION
        lines.append(
            f"[{segment.index}, {segment.start_s:g}-{segment.end_s:g} s] {text}"
        )
    return "\n".join(lines)


def build_video_question(
    sop: Sop,
    graph: TaskGraph,
    template: str,
    *,
    fps: float,
    duration_s: float,
) -> str:
    """Render the per-step question of the video arm."""
    return (
        template.replace("{recipe_name}", sop.recipe_name)
        .replace("{fps}", format_fps(fps))
        .replace("{duration_s}", str(int(round(duration_s))))
        .replace("{steps}", _render_steps(sop))
        .replace("{prerequisites}", _render_prerequisites(sop, graph))
        .replace("{schema}", VIDEO_ANSWER_SCHEMA)
    )


def build_captions_question(
    sop: Sop,
    graph: TaskGraph,
    template: str,
    *,
    segments: Sequence[Segment],
    captions: Sequence[str],
) -> str:
    """Render the per-step question of the captions arm."""
    return (
        template.replace("{recipe_name}", sop.recipe_name)
        .replace("{n_segments}", str(len(segments)))
        .replace("{captions}", render_captions(segments, captions))
        .replace("{steps}", _render_steps(sop))
        .replace("{prerequisites}", _render_prerequisites(sop, graph))
        .replace("{schema}", ANSWER_SCHEMA)
    )


def preamble_of(question: str) -> str:
    """The opening of a rendered question: everything before the step list.

    In the video arm that is one paragraph; in the captions arm it is the
    paragraph *and* the block of captions, which is intended: the
    global question has to be asked of the same evidence as the per-step one.
    """
    head, marker, _ = question.partition(STEPS_MARKER)
    if not marker:
        raise ValueError(f"a rendered question with no {STEPS_MARKER!r} line")
    return head.rstrip()


def with_parts_note(question: str, n_parts: int) -> str:
    """Add the absolute-time note of ``--video-parts-s`` to a question's preamble.

    Written here and not in the template because the split is a transport
    decision taken per run: the templates read the same whether or not the
    video is split.
    """
    head, marker, tail = question.partition(STEPS_MARKER)
    if not marker:
        raise ValueError(f"a rendered question with no {STEPS_MARKER!r} line")
    note = PARTS_NOTE.replace("{n}", str(n_parts))
    return f"{head.rstrip()} {note}\n\n{marker}{tail}"


def build_global_question(
    preamble: str, sop: Sop, graph: TaskGraph, template: str
) -> str:
    """Render the yes/no question from the arm's own preamble."""
    return (
        template.replace("{preamble}", preamble)
        .replace("{steps}", _render_steps(sop))
        .replace("{prerequisites}", _render_prerequisites(sop, graph))
    )


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    """Whether ``value`` is a JSON number (``True``/``False`` are not)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def span_to_segment_ids(
    value: object, n_segments: int, segment_s: float = SECONDS_PER_SEGMENT
) -> list[int] | None:
    """Map ``[start, end]`` seconds onto the grid ids ``[first, last]``.

    Ids are clamped to the grid, as ``verdicts._parse_anchor`` clamps the ones
    a model writes itself. A reversed span is *not* swapped: with the pixels
    in front of it the model is reading a clock, and ``[400, 40]`` is a
    number it did not mean rather than a range it miscounted; the entry
    keeps its anchor as ``null`` and its raw text in the record.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, end = value
    if not _is_number(start) or not _is_number(end) or start > end:
        return None
    highest = int(n_segments) - 1
    if highest < 0 or segment_s <= 0:
        return None
    return [
        min(max(math.floor(start / segment_s), 0), highest),
        min(max(math.floor(end / segment_s), 0), highest),
    ]


def _convert_anchor(
    entry: Mapping, n_segments: int, segment_s: float
) -> dict:
    """Copy one answer entry with its time span turned into segment ids."""
    converted = dict(entry)
    if SPAN_KEY in entry:
        span = entry[SPAN_KEY]
    elif SEGMENTS_KEY in entry:
        # The model answered in the packaged key while being asked for
        # seconds. The numbers are still seconds (it is the name that is
        # wrong), so they are read as such and kept under ``span_s``, which
        # is what they are, before the converted ids take the other key.
        span = entry[SEGMENTS_KEY]
        converted[SPAN_KEY] = span
    else:
        return converted
    converted[SEGMENTS_KEY] = span_to_segment_ids(span, n_segments, segment_s)
    return converted


def spans_to_segments(
    payload: Mapping, n_segments: int, segment_s: float = SECONDS_PER_SEGMENT
) -> dict:
    """Rewrite a video-arm answer in the segment ids ``verdicts`` speaks.

    Pure, and it never mutates ``payload``: the raw answer is what an error
    analysis reads, and a converted copy that overwrote it
    would leave no way to tell a model that wrote ``[40, 44]`` seconds from
    one that wrote segments 40 to 44.

    Every original key survives into the copy, ``span_s`` included, so the
    ``raw`` field of each verdict shows both what the model wrote and what it
    was read as.
    """
    converted = dict(payload)
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return converted
    entries: list[Any] = []
    for entry in steps:
        if not isinstance(entry, dict):
            entries.append(entry)
            continue
        new_entry = _convert_anchor(entry, n_segments, segment_s)
        prerequisite = entry.get("violated_prerequisite")
        if isinstance(prerequisite, dict):
            new_entry["violated_prerequisite"] = _convert_anchor(
                prerequisite, n_segments, segment_s
            )
        entries.append(new_entry)
    converted["steps"] = entries
    return converted


def answer_key(payload: Mapping | None) -> str | None:
    """Which anchor key the model actually used, for the record.

    ``"span_s"`` is what the video arm asked for and ``"segments"`` what the
    captions arm asked for; the other value in each arm is the model falling
    back to a habit, which changes nothing about how the answer is read but
    everything about how a table of anchors should be believed.
    """
    found: set[str] = set()
    steps = (payload or {}).get("steps")
    for entry in steps or ():
        if not isinstance(entry, dict):
            continue
        for candidate in (entry, entry.get("violated_prerequisite")):
            if not isinstance(candidate, dict):
                continue
            if SPAN_KEY in candidate:
                found.add(SPAN_KEY)
            elif SEGMENTS_KEY in candidate:
                found.add(SEGMENTS_KEY)
    if not found:
        return None
    if len(found) == 1:
        return found.pop()
    return "mixed"


FOLLOWED_VALUES = ("yes", "no")


@dataclass(frozen=True)
class GlobalAnswer:
    """The yes/no verdict read out of a global reply."""

    followed: str | None
    justification: str | None
    status: str  # "ok" | "parse_failed"


def parse_global_answer(text: str | None) -> GlobalAnswer:
    """Read the last ``{"followed": ...}`` object of a reply.

    The last and not the first, for the reason ``parse_final_answer`` gives:
    a model that writes twice is revising. Anything else (no object, a
    ``followed`` that is neither yes nor no, prose) is ``parse_failed`` with
    a null verdict, because an unreadable answer is scored as an abstention
    and never as a guess.
    """
    followed: str | None = None
    justification: str | None = None
    for payload in _json_objects(_FENCE.sub("", text or "")):
        if not isinstance(payload, dict):
            continue
        value = payload.get("followed")
        if not isinstance(value, str) or value.strip().lower() not in FOLLOWED_VALUES:
            continue
        followed = value.strip().lower()
        reason = payload.get("justification")
        justification = reason if isinstance(reason, str) else None
    if followed is None:
        return GlobalAnswer(followed=None, justification=None, status="parse_failed")
    return GlobalAnswer(
        followed=followed, justification=justification, status="ok"
    )


# --------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Completion:
    """One chat completion and what it cost."""

    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    model: str
    finish_reason: str | None


def is_retryable(error: BaseException) -> bool:
    """Whether one more attempt could plausibly succeed.

    Duck-typed rather than caught by class: this script must not import
    ``openai`` to decide, and the tests inject a client that raises its own
    look-alikes. A 4xx is the request being wrong and repeating it only
    wastes a long prompt; a 5xx or a dropped connection is
    the server, which is worth one more try.
    """
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status >= 500
    return type(error).__name__ in RETRYABLE_ERRORS


class OneShotClient:
    """One user message in, one completion out, against an OpenAI-compatible API.

    Deliberately not ``adapters.chat_openai``: that adapter builds the
    message itself (text before video) and retries an empty reply, neither of
    which this script wants. ``client`` injects an already-built SDK client,
    which is how the tests run without a server.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        extra_body: dict | None = None,
        retry_wait_s: float = RETRY_WAIT_S,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.extra_body = extra_body
        self.retry_wait_s = retry_wait_s
        if client is not None:
            self.client = client
            return
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("VLLM_API_KEY") or "EMPTY",
            # The SDK's own retries are off: a retried 4xx is a retried
            # mistake, and a batch must not spend several attempts per
            # recording discovering the same one.
            max_retries=0,
            timeout=timeout_s,
        )

    def request(self, content: Sequence[dict]) -> dict:
        """The request body, as a dict, so a test can read it without a server."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": list(content)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if self.extra_body:
            body["extra_body"] = self.extra_body
        return body

    def complete(self, content: Sequence[dict]) -> Completion:
        """Send one user message and return the completion."""
        body = self.request(content)
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**body)
        except Exception as error:
            if not is_retryable(error):
                raise
            time.sleep(self.retry_wait_s)
            started = time.perf_counter()
            response = self.client.chat.completions.create(**body)
        latency_s = time.perf_counter() - started
        return self._to_completion(response, latency_s)

    def _to_completion(self, response: Any, latency_s: float) -> Completion:
        choices = getattr(response, "choices", None) or ()
        if not choices:
            raise ValueError(f"no choices in response from {self.model}")
        choice = choices[0]
        text = getattr(choice.message, "content", None) or ""
        counts = getattr(response, "usage", None)
        return Completion(
            text=text,
            tokens_in=int(getattr(counts, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(counts, "completion_tokens", 0) or 0),
            latency_s=latency_s,
            model=str(getattr(response, "model", None) or self.model),
            finish_reason=getattr(choice, "finish_reason", None),
        )


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


def _now() -> str:
    """UTC timestamp at second resolution, as the records store it."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def summarize_exception(error: BaseException, *, lines: int = 6) -> str:
    """The name, the message and the last frames of a traceback.

    A recording that failed is a result the record has to keep (the
    evaluation counts it against the system), and "ValueError" alone never
    says which of
    the decoder, the encoder or the client raised it.
    """
    formatted = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    tail = formatted.strip().splitlines()[-lines:]
    return "\n".join(tail)


def build_config(
    *,
    arm: str,
    question: str,
    question_template_sha256: str,
    model: str,
    temperature: float,
    seed: int | None,
    max_tokens: int,
    fps: float | None,
    width: int | None,
    height: int | None,
    visual_token_budget: int | None,
    video_parts_s: float | None,
    caption_variant: str | None,
    preamble_template_sha256: str | None = None,
    question_template_file: str | None = None,
    preamble_template_file: str | None = None,
    provider: str = PROVIDER_VLLM,
) -> dict:
    """Freeze the run's configuration and name it with a digest.

    One dictionary for the whole run: nothing that varies from recording to
    recording may enter it (that is what the record's ``input`` block is
    for), because ``eval.report`` refuses a run whose records claim two
    ``config_id``s, since a table row averaged over two
    configurations is not a comparison of anything.

    ``preamble_template_sha256`` is written for the global question and for
    it alone, because that question is made of *two* templates: the global
    one, and the arm's, whose opening paragraph it quotes (``preamble_of``).
    Without it, rewording the video arm's first paragraph would change what
    the model is asked and leave the global configuration with the same id,
    which is exactly the mix-up ``config_id`` exists to prevent. The per-step
    configuration has one template and therefore no such key.

    ``question_template_file`` and ``preamble_template_file`` name the arm's
    template when a non-default one was chosen with ``--video-template`` or
    ``--captions-template``. They are written only then, so that a run on the
    default templates has a configuration without them; a different wording
    already moves the id through its digest, and the name is there for the
    reader.

    ``provider`` is recorded as ``chat_backend``. With ``vllm`` thinking is
    disabled in the request (``"thinking": False``); with
    ``openai-compatible`` neither thinking nor the seed is controlled by the
    request, so both are recorded as ``None``.
    """
    is_vllm = provider == PROVIDER_VLLM
    config = {
        "system": SYSTEM,
        "arm": arm,
        "question": question,
        "orchestrator_model": model,
        "chat_backend": provider,
        "thinking": False if is_vllm else None,
        "temperature": float(temperature),
        "seed": None if seed is None or not is_vllm else int(seed),
        "max_tokens": int(max_tokens),
        "fps": None if fps is None else float(fps),
        "width": None if width is None else int(width),
        "height": None if height is None else int(height),
        "visual_token_budget": (
            None if visual_token_budget is None else int(visual_token_budget)
        ),
        "video_parts_s": None if video_parts_s is None else float(video_parts_s),
        "caption_variant": caption_variant,
        "question_template_sha256": question_template_sha256,
        "code_version": code_version(),
    }
    if preamble_template_sha256 is not None:
        config["preamble_template_sha256"] = preamble_template_sha256
    if question_template_file is not None:
        config["question_template_file"] = question_template_file
    if preamble_template_file is not None:
        config["preamble_template_file"] = preamble_template_file
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CONFIG_ID_LENGTH]
    return {**config, "config_id": digest}


def build_turn(completion: Completion, *, wall_s: float) -> LLMTurn:
    """The single turn of a one-shot episode."""
    return LLMTurn(
        index=0,
        completion=completion.text,
        usage=Usage(
            llm_calls=1,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            wall_s=wall_s,
        ),
        latency_s=completion.latency_s,
        model=completion.model,
        finish_reason=completion.finish_reason,
        reasoning=None,
    )


def build_transcript(question: str, turn: LLMTurn) -> Transcript:
    """One episode with one step in it, in the package's own shape.

    There is no loop and no tool, so the step carries the completion and
    nothing else; ``Transcript.from_dict`` reads it back like any other, which
    is what lets the readers of the other runs open this one.
    """
    return Transcript(
        prompt=question,
        steps=[
            ReActStep(
                turn=turn,
                action=None,
                final_answer=turn.completion,
                parse_error=None,
            )
        ],
        final_answer=turn.completion,
        status="final_answer",
        usage=turn.usage,
    )


def answer_status(raw: dict | None, finish_reason: str | None) -> str:
    """The record's status for an answer that came back.

    ``length`` is kept apart from ``parse_failed`` because they are different
    failures with different fixes: one is an answer that needed more tokens
    than the cap allows, the other is a model that wrote something else. A
    truncated answer that still carries a readable object is an ``ok`` run:
    the cap cut the prose, not the JSON.
    """
    if raw is not None:
        return "ok"
    if finish_reason == LENGTH_FINISH_REASON:
        return "length"
    return "parse_failed"


def make_run_record(
    *,
    recording_id: str,
    sop: Sop,
    segments: Sequence[Segment],
    video: Video,
    question: str,
    config: Mapping,
    started_at: str,
    wall_s: float,
    inputs: Mapping,
    completion: Completion | None,
    raw: dict | None,
    error: str | None,
) -> dict:
    """One recording's per-step record, keyed exactly like ``run.make_record``.

    Built by hand and not by ``make_record``, which takes a ``TaskConfig`` and
    an ``AgentResult`` this script has neither of; the keys are the same ones
    so that ``eval/report.py`` and ``scripts/evaluate.py`` read this run and
    the agent's runs with the same code. The only addition is ``input``, which
    holds everything that varies from recording to recording and therefore
    cannot live in ``config``.
    """
    final_answer_text = None if completion is None else completion.text
    if completion is None:
        status = "error"
        transcript = None
        usage = Usage()
    else:
        status = answer_status(raw, completion.finish_reason)
        turn = build_turn(completion, wall_s=wall_s)
        transcript = build_transcript(question, turn)
        usage = turn.usage
    verdicts = normalize(raw, sop, segments)
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "config": dict(config),
        "recording_id": recording_id,
        "activity_id": sop.activity_id,
        "started_at": started_at,
        "finished_at": _now(),
        "wall_s": wall_s,
        "status": status,
        "error": error,
        "n_segments": len(segments),
        "video": {
            "fps": float(video.fps),
            "frame_count": int(video.frame_count),
            "n_segments": len(segments),
        },
        "question": question,
        "final_answer_text": final_answer_text,
        "n_answer_candidates": len(final_answer_candidates(final_answer_text or "")),
        "verdicts": [verdict.to_dict() for verdict in verdicts],
        "transcript": None if transcript is None else transcript.to_dict(),
        "usage": usage.to_dict(),
        "usage_total": usage.to_dict(),
        "input": dict(inputs),
    }


def make_global_record(
    *,
    recording_id: str,
    question: str,
    config: Mapping,
    started_at: str,
    wall_s: float,
    inputs: Mapping,
    completion: Completion | None,
    answer: GlobalAnswer | None,
    error: str | None,
) -> dict:
    """One recording's global record, written beside its ``run.json``.

    Its own file and not a key of the record because it is its own call with
    its own configuration digest: nothing about the per-step verdicts may
    depend on it, and a reader comparing global answers across arms should
    not have to load every transcript to find them.
    """
    if completion is None:
        status = "error"
        turn = None
        usage = Usage()
    else:
        status = "ok" if answer is not None and answer.status == "ok" else "parse_failed"
        turn = build_turn(completion, wall_s=wall_s)
        usage = turn.usage
    return {
        "schema_version": GLOBAL_SCHEMA_VERSION,
        "config": dict(config),
        "recording_id": recording_id,
        "started_at": started_at,
        "finished_at": _now(),
        "wall_s": wall_s,
        "status": status,
        "error": error,
        "question": question,
        "answer_text": None if completion is None else completion.text,
        "followed": None if answer is None else answer.followed,
        "justification": None if answer is None else answer.justification,
        "input": dict(inputs),
        "usage": usage.to_dict(),
        "turn": None if turn is None else turn.to_dict(),
    }


# --------------------------------------------------------------------------
# Storage and selection
# --------------------------------------------------------------------------


def global_path(store: RunStore, recording_id: str) -> Path:
    return store.directory / recording_id / GLOBAL_FILE


def has_answer(store: RunStore, recording_id: str, question: str) -> bool:
    """Whether this recording already answered ``question``.

    Both paths are read directly, never through ``VideoArtifacts``, whose
    constructor would create the directory: asking a question must not leave
    behind an empty directory that the next pass then has to ignore
    (``RunStore.has`` reads its path for the same reason).
    """
    if question == QUESTION_STEPS:
        return store.has(recording_id)
    return global_path(store, recording_id).is_file()


def load_answer(store: RunStore, recording_id: str, question: str) -> dict | None:
    """The record of one answered question, or ``None``."""
    if not has_answer(store, recording_id, question):
        return None
    path = (
        store.directory / recording_id / RUN_FILE
        if question == QUESTION_STEPS
        else global_path(store, recording_id)
    )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        # A half-written file is not an answer; redo it.
        return None


def save_answer(store: RunStore, recording_id: str, question: str, record: dict) -> None:
    if question == QUESTION_STEPS:
        store.save(recording_id, record)
        return
    VideoArtifacts(store.directory, recording_id).save_json(GLOBAL_FILE, record)


def pending_questions(
    store: RunStore,
    recording_id: str,
    questions: Sequence[str],
    *,
    force: bool = False,
    retry_errors: bool = False,
) -> list[str]:
    """Which of ``questions`` this recording still owes, in call order.

    Per question and not per recording: the two calls are independent, and a
    recording whose global answer failed should not pay for its visual
    tokens again to redo the per-step one that worked.
    """
    pending: list[str] = []
    for question in QUESTION_ORDER:
        if question not in questions:
            continue
        if force or not has_answer(store, recording_id, question):
            pending.append(question)
            continue
        if retry_errors:
            record = load_answer(store, recording_id, question)
            if record is None or record.get("status") == "error":
                pending.append(question)
    return pending


def select_recordings(
    ids_error: Sequence[str],
    ids_normal: Sequence[str],
    *,
    store: RunStore,
    questions: Sequence[str],
    shard: str | None = None,
    limit: int | None = None,
    force: bool = False,
    retry_errors: bool = False,
) -> list[str]:
    """Decide which recordings this process runs, in which order.

    Same semantics as ``scripts/run_task.py::select_recordings``: sorted
    first, so sharding is a property of the split and not of the file's line
    order; resumption after sharding and before ``--limit``, so a second pass
    with the same flags picks up exactly what is missing.
    """
    recording_ids = sorted(set(list(ids_error) + list(ids_normal)))
    if shard:
        index, total = (int(part) for part in shard.split("/"))
        if total <= 0 or not 0 <= index < total:
            raise ValueError(f"--shard {shard} is not a valid I/N with 0 <= I < N")
        recording_ids = recording_ids[index::total]
    recording_ids = [
        recording_id
        for recording_id in recording_ids
        if pending_questions(
            store,
            recording_id,
            questions,
            force=force,
            retry_errors=retry_errors,
        )
    ]
    if limit:
        recording_ids = recording_ids[:limit]
    return recording_ids


# --------------------------------------------------------------------------
# One recording
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingInputs:
    """What one recording is, before anything expensive is done to it."""

    video: Video
    segments: list[Segment]
    captions: list[str] | None


def load_recording(
    recording_id: str,
    *,
    artifacts_root: Path,
    videos_dir: Path,
    arm: str,
    caption_variant: str,
) -> RecordingInputs:
    """Read the video's metadata, the 2-second grid and the arm's captions.

    Cheap, and kept apart from ``prepare_recording`` for that reason: a
    failure of the decoder or of the encoder should still leave a record with
    a real ``video`` block in it, which the evaluation needs to rebuild the
    grid a recording was anchored against.

    The captions arm takes the grid its caption file was written for, which
    is the grid the agent's runs answered on. The video arm has no memory by
    construction, so it rebuilds the same grid from the video's own metadata
    with the function that built it in the first place, and therefore runs on
    a recording whose captions were never preprocessed.
    """
    stem = f"{recording_id}{VIDEO_SUFFIX}"
    artifacts = VideoArtifacts(artifacts_root, stem)
    video = load_video_meta(artifacts, Path(videos_dir) / f"{stem}.mp4")
    if arm == ARM_CAPTIONS:
        stored = artifacts.load_json(captions_file(caption_variant))
        captions = [str(text) for text in stored]
        return RecordingInputs(
            video=video, segments=segment_grid(video, len(captions)), captions=captions
        )
    return RecordingInputs(
        video=video, segments=get_video_segments(video), captions=None
    )


@dataclass
class Prepared:
    """Everything the two calls of one recording share."""

    inputs: RecordingInputs
    question_steps: str
    question_global: str
    plan: SamplePlan | None
    encoded: EncodedVideo | None
    fields: dict

    @property
    def video(self) -> Video:
        return self.inputs.video

    @property
    def segments(self) -> list[Segment]:
        return self.inputs.segments


def prepare_recording(
    loaded: RecordingInputs,
    *,
    sop: Sop,
    graph: TaskGraph,
    arm: str,
    fps: float | None,
    width: int | None,
    height: int | None,
    budget: int,
    video_parts_s: float | None,
    templates: Mapping[str, str],
    workdir: Path,
) -> Prepared:
    """Resample the video and render both questions, once per recording.

    Once and not once per question: the video prefix is the same for both
    calls, and re-encoding a long video to ask a second question would
    double the CPU cost of the batch.
    """
    started = time.perf_counter()
    video = loaded.video
    plan: SamplePlan | None = None
    encoded: EncodedVideo | None = None
    if arm == ARM_VIDEO:
        plan = sample_plan(
            video.duration_s, float(fps), int(width), int(height), budget
        )
        parts = split_into_parts(plan, video_parts_s)
        question_steps = build_video_question(
            sop,
            graph,
            templates[ARM_VIDEO],
            fps=plan.fps_effective,
            duration_s=video.duration_s,
        )
        if len(parts) > 1:
            question_steps = with_parts_note(question_steps, len(parts))
        encoded = encode_parts(
            video,
            parts,
            width=int(width),
            height=int(height),
            fps=plan.fps_effective,
            directory=workdir,
        )
    else:
        question_steps = build_captions_question(
            sop,
            graph,
            templates[ARM_CAPTIONS],
            segments=loaded.segments,
            captions=loaded.captions or [],
        )
    question_global = build_global_question(
        preamble_of(question_steps), sop, graph, templates[QUESTION_GLOBAL]
    )
    fields = {
        "fps_effective": None if encoded is None else encoded.fps,
        "n_frames": None if encoded is None else encoded.n_frames,
        "est_visual_tokens": None if encoded is None else encoded.est_visual_tokens,
        "mp4_bytes": None if encoded is None else encoded.size_bytes,
        "duration_s": video.duration_s,
        "n_captions": None if loaded.captions is None else len(loaded.captions),
        "prompt_chars": None,  # filled per question
        "answer_key": None,  # filled from the answer
        # Decoding and re-encoding a long recording takes minutes of CPU that
        # belong to the cost of the system and appear in no other field,
        # since ``wall_s`` measures one call.
        "prepare_s": time.perf_counter() - started,
    }
    return Prepared(
        inputs=loaded,
        question_steps=question_steps,
        question_global=question_global,
        plan=plan,
        encoded=encoded,
        fields=fields,
    )


def build_content(prepared: Prepared, question: str) -> list[dict]:
    """The user message for one question of one recording."""
    if prepared.encoded is None:
        return build_text_content(question)
    return build_video_content(prepared.encoded, question)


def ask_question(
    *,
    question: str,
    prepared: Prepared,
    recording_id: str,
    sop: Sop,
    client: OneShotClient,
    config: Mapping,
) -> dict:
    """Make one call and turn it into the record it belongs in."""
    text = (
        prepared.question_global
        if question == QUESTION_GLOBAL
        else prepared.question_steps
    )
    inputs = dict(prepared.fields)
    inputs["prompt_chars"] = len(text)
    started_at = _now()
    started = time.perf_counter()
    try:
        completion = client.complete(build_content(prepared, text))
    except Exception as error:  # the batch outlives one broken recording
        wall_s = time.perf_counter() - started
        detail = summarize_exception(error)
        if question == QUESTION_GLOBAL:
            return make_global_record(
                recording_id=recording_id,
                question=text,
                config=config,
                started_at=started_at,
                wall_s=wall_s,
                inputs=inputs,
                completion=None,
                answer=None,
                error=detail,
            )
        return make_run_record(
            recording_id=recording_id,
            sop=sop,
            segments=prepared.segments,
            video=prepared.video,
            question=text,
            config=config,
            started_at=started_at,
            wall_s=wall_s,
            inputs=inputs,
            completion=None,
            raw=None,
            error=detail,
        )
    wall_s = time.perf_counter() - started

    if question == QUESTION_GLOBAL:
        answer = parse_global_answer(completion.text)
        return make_global_record(
            recording_id=recording_id,
            question=text,
            config=config,
            started_at=started_at,
            wall_s=wall_s,
            inputs=inputs,
            completion=completion,
            answer=answer,
            error=None,
        )
    raw = parse_final_answer(completion.text)
    inputs["answer_key"] = answer_key(raw)
    if raw is not None and prepared.encoded is not None:
        raw = spans_to_segments(raw, len(prepared.segments))
    return make_run_record(
        recording_id=recording_id,
        sop=sop,
        segments=prepared.segments,
        video=prepared.video,
        question=text,
        config=config,
        started_at=started_at,
        wall_s=wall_s,
        inputs=inputs,
        completion=completion,
        raw=raw,
        error=None,
    )


def error_record(
    *,
    question: str,
    recording_id: str,
    sop: Sop,
    video: Video | None,
    segments: Sequence[Segment],
    config: Mapping,
    started_at: str,
    wall_s: float,
    detail: str,
) -> dict:
    """The record a recording that never reached the server leaves behind.

    Written rather than raised: a failed recording counts against the system
    in the evaluation, and a silent gap would make a run with missing
    recordings look better than one that recorded its errors.
    """
    inputs = {
        "fps_effective": None,
        "n_frames": None,
        "est_visual_tokens": None,
        "mp4_bytes": None,
        "duration_s": None if video is None else video.duration_s,
        "n_captions": None,
        "prompt_chars": None,
        "answer_key": None,
        "prepare_s": None,
    }
    if question == QUESTION_GLOBAL:
        return make_global_record(
            recording_id=recording_id,
            question="",
            config=config,
            started_at=started_at,
            wall_s=wall_s,
            inputs=inputs,
            completion=None,
            answer=None,
            error=detail,
        )
    # A recording whose metadata could not even be read still needs a video
    # block, because the record's shape is a contract; an empty grid is the
    # honest value for one nobody managed to open.
    placeholder = (
        video
        if video is not None
        else Video(path=Path(recording_id), fps=1.0, frame_count=0)
    )
    return make_run_record(
        recording_id=recording_id,
        sop=sop,
        segments=segments,
        video=placeholder,
        question="",
        config=config,
        started_at=started_at,
        wall_s=wall_s,
        inputs=inputs,
        completion=None,
        raw=None,
        error=detail,
    )


# --------------------------------------------------------------------------
# The batch
# --------------------------------------------------------------------------


def describe_plan(recording_id: str, prepared: Prepared) -> str:
    """The line ``--dry-run`` prints for one recording."""
    fields = prepared.fields
    fps = fields["fps_effective"]
    parts = 0 if prepared.encoded is None else len(prepared.encoded.parts)
    return (
        f"{recording_id}: n_frames={fields['n_frames']} "
        f"fps={'-' if fps is None else format_fps(fps)} "
        f"est_visual_tokens={fields['est_visual_tokens']} "
        f"mp4_bytes={fields['mp4_bytes']} parts={parts} "
        f"n_captions={fields['n_captions']} "
        f"steps_chars={len(prepared.question_steps)} "
        f"global_chars={len(prepared.question_global)} "
        f"prepare_s={fields['prepare_s']:.1f}"
    )


def run_batch(args: argparse.Namespace, client: OneShotClient | None) -> int:
    """Run every selected recording; return the number of failed recordings."""
    template_paths = {
        ARM_VIDEO: args.video_template,
        ARM_CAPTIONS: args.captions_template,
    }
    templates = {
        ARM_VIDEO: read_template(args.video_template, VIDEO_PLACEHOLDERS),
        ARM_CAPTIONS: read_template(args.captions_template, CAPTIONS_PLACEHOLDERS),
        QUESTION_GLOBAL: read_template(
            PROMPTS_DIR / GLOBAL_TEMPLATE_FILE, GLOBAL_PLACEHOLDERS
        ),
    }
    arm_template = template_paths[args.arm]
    arm_template_file = (
        None
        if is_default_template(args.arm, arm_template)
        else template_label(arm_template)
    )
    annotations = AnnotationSet.load(args.annotations_dir)
    ids_error, ids_normal = load_split(args.split)
    store = RunStore(args.out, args.run_name)
    recording_ids = select_recordings(
        ids_error,
        ids_normal,
        store=store,
        questions=args.questions,
        shard=args.shard,
        limit=args.limit,
        force=args.force,
        retry_errors=args.retry_errors,
    )

    common = dict(
        arm=args.arm,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens,
        fps=args.fps if args.arm == ARM_VIDEO else None,
        width=args.width if args.arm == ARM_VIDEO else None,
        height=args.height if args.arm == ARM_VIDEO else None,
        visual_token_budget=(
            args.visual_token_budget if args.arm == ARM_VIDEO else None
        ),
        video_parts_s=args.video_parts_s if args.arm == ARM_VIDEO else None,
        caption_variant=args.caption_variant if args.arm == ARM_CAPTIONS else None,
        provider=args.provider,
    )
    arm_sha256 = template_sha256(templates[args.arm])
    configs = {
        QUESTION_STEPS: build_config(
            question=QUESTION_STEPS,
            question_template_sha256=arm_sha256,
            question_template_file=arm_template_file,
            **common,
        ),
        QUESTION_GLOBAL: build_config(
            question=QUESTION_GLOBAL,
            question_template_sha256=template_sha256(templates[QUESTION_GLOBAL]),
            # The global question quotes the arm's opening paragraph, so the
            # arm's wording is part of what identifies it.
            preamble_template_sha256=arm_sha256,
            preamble_template_file=arm_template_file,
            **common,
        ),
    }
    print(
        f"{len(recording_ids)} recordings -> {store.directory} "
        f"(arm={args.arm}, questions={','.join(args.questions)}, "
        f"template={template_label(arm_template)} sha256={arm_sha256[:16]}, "
        f"model={args.model}, config {configs[QUESTION_STEPS]['config_id']}/"
        f"{configs[QUESTION_GLOBAL]['config_id']}, "
        f"code {configs[QUESTION_STEPS]['code_version']})",
        flush=True,
    )

    failures = 0
    for position, recording_id in enumerate(recording_ids, 1):
        pending = pending_questions(
            store,
            recording_id,
            args.questions,
            force=args.force,
            retry_errors=args.retry_errors,
        )
        try:
            sop, graph = annotations.for_recording(recording_id)
        except Exception:
            # Without a SOP there is no record to write: the verdicts, the
            # activity and the question are all derived from it.
            failures += 1
            print(f"[{position}/{len(recording_ids)}] {recording_id}: FAILED", flush=True)
            traceback.print_exc()
            continue

        started_at = _now()
        started = time.perf_counter()
        loaded: RecordingInputs | None = None
        # Two stages, two failure modes: whatever breaks, the recording still
        # leaves the records it owes, with as much of the truth in them as
        # was known by then.
        with tempfile.TemporaryDirectory(prefix=f"single-call-{recording_id}-") as workdir:
            try:
                loaded = load_recording(
                    recording_id,
                    artifacts_root=args.artifacts_root,
                    videos_dir=args.videos_dir,
                    arm=args.arm,
                    caption_variant=args.caption_variant,
                )
                prepared = prepare_recording(
                    loaded,
                    sop=sop,
                    graph=graph,
                    arm=args.arm,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                    budget=args.visual_token_budget,
                    video_parts_s=args.video_parts_s,
                    templates=templates,
                    workdir=Path(workdir),
                )
            except Exception as error:
                failures += 1
                detail = summarize_exception(error)
                print(
                    f"[{position}/{len(recording_ids)}] {recording_id}: error "
                    f"({detail.splitlines()[-1]})",
                    flush=True,
                )
                if args.dry_run:
                    # A rehearsal writes nothing. An error record here would
                    # be read as an answer by the resume check, and the real
                    # run would then skip the very recording that failed.
                    continue
                for question in pending:
                    save_answer(
                        store,
                        recording_id,
                        question,
                        error_record(
                            question=question,
                            recording_id=recording_id,
                            sop=sop,
                            video=None if loaded is None else loaded.video,
                            segments=() if loaded is None else loaded.segments,
                            config=configs[question],
                            started_at=started_at,
                            wall_s=time.perf_counter() - started,
                            detail=detail,
                        ),
                    )
                continue

            if args.dry_run:
                print(describe_plan(recording_id, prepared), flush=True)
                continue

            for question in pending:
                record = ask_question(
                    question=question,
                    prepared=prepared,
                    recording_id=recording_id,
                    sop=sop,
                    client=client,
                    config=configs[question],
                )
                save_answer(store, recording_id, question, record)
                if record["status"] == "error":
                    failures += 1
                usage = record["usage"]
                print(
                    f"[{position}/{len(recording_ids)}] {recording_id} {question}: "
                    f"{record['status']} ({record['wall_s']:.0f}s) "
                    f"tokens={usage['tokens_in']}+{usage['tokens_out']}",
                    flush=True,
                )
    print(f"done, {failures} failures", flush=True)
    return failures


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_questions(value: str) -> list[str]:
    """Read ``--questions`` into the canonical call order."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in QUESTION_ORDER]
    if unknown or not names:
        raise ValueError(
            f"--questions takes {','.join(QUESTION_ORDER)}, got {value!r}"
        )
    return [name for name in QUESTION_ORDER if name in names]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True, help="root of the runs")
    parser.add_argument("--run-name", required=True, help="subdirectory of --out")
    parser.add_argument(
        "--fps",
        type=float,
        help="frames per second sent; required by --arm video, no default "
        "(the published configuration uses 1)",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="frame width in pixels; required by --arm video, no default "
        "(the published configuration uses 448)",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="frame height in pixels; required by --arm video, no default "
        "(the published configuration uses 256)",
    )
    parser.add_argument(
        "--visual-token-budget",
        type=int,
        default=DEFAULT_VISUAL_TOKEN_BUDGET,
        help="a recording whose frames would cost more than this is sampled "
        "at a proportionally lower rate, recorded per recording "
        f"(default: {DEFAULT_VISUAL_TOKEN_BUDGET})",
    )
    parser.add_argument(
        "--video-parts-s",
        type=float,
        help="optional: send the video as consecutive parts of this many "
        "seconds, one video item each, instead of a single item",
    )
    parser.add_argument(
        "--caption-variant",
        default=DEFAULT_CAPTION_VARIANT,
        help="caption artifact suffix (captions.<name>.json) of --arm captions",
    )
    parser.add_argument(
        "--video-template",
        type=Path,
        default=PROMPTS_DIR / VIDEO_TEMPLATE_FILE,
        help="per-step question template of --arm video (default: "
        "prompts/single_call/video_question.txt); a non-default file is "
        "named, with its sha256, in the run's configuration",
    )
    parser.add_argument(
        "--captions-template",
        type=Path,
        default=PROMPTS_DIR / CAPTIONS_TEMPLATE_FILE,
        help="per-step question template of --arm captions (default: "
        "prompts/single_call/captions_question.txt); same rules as "
        "--video-template",
    )
    parser.add_argument(
        "--questions",
        default=",".join(QUESTION_ORDER),
        help="which questions to ask: steps, global or both (default: both, "
        "the global question first)",
    )
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=PROVIDER_VLLM,
        help="vllm (default): the local vLLM server, thinking disabled, seed "
        "sent; openai-compatible: a hosted model behind an OpenAI-compatible "
        "endpoint, no vLLM-specific request fields and no seed (--arm "
        "captions only)",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="environment variable holding the API key for --provider "
        f"openai-compatible (default: {DEFAULT_API_KEY_ENV}); the vllm "
        "provider reads $VLLM_API_KEY",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--shard", help="I/N: process only recordings where index %% N == I"
    )
    parser.add_argument("--limit", type=int, help="stop after this many recordings")
    parser.add_argument(
        "--force", action="store_true", help="redo recordings that already answered"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="redo the questions whose record has status 'error'",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the frames, the MP4 and the questions and print what they "
        "cost, without calling the server",
    )
    args = parser.parse_args(argv)
    try:
        args.questions = parse_questions(args.questions)
    except ValueError as error:
        parser.error(str(error))
    if args.provider != PROVIDER_VLLM and args.arm != ARM_CAPTIONS:
        parser.error(f"--provider {args.provider} supports only --arm captions")
    if args.arm == ARM_VIDEO:
        missing = [
            name
            for name in ("fps", "width", "height")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "--arm video requires " + ", ".join(f"--{name}" for name in missing)
            )
    # A template that cannot be rendered is a mistake to discover now, not
    # hours into a batch; run_task.py reads its overrides in parse_args for
    # the same reason.
    try:
        read_template(args.video_template, VIDEO_PLACEHOLDERS)
        read_template(args.captions_template, CAPTIONS_PLACEHOLDERS)
        read_template(PROMPTS_DIR / GLOBAL_TEMPLATE_FILE, GLOBAL_PLACEHOLDERS)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args


def build_client(args: argparse.Namespace) -> OneShotClient:
    """The client this run talks to, with the provider's and arm's request body.

    ``openai-compatible`` sends neither the vLLM switches nor the seed, and
    reads its API key from the variable named by ``--api-key-env`` (the
    repository's ``.env`` is loaded first); a missing key stops the run here
    rather than at the first request.
    """
    if args.provider == PROVIDER_OPENAI_COMPATIBLE:
        load_dotenv()
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(
                f"error: --provider {args.provider} needs an API key in the "
                f"environment variable {args.api_key_env}, which is not set"
            )
        return OneShotClient(
            model=args.model,
            base_url=args.base_url,
            api_key=api_key,
            temperature=args.temperature,
            seed=None,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
            extra_body=None,
        )
    extra_body = dict(NO_THINKING_BODY)
    if args.arm == ARM_VIDEO:
        extra_body.update(VIDEO_BODY)
    return OneShotClient(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout,
        extra_body=extra_body,
    )


def main(
    argv: Sequence[str] | None = None, *, client: OneShotClient | None = None
) -> int:
    """Entry point; ``client`` is the seam the tests inject a fake through.

    Returns the process's exit code, not the number of failures: ``EXIT_OK``
    or ``EXIT_FAILED_RECORDINGS``, which lets a calling script distinguish a
    batch that finished with failed recordings from a crash or a usage error.
    The count itself is on the last line of the log.
    """
    args = parse_args(argv)
    if client is None and not args.dry_run:
        client = build_client(args)
    return EXIT_FAILED_RECORDINGS if run_batch(args, client) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

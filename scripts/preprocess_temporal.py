"""Batch temporal-memory preprocessing with per-stage artifacts and resume.

Stages per video (each skipped when its artifact already exists):
    visual_embed  -> segment_visual_embeddings[.slug].npy (GPU, aligner; stable)
    caption       -> captions.json                 (Qwen3-VL; prompt-dependent)
    caption_embed -> caption_embeddings.npy        (GPU, Qwen3-Embedding)

``--aligner-model`` picks the aligner from ``adapters.aligner``'s registry and
defaults to PE-Core L (``timm/PE-Core-L-14-336``), spelled once in
``agent.config.DEFAULT_ALIGNER_MODEL``, so the stage writes what inference
reads. Every aligner except X-CLIP (unsuffixed artifact name) suffixes its
artifact with the model's slug (segment_visual_embeddings.<slug>.npy), so
several aligners can coexist in one root the way the caption variants do. Caption embeddings default to Qwen3-Embedding-4B
(``--text-embedder-model``).

``caption`` runs either on a local checkpoint (``--captioner qwen``, the GPU
of this process) or through a chat endpoint (``--captioner api``, the model
served by ``serving/vllm``), which is where ``--workers`` concurrent requests
buy throughput and ``--caption-transport`` decides whether a clip travels as
one video or as one image per frame.

The instruction can come from a file (``--prompt-file``) instead of the
command line, with one block per clip position; with ``--context-segments k``
the clip sent for a segment carries its ``k`` neighbours on each side and the
block matching its position asks about the middle of it.

``visual_embed`` depends only on the segment grid and sampling constants, so
it is shared by every caption set. Caption artifacts are suffixed with the
variant name (captions.<name>.json, defaulting to ``--prompt-name``) so
several caption sets coexist in one artifact root.

The published caption sets are:
    action    ``--prompt-name action`` (built-in instruction)
    abst_ctx  ``--prompt-file prompts/captions/abstention_context.md
              --variant-name abst_ctx --context-segments 1``
    abst      ``--prompt-file prompts/captions/abstention.md --variant-name abst``
    ego       ``--prompt-file prompts/captions/ego.md --variant-name ego``
              (EgoSchema)
    state     ``--prompt-name state`` (built-in instruction)
    order     ``--prompt-name order`` (built-in "phase" instruction)

Usage:
    uv run python scripts/preprocess_temporal.py --videos-dir VIDEOS_DIR \
        --out PREPROCESS_DIR
    uv run python scripts/preprocess_temporal.py --videos-dir VIDEOS_DIR \
        --out PREPROCESS_DIR --stages caption,caption_embed \
        --prompt-file prompts/captions/abstention_context.md \
        --variant-name abst_ctx --context-segments 1
"""

import argparse
import json
import time
import traceback
from pathlib import Path

from _env import load_dotenv
from _wiring import LOCAL_BASE_URL, local_api_key, request_body
from video_agent.adapters.aligner import aligner_slug, build_aligner
from video_agent.adapters.captioner import (
    CAPTION_INSTRUCTIONS,
    DEFAULT_POSITION,
    DEFAULT_PROMPT,
    PROMPT_POSITIONS,
    load_prompt_file,
)
from video_agent.adapters.captioner_openai import TRANSPORTS
from video_agent.agent.config import DEFAULT_ALIGNER_MODEL
from video_agent.lineage import code_version
from video_agent.media import open_video
from video_agent.memory.artifacts import VideoArtifacts
from video_agent.memory.preprocess import get_video_segments
from video_agent.memory.store import visual_embed_stage, visual_embeddings_file
from video_agent.memory.temporal import (
    context_position,
    generate_segment_caption_embeddings,
    generate_segment_captions_chunked,
    generate_segment_visual_embeddings_chunked,
)

CLIP_SAMPLE_FPS = 5.0  # 10 frames per 2-s segment (reference aligner sampling)
DEFAULT_VARIANT = "default"
"""Artifact suffix when no variant and no instruction were named."""

CONTEXT_CAPABLE_BACKENDS = ("api", "qwen")
"""Captioner backends that take one prompt per clip.

Both current backends do; the tuple is here so a backend added without that
ability fails at the flag instead of silently captioning every clip of a
context run with the middle instruction.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("videos", nargs="*", type=Path)
    parser.add_argument("--videos-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stages",
        default="visual_embed",
        help="comma-separated: visual_embed,caption,caption_embed",
    )
    parser.add_argument(
        "--variant-name",
        default=DEFAULT_VARIANT,
        help="artifact suffix for caption stages (captions.<name>.json); "
        "lets caption sets from different prompts coexist in one root. "
        "Defaults to --prompt-name when one is given",
    )
    instruction = parser.add_mutually_exclusive_group()
    instruction.add_argument(
        "--prompt-text",
        help="captioner prompt; omitted = adapter's DEFAULT_PROMPT",
    )
    instruction.add_argument(
        "--prompt-name",
        choices=list(CAPTION_INSTRUCTIONS),
        help="one of the built-in caption instructions",
    )
    instruction.add_argument(
        "--prompt-file",
        type=Path,
        help="markdown file with one '## user [middle|first|last]' block per "
        "clip position; the middle block is the instruction and the other "
        "two are only used with --context-segments > 0",
    )
    parser.add_argument(
        "--context-segments",
        type=int,
        default=0,
        help="neighbouring segments added to each side of the clip sent to "
        "the captioner (0 = the 2-second clip alone). The question is still "
        "about the middle segment; the prompt of the position says so",
    )
    parser.add_argument(
        "--captioner",
        choices=["qwen", "api"],
        default="api",
        help="captioner backend; 'api' captions through a chat endpoint "
        "(the served multimodal model, see serving/vllm/README.md)",
    )
    parser.add_argument(
        "--base-url",
        default=LOCAL_BASE_URL,
        help="endpoint of the --captioner api backend",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="concurrent requests of the --captioner api backend; the server "
        "batches whatever arrives together, so raising this is what buys "
        "throughput, up to --clips-per-batch (the clips decoded at a time)",
    )
    parser.add_argument(
        "--caption-max-tokens",
        type=int,
        default=64,
        help="reply budget per clip on the --captioner api backend",
    )
    parser.add_argument(
        "--caption-transport",
        choices=list(TRANSPORTS),
        default="video",
        help="how a clip reaches the --captioner api backend: 'video' sends "
        "its frames as one MP4 (temporal patch merging, ~4.6x fewer prompt "
        "tokens), 'images' as one still per frame. Ignored by --captioner "
        "qwen, which always feeds the frames as a video locally",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-candidates", type=int, default=1)
    parser.add_argument(
        "--aligner-model",
        default=DEFAULT_ALIGNER_MODEL,
        help="video-text aligner of the visual_embed stage; every id except "
        "X-CLIP (unsuffixed artifact name) writes "
        "segment_visual_embeddings.<slug>.npy "
        "(see adapters/aligner.py for the ids)",
    )
    parser.add_argument(
        "--captioner-model",
        default="Qwen/Qwen3.5-9B",
        help="local checkpoint, or the model id the endpoint serves with "
        "--captioner api",
    )
    parser.add_argument("--text-embedder-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--clips-per-batch", type=int, default=16)
    parser.add_argument("--caption-max-frames", type=int, default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard", help="I/N: process videos where idx %% N == I")
    parser.add_argument("--force", action="store_true", help="recompute cached stages")
    args = parser.parse_args()

    # A named instruction names the artifact too, unless the caller said
    # otherwise: --prompt-name state writes captions.state.json.
    if args.prompt_name and args.variant_name == DEFAULT_VARIANT:
        args.variant_name = args.prompt_name
    # Resolve the aligner's artifact suffix here so an unknown id fails
    # before a single video is decoded, and so every use of the name in this
    # script reads the same string.
    try:
        args.aligner_slug = aligner_slug(args.aligner_model)
    except ValueError as error:
        parser.error(str(error))
    if args.captioner == "api" and (args.do_sample or args.num_candidates > 1):
        parser.error(
            "--do-sample/--num-candidates > 1 need the local captioner: "
            "candidate sampling is a generation mechanic the chat endpoint "
            "does not expose (drop them, or use --captioner qwen)"
        )
    if args.context_segments < 0:
        parser.error(f"--context-segments must be >= 0, got {args.context_segments}")
    if args.context_segments and args.captioner not in CONTEXT_CAPABLE_BACKENDS:
        parser.error(
            f"--context-segments needs a captioner that takes one prompt per "
            f"clip ({', '.join(CONTEXT_CAPABLE_BACKENDS)}), not "
            f"{args.captioner!r}"
        )
    if args.context_segments and args.caption_max_frames is not None:
        parser.error(
            "--context-segments and --caption-max-frames are incompatible: "
            "the frame budget would resample the whole widened clip and throw "
            "away most of the very context the flag adds (drop one of them)"
        )
    # A prompt file is read before a single video is decoded, so a missing
    # block is a flag error and not a crash three hours into a batch.
    if args.prompt_file:
        try:
            load_prompt_file(args.prompt_file)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    return args


def prompt_blocks(args: argparse.Namespace) -> dict[str, str]:
    """The instruction of each clip position, gaps filled by the middle one.

    Pure and cheap, so the stage that needs one prompt per segment can call
    it per video; whether a missing ``first``/``last`` deserves a warning is
    left to ``missing_positions``, which the caller prints once.
    """
    if getattr(args, "prompt_file", None):
        blocks = load_prompt_file(args.prompt_file)
    elif args.prompt_name:
        blocks = {DEFAULT_POSITION: CAPTION_INSTRUCTIONS[args.prompt_name]}
    else:
        blocks = {DEFAULT_POSITION: args.prompt_text or DEFAULT_PROMPT}
    return {
        position: blocks.get(position, blocks[DEFAULT_POSITION])
        for position in PROMPT_POSITIONS
    }


def missing_positions(args: argparse.Namespace) -> list[str]:
    """Positions this run will caption with the middle block for lack of one."""
    if not getattr(args, "prompt_file", None) or not args.context_segments:
        return []
    written = load_prompt_file(args.prompt_file)
    return [position for position in PROMPT_POSITIONS if position not in written]


def resolve_prompt(args: argparse.Namespace) -> str:
    """The instruction the captioner is given, whatever named it.

    One string — the middle block — because that is what ``meta.json``
    records as *the* prompt of the variant and what a run without context
    sends to every clip; ``prompt_blocks`` has the other positions.
    """
    return prompt_blocks(args)[DEFAULT_POSITION]


def captioner_settings(args: argparse.Namespace) -> dict:
    """How this run's captioner would be built, without building it.

    Pure, so a test can read the request shape a flag combination produces
    without a checkpoint or a server: ``backend`` and ``model`` name the
    adapter, ``captioner`` and (on the API path) ``chat`` are the keyword
    arguments each one gets.
    """
    settings: dict = {
        "backend": args.captioner,
        "model": args.captioner_model,
        # Not a captioner argument: the widened clip is assembled by
        # ``memory.temporal`` and the adapter only sees the prompts it
        # produces. It is recorded here so a test can read it off the flags.
        "context_segments": args.context_segments,
        "captioner": {"prompt": resolve_prompt(args)},
    }
    if args.captioner == "api":
        transport = args.caption_transport
        settings["chat"] = {
            "base_url": args.base_url,
            "api_key": local_api_key(),
            "temperature": 0.0,
            # The rate the clips were decoded at: it stamps the MP4 a clip
            # is written into, and nothing else.
            "video_fps": CLIP_SAMPLE_FPS,
            "extra_body": request_body(video=transport == "video"),
        }
        settings["captioner"].update(
            frames_per_clip=args.caption_max_frames,
            max_tokens=args.caption_max_tokens,
            workers=args.workers,
            transport=transport,
        )
    else:
        settings["captioner"].update(
            max_frames_per_clip=args.caption_max_frames,
            do_sample=args.do_sample,
            top_p=args.top_p,
            temperature=args.temperature,
            num_candidates=args.num_candidates,
        )
    return settings


class LazyModels:
    """Load each model on first use so skipped stages cost nothing."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._aligner = None
        self._captioner = None
        self._text_embedder = None

    @property
    def aligner(self):
        if self._aligner is None:
            # The aligner never receives more clips than a decode batch, so
            # giving it the same number makes --clips-per-batch the single
            # throughput knob: the heavier aligners (PE-Core G at 448 px,
            # SigLIP2 SO400M at 384 px) encode 8 frames per clip and need it.
            self._aligner = build_aligner(
                self.args.aligner_model,
                device=self.args.device,
                batch_size=self.args.clips_per_batch,
            )
        return self._aligner

    @property
    def captioner(self):
        if self._captioner is None:
            settings = captioner_settings(self.args)
            if settings["backend"] == "api":
                from video_agent.adapters.captioner_openai import OpenAIVisionCaptioner
                from video_agent.adapters.chat_openai import OpenAICompatibleChat

                chat = OpenAICompatibleChat(settings["model"], **settings["chat"])
                self._captioner = OpenAIVisionCaptioner(
                    chat, **settings["captioner"]
                )
            else:
                from video_agent.adapters.captioner import Qwen3VLCaptioner

                self._captioner = Qwen3VLCaptioner(
                    settings["model"], **settings["captioner"]
                )
        return self._captioner

    @property
    def text_embedder(self):
        if self._text_embedder is None:
            from video_agent.adapters.text_embedder import SentenceTransformerEmbedder

            self._text_embedder = SentenceTransformerEmbedder(
                self.args.text_embedder_model, device=self.args.device
            )
        return self._text_embedder


def stage_visual_embed(video, segments, artifacts, models, args, version):
    intervals = [(s.start_s, s.end_s) for s in segments]
    aligner = models.aligner
    embeddings = generate_segment_visual_embeddings_chunked(
        aligner,
        video,
        intervals,
        clip_sample_fps=CLIP_SAMPLE_FPS,
        clips_per_batch=args.clips_per_batch,
    )
    name = visual_embeddings_file(args.aligner_slug)
    artifacts.save_array(name, embeddings)
    artifacts.record_stage(
        visual_embed_stage(args.aligner_slug),
        artifact=name,
        model=args.aligner_model,
        params={
            "aligner_slug": args.aligner_slug,
            # What the aligner actually consumed and produced, read off the
            # loaded model, so aligners can be compared by the frame count
            # and the width each one really used.
            "frames_per_clip": getattr(aligner, "num_frames", None),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.size else None,
            "clip_sample_fps": CLIP_SAMPLE_FPS,
            "seconds_per_segment": 2.0,
            "n_segments": len(segments),
        },
        code_version=version,
    )
    return embeddings.shape


def usage_delta(before: dict, after: dict) -> dict:
    """What one stage alone cost, field by field.

    The captioner is built once and bills every video it captions, so the
    counters have to be differenced to belong to this stage's record.
    """
    return {key: value - before[key] for key, value in after.items()}


def stage_caption(video, segments, artifacts, models, args, version):
    api = args.captioner == "api"
    started = time.monotonic()
    captioner = models.captioner
    billed_before = captioner.usage.to_dict() if api else {}
    intervals = [(s.start_s, s.end_s) for s in segments]
    blocks = prompt_blocks(args)
    prompts = [
        blocks[context_position(index, len(segments), args.context_segments)]
        for index in range(len(segments))
    ]
    captions = generate_segment_captions_chunked(
        captioner,
        video,
        intervals,
        clip_sample_fps=CLIP_SAMPLE_FPS,
        clips_per_batch=args.clips_per_batch,
        context_segments=args.context_segments,
        prompts=prompts,
    )
    elapsed = time.monotonic() - started
    name = f"captions.{args.variant_name}.json"
    artifacts.save_json(name, captions)
    artifacts.record_stage(
        f"caption.{args.variant_name}",
        artifact=name,
        model=args.captioner_model,
        params={
            "variant_name": args.variant_name,
            "prompt_name": args.prompt_name,
            # The text actually sent, not the flag: the default and the
            # named instructions would otherwise record as null.
            "prompt": resolve_prompt(args),
            "prompt_file": str(args.prompt_file) if args.prompt_file else None,
            # Only when the clip is widened or the blocks came from a file:
            # otherwise the three positions are the one prompt above.
            "prompts_by_position": (
                blocks if (args.prompt_file or args.context_segments) else None
            ),
            "context_segments": args.context_segments,
            "captioner_backend": args.captioner,
            # Only the API path has a choice of transport; the local one
            # always feeds the frames to the processor as a video.
            "transport": args.caption_transport if api else None,
            "do_sample": args.do_sample,
            "top_p": args.top_p if args.do_sample else None,
            "temperature": args.temperature if args.do_sample else None,
            "num_candidates": args.num_candidates,
            "workers": args.workers if api else None,
            "max_tokens": args.caption_max_tokens if api else None,
            "clip_sample_fps": CLIP_SAMPLE_FPS,
            "max_frames_per_clip": args.caption_max_frames,
            "n_segments": len(segments),
            # Cost of the stage: what the endpoint billed and how long the
            # whole thing took, since the API path is what a run is charged
            # for.
            "usage": (
                usage_delta(billed_before, captioner.usage.to_dict())
                if api
                else None
            ),
            "wall_s": round(elapsed, 3),
        },
        code_version=version,
    )
    return len(captions)


def stage_caption_embed(artifacts, models, args, version):
    captions = artifacts.load_json(f"captions.{args.variant_name}.json")
    embeddings = generate_segment_caption_embeddings(models.text_embedder, captions)
    name = f"caption_embeddings.{args.variant_name}.npy"
    artifacts.save_array(name, embeddings)
    artifacts.record_stage(
        f"caption_embed.{args.variant_name}",
        artifact=name,
        model=args.text_embedder_model,
        params={"variant_name": args.variant_name},
        code_version=version,
    )
    return embeddings.shape


def main() -> None:
    args = parse_args()
    load_dotenv()
    stages = [s for s in args.stages.split(",") if s]
    unknown = set(stages) - {"visual_embed", "caption", "caption_embed"}
    if unknown:
        raise SystemExit(f"unknown stages: {sorted(unknown)}")


    videos = list(args.videos)
    if args.videos_dir:
        videos.extend(sorted(args.videos_dir.glob("*.mp4")))
    if args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        videos = videos[index::total]
    if args.limit:
        videos = videos[: args.limit]

    absent = missing_positions(args)
    if absent:
        print(
            f"warning: {args.prompt_file} has no {absent} block; those clips "
            f"get the {DEFAULT_POSITION!r} instruction",
            flush=True,
        )

    version = code_version()
    models = LazyModels(args)
    print(f"{len(videos)} videos, stages={stages} -> {args.out} "
          f"(code {version})", flush=True)

    failures = 0
    for position, path in enumerate(videos, 1):
        started = time.monotonic()
        try:
            video = open_video(path)
            segments = get_video_segments(video)
            artifacts = VideoArtifacts(args.out, path.stem)
            done = []
            if "visual_embed" in stages and (
                args.force
                or not artifacts.has(visual_embeddings_file(args.aligner_slug))
            ):
                shape = stage_visual_embed(
                    video, segments, artifacts, models, args, version
                )
                done.append(f"visual={shape}")
            if "caption" in stages and (
                args.force
                or not artifacts.has(f"captions.{args.variant_name}.json")
            ):
                count = stage_caption(
                    video, segments, artifacts, models, args, version
                )
                done.append(f"captions={count}")
            if "caption_embed" in stages and (
                args.force
                or not artifacts.has(f"caption_embeddings.{args.variant_name}.npy")
            ):
                shape = stage_caption_embed(artifacts, models, args, version)
                done.append(f"caption_embed={shape}")
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

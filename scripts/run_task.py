"""Run the procedural task over a split: one agent episode per recording.

For every recording of a split the agent gets the recipe's SOP, its task
graph and the video's memories, and returns one execution and one order
verdict per step; the run record of each recording is written under
``<out>/<run-name>/<recording_id>/run.json`` and nothing is scored here —
``evaluate.py`` reads those records.

A run over a whole split takes hours, so it resumes: a recording that
already has a record is skipped unless ``--force``, and
``--shard I/N`` splits the work across workers deterministically.

Requires the ``agent`` extra for the API chat backend and the ``models``
extra for anything local (uv sync --extra agent --extra models).

Usage:
    uv run python scripts/run_task.py --split data_splits/dev.json \\
        --artifacts-root PREPROCESS_DIR --videos-dir VIDEOS_DIR \\
        --out RUNS_DIR --run-name NAME \\
        --chat vllm --model Qwen/Qwen3.5-9B --vqa api --device cpu \\
        --max-iterations 100 --caption-variant action --seed 20260903

    The same command with ``--caption-variant abst_ctx`` runs the agent on the
    abstention-with-context captions, and ``--memory empty`` or
    ``--memory shuffled`` runs the memory controls.
"""

import argparse
import random
import time
import traceback
from pathlib import Path
from typing import Sequence

from _env import load_dotenv
from _wiring import (
    DEFAULT_MAX_TOKENS,
    LazyModels,
    build_toolkit,
    check_object_memory_slug,
    react_config,
    tool_config,
)
from video_agent.adapters.aligner import aligner_slug
from video_agent.agent.config import DEFAULT_ALIGNER_MODEL
from video_agent.agent.tools import MEMORY_UNAVAILABLE
from video_agent.lineage import code_version
from video_agent.memory.ablation import (
    MEMORY_EMPTY,
    MEMORY_FULL,
    MEMORY_MODES,
    PERMUTATION_BASE_SEED,
)
from video_agent.memory.artifacts import VideoArtifacts
from video_agent.memory.store import object_memory_file
from video_agent.task.annotations import AnnotationSet, load_split
from video_agent.task.run import (
    PACKAGE_TEMPLATE,
    RunStore,
    TaskConfig,
    config_id,
    run_recording,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
"""Repository root; relative defaults resolve from here, not from the cwd."""

DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "data" / "captaincook4d" / "annotations"
VIDEO_SUFFIX = "_360p"
"""The videos are the 360p renders, and artifacts are keyed by their stem."""

OWN_SCHEMA_MARK = '{"steps": ['
"""A question file that carries this text spells out its own answer object and
may omit ``{schema}`` (see ``read_template``)."""

QUESTION_PLACEHOLDERS = (
    "{recipe_name}",
    "{n_segments}",
    "{steps}",
    "{prerequisites}",
    "{schema}",
)
"""Every placeholder ``task.prompt.build_question`` fills in the question."""

REACT_PLACEHOLDERS = ("{tools}", "{tool_names}", "{input}", "{agent_scratchpad}")
"""Every placeholder the ReAct loop fills in the orchestrator's prompt."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--subset",
        choices=["error", "normal", "all"],
        default="all",
        help="which half of the split to run",
    )
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument(
        "--caption-variant",
        default="default",
        help="caption artifact suffix (captions.<name>.json)",
    )
    parser.add_argument("--out", type=Path, required=True, help="root of the runs")
    parser.add_argument("--run-name", required=True, help="subdirectory of --out")
    parser.add_argument(
        "--question-file",
        type=Path,
        help="read the task question from this file (UTF-8) instead of the "
        f"packaged task_question; it must contain {', '.join(QUESTION_PLACEHOLDERS)} "
        "({schema} may be left out by a file that spells out its own answer "
        "object, i.e. contains '{\"steps\": ['). "
        "The digest of the text is part of config_id, so another wording is "
        "another configuration",
    )
    parser.add_argument(
        "--task-prompt-file",
        type=Path,
        help="read the orchestrator's ReAct prompt from this file (UTF-8) "
        "instead of the packaged task_prompt; it must contain "
        f"{', '.join(REACT_PLACEHOLDERS)}",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--chat",
        choices=["openrouter", "vllm", "transformers"],
        default="vllm",
        help="orchestrator backend; 'vllm' is the locally served model "
        "(see serving/vllm/README.md)",
    )
    parser.add_argument(
        "--base-url",
        help="endpoint of the chat backend; defaults to the usual address of "
        "--chat (OpenRouter, or localhost:8000 for vllm)",
    )
    parser.add_argument(
        "--vqa",
        choices=["qwen", "api", "none"],
        default="api",
        help="visual_question_answering backend; 'none' drops the tool. "
        "When the orchestrator is served on the same GPU there may be no room "
        "left for the local Qwen3-VL; then use 'api' (the served model "
        "answers) or 'none'",
    )
    parser.add_argument("--vqa-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--vqa-style",
        choices=["two_questions", "single"],
        default="two_questions",
        help="how the tool queries the VQA model; 'single' asks one question "
        "and returns the reply verbatim",
    )
    parser.add_argument(
        "--no-objects",
        action="store_true",
        help="ablation: drop the object memory and its sub-agent",
    )
    parser.add_argument(
        "--memory",
        choices=list(MEMORY_MODES),
        default=MEMORY_FULL,
        help="which memory the agent gets: 'full' is the recording's own; "
        "'empty' withholds it and every tool answers "
        f"'{MEMORY_UNAVAILABLE}' (empty-memory control); 'shuffled' keeps the "
        "same captions, vectors and objects under permuted segment names, "
        f"drawn per recording from {PERMUTATION_BASE_SEED} plus its position "
        "in the split and written to permutation.json beside the record "
        "(shuffled-memory control)",
    )
    parser.add_argument("--text-embedder-model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument(
        "--aligner-model",
        default=DEFAULT_ALIGNER_MODEL,
        help="video-text aligner of segment_localization: it embeds the step "
        "text and names the visual embeddings the memory is read from "
        "(segment_visual_embeddings.<slug>.npy; X-CLIP uses the unsuffixed "
        "artifact name; see adapters/aligner.py for the ids)",
    )
    parser.add_argument(
        "--object-memory",
        help="which detector's object memory the sub-agent queries "
        "(object_memory.<slug>.pkl); omitted reads the unsuffixed files of "
        "the default detector (RT-DETRv2)",
    )
    parser.add_argument(
        "--vqa-base-url",
        help="with --vqa api, send the visual questions to this OpenAI-compatible "
        "endpoint (a local vLLM server) instead of the orchestrator's; used to "
        "run a hosted orchestrator while Qwen3.5-9B keeps answering the "
        "visual questions",
    )
    parser.add_argument(
        "--vqa-api-model",
        default="Qwen/Qwen3.5-9B",
        help="model name served at --vqa-base-url",
    )
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument(
        "--device",
        default="cpu",
        help="where the embedders run (text embedder, aligner, SigLIP); pass "
        "cpu when the GPU is taken by the served orchestrator",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="per-turn token cap for the orchestrator; 0 means unlimited",
    )
    parser.add_argument(
        "--chat-timeout",
        type=float,
        help="seconds one request may take; defaults to 300 for --chat vllm "
        "and to 120 elsewhere",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="let the served model reason before it answers; the reasoning "
        "block is recorded per turn and never reaches the scratchpad",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--shard", help="I/N: process only recordings where index %% N == I"
    )
    parser.add_argument("--limit", type=int, help="stop after this many recordings")
    parser.add_argument(
        "--force", action="store_true", help="rerun recordings that already have a record"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="rerun recordings whose record has status 'error'",
    )
    parser.add_argument(
        "--order",
        choices=["split", "shuffle"],
        default="split",
        help="'shuffle' randomizes the order with --seed, so a partial run "
        "still covers every recipe",
    )
    args = parser.parse_args(argv)
    if args.vqa_base_url and args.vqa != "api":
        parser.error("--vqa-base-url needs --vqa api")
    # An unknown aligner is a typo, not a run: refuse it here, before a batch
    # of recordings fails one by one on the same name. A template file
    # that is missing or unrenderable is the same kind of mistake, so it is
    # read here too — hours before the first model is loaded.
    try:
        aligner_slug(args.aligner_model)
        check_object_memory_slug(args.object_memory)
        if args.question_file is not None:
            read_template(args.question_file, QUESTION_PLACEHOLDERS)
        if args.task_prompt_file is not None:
            read_template(args.task_prompt_file, REACT_PLACEHOLDERS)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return args


def read_template(path: Path, placeholders: Sequence[str]) -> str:
    """Read a template override, refusing one that could not be rendered.

    Both templates are rendered by ``str.replace`` (see ``task.prompt`` and
    the ReAct loop), so a missing placeholder fails silently: the question
    would reach the model without its steps, or the prompt without its
    tools, and the run would read as a bad model rather than as a bad file.
    """
    text = Path(path).read_text(encoding="utf-8")
    missing = [name for name in placeholders if name not in text]
    if "{schema}" in missing and OWN_SCHEMA_MARK in text:
        # The question spells out its own answer object instead of taking the
        # packaged one (e.g. a question that drops the order fields from the
        # answer). ``verdicts`` reads absent fields as
        # indeterminate, so the record stays parseable.
        missing.remove("{schema}")
    if missing:
        raise ValueError(f"{path} is missing the placeholder(s) {', '.join(missing)}")
    return text


def resolve_template(
    path: Path | None, name: str, placeholders: Sequence[str]
) -> tuple[str, str]:
    """The template this run runs, and where it was read from.

    Without an override the packaged template is used byte for byte, and the
    source is recorded as ``"package"``; with one, the file's text and its
    path, which the record keeps beside the digest of the text.
    """
    from video_agent.agent.prompts import load_prompt

    if path is None:
        return load_prompt(name), PACKAGE_TEMPLATE
    return read_template(path, placeholders), str(path)


def select_recordings(
    ids_error: Sequence[str],
    ids_normal: Sequence[str],
    args: argparse.Namespace,
    store: RunStore,
) -> list[str]:
    """Decide which recordings this process runs, in which order.

    Sorted first so that sharding is a property of the split and not of the
    file's line order: worker ``I`` of ``N`` always gets the same recordings.
    Resumption is decided after sharding and before ``--limit``, so a second
    pass with the same flags picks up exactly what is missing.
    """
    if args.subset == "error":
        selected = list(ids_error)
    elif args.subset == "normal":
        selected = list(ids_normal)
    else:
        selected = list(ids_error) + list(ids_normal)
    recording_ids = sorted(set(selected))

    if args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        if total <= 0 or not 0 <= index < total:
            raise ValueError(f"--shard {args.shard} is not a valid I/N with 0 <= I < N")
        recording_ids = recording_ids[index::total]

    if not args.force:
        recording_ids = [
            recording_id
            for recording_id in recording_ids
            if not store.has(recording_id)
            or (args.retry_errors and store.load(recording_id).get("status") == "error")
        ]

    if args.order == "shuffle":
        random.Random(args.seed).shuffle(recording_ids)
    if args.limit:
        recording_ids = recording_ids[: args.limit]
    return recording_ids


def split_ordinals(
    ids_error: Sequence[str], ids_normal: Sequence[str]
) -> dict[str, int]:
    """Position of every recording in its split, 0-based, sorted by id.

    The number that seeds a recording's permutation under ``--memory
    shuffled``, and the reason it is read off the whole split rather than off
    the list this process runs: ``--shard``, ``--subset`` and ``--limit``
    decide who does the work, not which shuffle a recording gets, so the four
    workers of one run and a rerun of a single recording all draw the same
    permutation for it. It is the sorted order ``select_recordings`` starts
    from, before any of those flags apply.
    """
    return {
        recording_id: ordinal
        for ordinal, recording_id in enumerate(
            sorted(set(list(ids_error) + list(ids_normal)))
        )
    }


def check_object_memories(
    artifacts_root: Path, recording_ids: Sequence[str], slug: str | None
) -> list[str]:
    """Which of these recordings have no object memory under ``slug``.

    Asked before the batch and not per recording: a memory that was only
    preprocessed for half the split fails on the other half hours in, one
    recording at a time, and the record of each failure says nothing about
    the ones still to come. The paths are read directly rather than through
    ``VideoArtifacts``, whose constructor would create every directory it is
    asked about.
    """
    missing = []
    for recording_id in recording_ids:
        directory = Path(artifacts_root) / f"{recording_id}{VIDEO_SUFFIX}"
        if not (directory / object_memory_file(slug)).is_file():
            missing.append(recording_id)
    return missing


def task_config(
    args: argparse.Namespace,
    prompt_template: str,
    question_template: str,
    *,
    prompt_source: str = PACKAGE_TEMPLATE,
    question_source: str = PACKAGE_TEMPLATE,
) -> TaskConfig:
    """Freeze this run's configuration, prompts included, for the record.

    The templates enter the configuration twice over: by digest, which is
    part of ``config_id``, and by source, which is not — where a wording
    came from is provenance and would otherwise make the id depend on the
    machine the run happened on.
    """
    from video_agent.agent.prompts import prompt_sha256

    return TaskConfig(
        orchestrator_model=args.model,
        chat_backend=args.chat,
        thinking=args.thinking,
        vqa_backend=args.vqa,
        vqa_style=args.vqa_style,
        caption_variant=args.caption_variant,
        use_objects=not args.no_objects,
        seed=args.seed,
        temperature=args.temperature,
        react=react_config(args),
        tools=tool_config(args),
        prompt_sha256=prompt_sha256(prompt_template),
        question_template_sha256=prompt_sha256(question_template),
        code_version=code_version(),
        prompt_template_source=prompt_source,
        question_template_source=question_source,
        memory=args.memory,
        memory_seed=PERMUTATION_BASE_SEED,
        vqa_endpoint_model=(
            args.vqa_api_model if getattr(args, "vqa_base_url", None) else None
        ),
    )


def main() -> None:
    args = parse_args()
    load_dotenv()

    from video_agent.agent.react import ReActRunner
    from video_agent.memory.store import load_video_meta

    annotations = AnnotationSet.load(args.annotations_dir)
    ids_error, ids_normal = load_split(args.split)
    store = RunStore(args.out, args.run_name)
    recording_ids = select_recordings(ids_error, ids_normal, args, store)
    ordinals = split_ordinals(ids_error, ids_normal)

    # A named memory is a claim about what was preprocessed; check it once
    # here rather than let the batch discover it recording by recording. The
    # withheld condition opens no memory at all, so it has nothing to check.
    if args.object_memory and not args.no_objects and args.memory != MEMORY_EMPTY:
        missing = check_object_memories(
            args.artifacts_root, recording_ids, args.object_memory
        )
        if missing:
            raise SystemExit(
                f"{len(missing)} recordings have no "
                f"{object_memory_file(args.object_memory)}: "
                f"{', '.join(missing)}; run preprocess_objects.py for them, "
                "or drop --object-memory"
            )

    prompt_template, prompt_source = resolve_template(
        args.task_prompt_file, "task_prompt", REACT_PLACEHOLDERS
    )
    question_template, question_source = resolve_template(
        args.question_file, "task_question", QUESTION_PLACEHOLDERS
    )
    config = task_config(
        args,
        prompt_template,
        question_template,
        prompt_source=prompt_source,
        question_source=question_source,
    )
    models = LazyModels(args)
    runner = ReActRunner(models.chat, react_config(args))
    print(
        f"{len(recording_ids)} recordings -> {store.directory} "
        f"(model={args.model}, memory={args.memory}, config {config_id(config)}, "
        f"code {config.code_version})",
        flush=True,
    )

    failures = 0
    for position, recording_id in enumerate(recording_ids, 1):
        started = time.monotonic()
        try:
            sop, graph = annotations.for_recording(recording_id)
            stem = f"{recording_id}{VIDEO_SUFFIX}"
            artifacts = VideoArtifacts(args.artifacts_root, stem)
            video = load_video_meta(artifacts, args.videos_dir / f"{stem}.mp4")
            toolkit = build_toolkit(
                video, artifacts, models, args, runner, ordinals.get(recording_id)
            )
            if toolkit.permutation is not None:
                # Written before the episode, so the transcript it explains is
                # never on disk without it.
                store.save_permutation(recording_id, toolkit.permutation)
            try:
                record = run_recording(
                    recording_id=recording_id,
                    sop=sop,
                    graph=graph,
                    toolkit=toolkit,
                    runner=runner,
                    prompt_template=prompt_template,
                    question_template=question_template,
                    config=config,
                    permutation=toolkit.permutation,
                )
            finally:
                toolkit.close()
            store.save(recording_id, record)
            elapsed = time.monotonic() - started
            usage = record["usage_total"]
            print(
                f"[{position}/{len(recording_ids)}] {recording_id}: "
                f"{record['status']} ({elapsed:.0f}s) "
                f"llm={usage['llm_calls']} tools={usage['tool_calls']} "
                f"tokens={usage['tokens_in']}+{usage['tokens_out']}",
                flush=True,
            )
        except Exception:
            failures += 1
            print(
                f"[{position}/{len(recording_ids)}] {recording_id}: FAILED",
                flush=True,
            )
            traceback.print_exc()
    print(f"done, {failures} failures", flush=True)


if __name__ == "__main__":
    main()

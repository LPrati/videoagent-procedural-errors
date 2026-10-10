"""Answer one question about one preprocessed video with the ReAct agent.

Composition root for inference: loads the memories written by
``preprocess_temporal.py``/``preprocess_objects.py``, wires the tools around
them and runs the ReAct loop. Every knob that changes an answer (backends,
prompt, sampling, tool configuration) is echoed into the transcript so a run
can be reproduced from its own output.

Requires the ``agent`` extra for the API chat backend and the ``models``
extra for anything local (uv sync --extra agent --extra models).

Usage:
    uv run python scripts/ask.py VIDEOS_DIR/VIDEO.mp4 "question" \
        --artifacts-root PREPROCESS_DIR
    uv run python scripts/ask.py VIDEOS_DIR/VIDEO.mp4 "$(cat q.txt)" \
        --artifacts-root PREPROCESS_DIR --chat vllm --model Qwen/Qwen3.5-9B \
        --vqa api --device cpu --multiple-choice --out transcript.json --print-log
"""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from _env import load_dotenv
from _wiring import (
    DEFAULT_MAX_TOKENS,
    LazyModels,
    build_toolkit,
    check_object_memory_slug,
    react_config,
    tool_config,
)
from video_agent.agent.config import DEFAULT_ALIGNER_MODEL
from video_agent.lineage import code_version
from video_agent.memory.artifacts import VideoArtifacts

if TYPE_CHECKING:  # heavy/optional modules: imported inside the functions
    from video_agent.agent.runner import AgentResult, AgentRunner
    from video_agent.agent.tools import ToolKit


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare every knob that decides *how* a question gets answered.

    Shared with ``run_egoschema.py``, which asks the same agent the same way
    over a whole benchmark: the batch has to accept the flags of the single
    run, and a flag that changes an answer must change it identically in
    both. The scripts keep their own arguments — what to ask, where to write
    — and only the model, backend and loop settings live here.
    """
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument(
        "--caption-variant",
        default="default",
        help="caption artifact suffix (captions.<name>.json)",
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
        "and returns the reply verbatim, and with --vqa api it reproduces the "
        "reference gpt-4v tool (4 frames, one prompt, 200 tokens)",
    )
    parser.add_argument(
        "--no-objects",
        action="store_true",
        help="ablation: drop the object memory and its sub-agent",
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
    parser.add_argument("--siglip-model", default="google/siglip2-base-patch16-224")
    parser.add_argument(
        "--device",
        default="cpu",
        help="where the embedders run (text embedder, aligner, SigLIP); pass "
        "cpu when the GPU is taken by the served orchestrator",
    )
    parser.add_argument("--max-iterations", type=int, default=15)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("question")
    parser.add_argument(
        "--multiple-choice",
        action="store_true",
        help="use the multiple-choice prompt (reference EgoSchema setting)",
    )
    add_model_arguments(parser)
    parser.add_argument("--out", type=Path, help="write the transcript JSON here")
    parser.add_argument("--print-log", action="store_true")
    args = parser.parse_args(argv)
    try:
        check_object_memory_slug(args.object_memory)
    except ValueError as error:
        parser.error(str(error))
    return args


def ask(
    question: str,
    toolkit: "ToolKit",
    runner: "AgentRunner",
    prompt_template: str,
) -> "AgentResult":
    """Run the ReAct loop over the toolkit's main tools."""
    return runner.run(question, toolkit.main_tools(), prompt_template=prompt_template)


def run_config(args: argparse.Namespace, prompt_template: str) -> dict:
    """Everything about this run that could change its answer.

    Split out of ``transcript_payload`` so a batch over many questions can
    freeze the configuration once and stamp every record with it (see
    ``run_egoschema.py``): what varies there is the question, never this.
    """
    from video_agent.agent.prompts import prompt_sha256

    return {
        "model": args.model,
        "chat_backend": args.chat,
        "thinking": args.thinking,
        "vqa_backend": args.vqa,
        "caption_variant": args.caption_variant,
        "temperature": args.temperature,
        "seed": args.seed,
        "tool_config": tool_config(args).to_dict(),
        "react_config": react_config(args).to_dict(),
        "prompt_sha256": prompt_sha256(prompt_template),
        "code_version": code_version(),
    }


def transcript_payload(
    args: argparse.Namespace, prompt_template: str, result: "AgentResult"
) -> dict:
    """Assemble the JSON record of one run: answer, configuration and log."""
    return {
        "question": args.question,
        "answer": result.answer,
        "status": result.status,
        "config": run_config(args, prompt_template),
        "usage": result.usage.to_dict(),
        "transcript": result.transcript.to_dict(),
    }


def main() -> None:
    args = parse_args()
    load_dotenv()

    from video_agent.agent.prompts import load_prompt
    from video_agent.agent.react import ReActRunner
    from video_agent.memory.store import load_video_meta

    artifacts = VideoArtifacts(args.artifacts_root, args.video.stem)
    video = load_video_meta(artifacts, args.video)
    models = LazyModels(args)
    runner = ReActRunner(models.chat, react_config(args))
    toolkit = build_toolkit(video, artifacts, models, args, runner)
    template = load_prompt(
        "multiple_choice_prompt" if args.multiple_choice else "prompt"
    )
    print(
        f"{args.video.stem}: {toolkit.n_segments} segments, "
        f"{len(toolkit.main_tools())} tools, model={args.model}",
        flush=True,
    )
    try:
        result = ask(args.question, toolkit, runner, template)
    finally:
        toolkit.close()

    if args.print_log:
        print(result.transcript.render(), flush=True)
    print(f"status: {result.status}", flush=True)
    print(f"answer: {result.answer}", flush=True)
    print(f"usage: {result.usage.to_dict()}", flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(transcript_payload(args, template, result), indent=1)
        )
        print(f"transcript -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

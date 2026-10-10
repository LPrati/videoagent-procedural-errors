"""Composition of one video's agent, shared by every inference script.

``ask.py`` answers one question about one video and ``run_task.py`` runs the
procedural task over a whole split, but both build the same thing: the same
lazily loaded backends, the same tool and loop configuration, the same
toolkit around the same memories. Keeping that in one place is what makes
the two scripts comparable — a flag that changes an answer changes it the
same way in both.

The functions take an ``argparse.Namespace`` because the flags are the
configuration; both scripts declare the same flag names for the knobs used
here.
"""

import argparse
import copy
import os
import re
from typing import TYPE_CHECKING

from video_agent.media import Video
from video_agent.memory.artifacts import VideoArtifacts

if TYPE_CHECKING:  # heavy/optional modules: imported inside the functions
    from video_agent.agent.config import ReActConfig, ToolConfig
    from video_agent.agent.runner import AgentRunner
    from video_agent.agent.tools import ToolKit

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LOCAL_BASE_URL = "http://localhost:8000/v1"
"""Default of ``vllm serve`` (``serving/vllm/serve.sh``)."""

LOCAL_API_KEY_ENV = "VLLM_API_KEY"
LOCAL_API_KEY_FALLBACK = "EMPTY"
"""vLLM accepts any bearer token unless it was started with ``--api-key``."""

NO_THINKING_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}
"""Turn Qwen3.5's reasoning block off, per request.

Thinking is on by default in the chat template, and a ``<think>`` block
breaks both ends of the ReAct contract: the ``\\nObservation`` stop sequence
fires inside the reasoning and ``Final Answer:`` never gets parsed. The
server takes the switch in the request body, so it belongs to every request
this code makes and not to how the server was started.

``--thinking`` drops the switch, which is only safe against a server started
with ``--reasoning-parser qwen3``: the block then comes back in
``reasoning_content`` and ``content`` stays the ReAct text.
"""

DEFAULT_MAX_TOKENS = 16384
"""Per-turn token cap for the orchestrator, as a flag default.

Without a cap, a runaway turn can deliberate until the request times out.
A cap turns that into a truncated completion the loop can observe and
retry. 16k rather than 8k because a long but valid final answer can exceed
8k tokens.
"""

LOCAL_TIMEOUT_S = 300.0
"""Request timeout of the local backend, in seconds.

The SDK's 120 s default is too tight for a locally served model: at
roughly 90 tokens/s a full 16k-token turn takes about three minutes, so the
local timeout leaves room for one.
"""

REMOTE_TIMEOUT_S = 120.0
"""Request timeout everywhere else: the SDK default, unchanged."""

LOCAL_MAX_RETRIES = 1
"""Retries against the local server.

The SDK's default of 5 would turn one runaway turn into six consecutive
timeouts. One retry still covers a connection dropped in transit, which is
the failure retrying can actually fix.
"""

REMOTE_MAX_RETRIES = 5
"""Retries against a hosted provider: the SDK default, unchanged."""

VIDEO_BODY: dict = {"mm_processor_kwargs": {"do_sample_frames": False}}
"""Take a ``video_url``'s frames as sent, without resampling them.

vLLM's default is to resample a video to 2 fps, which of a 2-second clip
decoded at 5 fps would keep four frames and throw six away — the clip the
caption stage chose is not the clip the model would then see.
``do_sample_frames: false`` makes the processor take the frames as they
arrive; the server also has to keep them all on the way in, which is what
``--media-io-kwargs '{"video": {"num_frames": -1}}'`` in ``serve.sh`` does.

The other documented combination, ``{"fps": 2, "do_sample_frames": true}``,
is rejected with HTTP 400 by the vLLM build in ``serving/vllm`` ("Failed to
apply Qwen3VLProcessor"), so it is never sent from here.
"""


OBJECT_MEMORY_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
"""What ``--object-memory`` may say, checked before anything is loaded.

The slug is pasted straight into a file name and into a ``meta.json`` key
(``object_memory.<slug>.pkl``), so a value with a separator in it would read
a file outside the recording's directory and one with an upper-case letter
would name two files on case-insensitive and case-sensitive filesystems.
"""


def check_object_memory_slug(slug: str | None) -> None:
    """Raise ``ValueError`` unless ``slug`` can name an object memory.

    Shared by the three scripts that take the flag so all three refuse the
    same strings; each turns the exception into its own ``parser.error``,
    the way an unknown aligner id is refused.
    """
    if slug is not None and not OBJECT_MEMORY_SLUG.match(slug):
        raise ValueError(
            f"invalid object memory slug {slug!r}; slugs match "
            f"{OBJECT_MEMORY_SLUG.pattern} (e.g. rtdetr-v2-r50)"
        )


def request_body(*, thinking: bool = False, video: bool = False) -> dict:
    """The ``extra_body`` a request carries, assembled from the switches.

    Deep copies, so a caller owns the dict it is given and nothing
    downstream can edit the shared defaults it was built from.
    """
    body: dict = {}
    if not thinking:
        body.update(copy.deepcopy(NO_THINKING_BODY))
    if video:
        body.update(copy.deepcopy(VIDEO_BODY))
    return body


def local_api_key() -> str:
    """Bearer token for a local endpoint: ``$VLLM_API_KEY`` or ``"EMPTY"``.

    Passed explicitly rather than left to the adapter's environment lookup,
    which raises when the variable is unset — and unset is the normal case
    for a server started without ``--api-key``.
    """
    return os.environ.get(LOCAL_API_KEY_ENV) or LOCAL_API_KEY_FALLBACK


def resolve_base_url(args: argparse.Namespace) -> str | None:
    """The endpoint this run talks to, or ``None`` for the in-process backend.

    An explicit ``--base-url`` always wins (a remote vLLM, a proxy, another
    provider); otherwise the backend name picks its usual address.
    """
    if args.base_url:
        return args.base_url
    if args.chat == "vllm":
        return LOCAL_BASE_URL
    if args.chat == "transformers":
        return None
    return OPENROUTER_BASE_URL


def chat_timeout_s(args: argparse.Namespace) -> float:
    """Seconds one request may take, from ``--chat-timeout`` or the backend.

    The local server generates a whole turn before answering, so its ceiling
    is the token cap divided by the generation rate; a hosted provider keeps
    the SDK default.
    """
    if args.chat_timeout is not None:
        return float(args.chat_timeout)
    return LOCAL_TIMEOUT_S if args.chat == "vllm" else REMOTE_TIMEOUT_S


def chat_settings(args: argparse.Namespace) -> dict:
    """Keyword arguments for ``OpenAICompatibleChat``, as a plain dict.

    Pure, so a test can read the request shape a flag combination produces
    without a network or an API key. The ``transformers`` backend builds a
    different adapter and ignores this.
    """
    local = args.chat == "vllm"
    return {
        "model_name": args.model,
        "base_url": resolve_base_url(args),
        "api_key": local_api_key() if local else None,
        "api_key_env": LOCAL_API_KEY_ENV if local else "OPENROUTER_API_KEY",
        "temperature": args.temperature,
        "seed": args.seed,
        "timeout_s": chat_timeout_s(args),
        "max_retries": LOCAL_MAX_RETRIES if local else REMOTE_MAX_RETRIES,
        # The orchestrator sends text only, so no video switch here.
        "extra_body": (
            request_body(thinking=args.thinking, video=False) if local else None
        ),
    }


def vqa_chat_settings(args: argparse.Namespace) -> dict | None:
    """Keyword arguments of the separate VQA endpoint, or ``None`` without one.

    ``--vqa-base-url`` points the visual questions at a local vLLM server
    while the orchestrator talks to another endpoint, so the request has the
    local server's shape: its bearer token, thinking off, local timeout and
    retries.
    """
    base_url = getattr(args, "vqa_base_url", None)
    if not base_url:
        return None
    return {
        "model_name": args.vqa_api_model,
        "base_url": base_url,
        "api_key": local_api_key(),
        "api_key_env": LOCAL_API_KEY_ENV,
        "temperature": args.temperature,
        "seed": args.seed,
        "timeout_s": LOCAL_TIMEOUT_S,
        "max_retries": LOCAL_MAX_RETRIES,
        "extra_body": request_body(thinking=False, video=False),
    }


class LazyModels:
    """Load each model on first use so unused backends cost nothing.

    A run with ``--vqa none --no-objects`` never touches the GPU, and the
    chat backend is shared between the main agent and the sub-agent.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._chat = None
        self._text_embedder = None
        self._aligner = None
        self._siglip = None
        self._vqa = None

    @property
    def chat(self):
        if self._chat is None:
            if self.args.chat == "transformers":
                from video_agent.adapters.chat_transformers import TransformersChat

                self._chat = TransformersChat(
                    self.args.model,
                    temperature=self.args.temperature,
                    device_map=self.args.device,
                )
            else:
                from video_agent.adapters.chat_openai import OpenAICompatibleChat

                self._chat = OpenAICompatibleChat(**chat_settings(self.args))
        return self._chat

    @property
    def text_embedder(self):
        if self._text_embedder is None:
            from video_agent.adapters.text_embedder import SentenceTransformerEmbedder

            self._text_embedder = SentenceTransformerEmbedder(
                self.args.text_embedder_model, device=self.args.device
            )
        return self._text_embedder

    @property
    def aligner(self):
        if self._aligner is None:
            from video_agent.adapters.aligner import build_aligner

            # Asked for through the tool configuration rather than off the
            # flag, so the tower that runs is the one the run record names.
            self._aligner = build_aligner(
                tool_config(self.args).aligner_model, device=self.args.device
            )
        return self._aligner

    @property
    def siglip(self):
        if self._siglip is None:
            from video_agent.adapters.image_embedder import SiglipImageEmbedder

            self._siglip = SiglipImageEmbedder(
                self.args.siglip_model, device=self.args.device
            )
        return self._siglip

    @property
    def vqa(self):
        if self._vqa is None and self.args.vqa != "none":
            if self.args.vqa == "api":
                # Only the in-process backend is text-only; the served
                # backends (openrouter, vllm) answer about frames.
                if self.args.chat == "transformers":
                    raise ValueError(
                        "--vqa api needs a multimodal chat backend, but "
                        f"--chat {self.args.chat} is text-only; use "
                        "--chat vllm or --chat openrouter"
                    )
                from video_agent.adapters.vqa_openai import (
                    GPT4V_PROMPT,
                    OpenAIVisionVideoQA,
                )

                # The reference prompt asks for the description and the
                # answer at once, which only makes sense when the tool asks
                # a single question.
                separate = vqa_chat_settings(self.args)
                if separate is None:
                    vqa_chat = self.chat
                else:
                    from video_agent.adapters.chat_openai import OpenAICompatibleChat

                    vqa_chat = OpenAICompatibleChat(**separate)
                self._vqa = OpenAIVisionVideoQA(
                    vqa_chat,
                    question_template=(
                        GPT4V_PROMPT if self.args.vqa_style == "single" else None
                    ),
                )
            else:
                from video_agent.adapters.vqa_qwen import Qwen3VLVideoQA

                self._vqa = Qwen3VLVideoQA(
                    self.args.vqa_model, device_map=self.args.device
                )
        return self._vqa


def tool_config(args: argparse.Namespace) -> "ToolConfig":
    """Tool settings for this run.

    Everything but the VQA style, the aligner and the object memory keeps its
    reference default; the transcript records the whole dataclass rather than
    the flags, so the record is self-describing whichever flags a script
    declares.

    ``--object-memory`` is read with ``getattr`` because not every script
    that builds a toolkit declares it, and a script that does not is asking
    for the memory the reproduction wrote — which is what ``None`` means.
    """
    from video_agent.agent.config import ToolConfig

    return ToolConfig(
        vqa_style=args.vqa_style,
        aligner_model=args.aligner_model,
        object_memory_slug=getattr(args, "object_memory", None),
    )


def react_config(args: argparse.Namespace) -> "ReActConfig":
    """ReAct loop settings for this run.

    ``--max-tokens 0`` (or a negative value) means unlimited: the cap becomes
    ``None`` and the adapter omits ``max_tokens`` from the request.
    """
    from video_agent.agent.config import ReActConfig

    return ReActConfig(
        max_iterations=args.max_iterations,
        max_tokens=args.max_tokens if args.max_tokens > 0 else None,
    )


def memory_mode(args: argparse.Namespace) -> str:
    """Which memory this run gives the agent (``--memory``), ``full`` by default.

    Read with ``getattr`` for the reason ``tool_config`` reads
    ``--object-memory`` that way: only the batch runner of the task declares
    the flag, and a script that does not declare it is asking for the
    recording's own memory.
    """
    from video_agent.memory.ablation import MEMORY_FULL

    return getattr(args, "memory", MEMORY_FULL)


def build_withheld_toolkit(
    video: Video,
    args: argparse.Namespace,
    runner: "AgentRunner",
) -> "ToolKit":
    """The toolkit of a recording whose memory is withheld (``--memory empty``).

    The empty-memory control: the agent keeps the recipe, the task graph,
    the question, the four tools and their descriptions, and every tool
    answers ``MEMORY UNAVAILABLE``. Nothing is read from the artifacts
    directory and no embedder, no VQA model and no object database is built —
    the condition is *no memory*, and a run that quietly loaded one to build
    a toolkit around it would be paying for a memory it must not use.

    The grid still comes from the video itself, so the prompt tells the agent
    how many segments the recording has: the control withholds the memory,
    not the length of the video, and the record's verdict anchors are
    read on that grid like any other run's.
    """
    from video_agent.agent.prompts import load_prompt
    from video_agent.agent.tools import ToolKit
    from video_agent.memory.ablation import WithheldVideoQA, withheld_temporal_memory
    from video_agent.memory.preprocess import get_video_segments
    from video_agent.memory.types import ObjectMemory

    with_objects = not args.no_objects
    return ToolKit(
        video=video,
        temporal=withheld_temporal_memory(get_video_segments(video)),
        text_embedder=None,
        aligner=None,
        # Stand-ins: which tools are listed is the run's ablation, and it has
        # to be the same list the full run offered (``memory.ablation``).
        objects=ObjectMemory(records=[]) if with_objects else None,
        vqa=WithheldVideoQA() if args.vqa != "none" else None,
        sub_runner=runner if with_objects else None,
        sub_prompt=load_prompt("database_query_prompt") if with_objects else None,
        config=tool_config(args),
        memory_available=False,
    )


def build_toolkit(
    video: Video,
    artifacts: VideoArtifacts,
    models: LazyModels,
    args: argparse.Namespace,
    runner: "AgentRunner",
    ordinal: int | None = None,
) -> "ToolKit":
    """Wire the memories of one video into the agent's tools.

    ``runner`` doubles as the object sub-agent's runner, as in the reference,
    where the nested executor used the same LLM and loop settings.

    ``ordinal`` is the recording's position in its split and is only read
    under ``--memory shuffled``, where it seeds that recording's permutation
    (``memory.ablation``); the toolkit carries the permutation it was built
    over so the caller can record it.
    """
    from video_agent.adapters.aligner import aligner_slug
    from video_agent.agent.prompts import load_prompt
    from video_agent.agent.tools import ToolKit
    from video_agent.memory.ablation import (
        MEMORY_EMPTY,
        MEMORY_SHUFFLED,
        segment_permutation,
        shuffle_memories,
    )
    from video_agent.memory.object_db import text_columns_for
    from video_agent.memory.store import (
        crop_embedding_widths,
        load_object_memory,
        load_temporal_memory,
    )

    mode = memory_mode(args)
    if mode == MEMORY_EMPTY:
        return build_withheld_toolkit(video, args, runner)
    if mode == MEMORY_SHUFFLED and ordinal is None:
        raise ValueError(
            "--memory shuffled needs the recording's position in the split "
            "to seed its permutation"
        )

    config = tool_config(args)
    # The visual channel is only meaningful against its own aligner's
    # embeddings, so the memory is read from the file that aligner wrote.
    temporal = load_temporal_memory(
        artifacts,
        args.caption_variant,
        video,
        aligner_slug=aligner_slug(config.aligner_model),
    )
    objects = None
    object_text_embedder = None
    object_text_columns = None
    sub_prompt = None
    if not args.no_objects:
        # One recording can hold several object memories; the slug
        # names both the memory and the embed stage whose widths say which
        # columns of a feature are text-aligned, so both reads take it.
        slug = config.object_memory_slug
        objects = load_object_memory(artifacts, slug)
        object_text_embedder = models.siglip
        object_text_columns = text_columns_for(
            crop_embedding_widths(artifacts, slug)
        )
        sub_prompt = load_prompt("database_query_prompt")
    permutation = None
    video_segments = None
    if mode == MEMORY_SHUFFLED:
        # A view over what was just loaded: the same captions, vectors and
        # objects, under permuted names. Nothing on disk is rewritten.
        shuffled = shuffle_memories(
            temporal, objects, segment_permutation(len(temporal.segments), ordinal)
        )
        temporal, objects = shuffled.temporal, shuffled.objects
        permutation = shuffled.permutation
        video_segments = shuffled.video_segments
    return ToolKit(
        video=video,
        temporal=temporal,
        text_embedder=models.text_embedder,
        aligner=models.aligner,
        objects=objects,
        object_text_embedder=object_text_embedder,
        object_text_columns=object_text_columns,
        vqa=models.vqa,
        sub_runner=runner if objects is not None else None,
        sub_prompt=sub_prompt,
        config=config,
        permutation=permutation,
        video_segments=video_segments,
    )

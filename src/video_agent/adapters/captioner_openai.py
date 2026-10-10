"""Clip captioning through any multimodal chat endpoint (vLLM, OpenRouter).

The reference captioned every segment with a local narrator (LaViLa); this
adapter puts the same job behind the chat API, so one served checkpoint can
be orchestrator, VQA and captioner at once. One chat completion per clip,
the clip's frames inlined in the request — as one video by default, or as
individual images the way ``adapters.vqa_openai`` sends them.

Deviation from ``adapters.captioner.Qwen3VLCaptioner``: no sampling
mechanics. ``do_sample``/``num_candidates`` ("draw 5 candidates, keep the
longest") is a local-generation reproduction of the reference narrator that
an OpenAI-compatible endpoint does not offer per request; temperature and
seed belong to the chat object, which is where a run's sampling settings are
already recorded.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from video_agent.adapters.captioner import DEFAULT_PROMPT, resolve_prompts
from video_agent.adapters.frames import evenly_spaced
from video_agent.media import Clip, Image
from video_agent.models.interfaces import ChatModel
from video_agent.models.types import ChatMessage, Usage

TRANSPORTS = ("images", "video")
"""How a clip's frames travel to the endpoint.

``video`` sends them as one video input, ``images`` as one still per frame.
"""


class OpenAIVisionCaptioner:
    """Captioner over any multimodal ``ChatModel``: one completion per clip.

    ``frames_per_clip`` caps the frames sent per clip (evenly spaced);
    ``None`` sends every frame the clip carries. ``workers`` > 1 issues the
    per-clip requests concurrently — a served engine batches whatever arrives
    together, so throughput comes from there and not from a bigger prompt.

    ``transport`` defaults to ``"video"`` for parity with the local
    ``Qwen3VLCaptioner``, which hands the frames to ``processor(videos=...)``
    and gets temporal patch merging out of it: on Qwen3-VL, ten 360p frames
    sent as one video take several times fewer prompt tokens than the same
    frames sent as ten stills. ``"images"`` is the fallback for an endpoint
    whose loader will not take a video ``data:`` URL.

    ``usage`` accumulates over every call. It is deliberately not part of the
    ``Captioner`` protocol: the memory-construction code does not read it,
    only the batch script that records what captioning cost.
    """

    def __init__(
        self,
        chat: ChatModel,
        *,
        prompt: str = DEFAULT_PROMPT,
        frames_per_clip: int | None = None,
        max_tokens: int = 64,
        workers: int = 1,
        transport: str = "video",
    ) -> None:
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        if transport not in TRANSPORTS:
            raise ValueError(
                f"transport must be one of {TRANSPORTS}, got {transport!r}"
            )
        self.chat = chat
        self.prompt = prompt
        self.frames_per_clip = frames_per_clip
        self.max_tokens = max_tokens
        self.workers = workers
        self.transport = transport
        self.usage = Usage()
        self._lock = threading.Lock()

    def _clip_images(self, clip: Clip) -> tuple[Image, ...]:
        frames = clip.frames
        if self.frames_per_clip is not None:
            frames = tuple(evenly_spaced(frames, self.frames_per_clip))
        return tuple(frame.data for frame in frames)

    def _message(self, clip: Clip, prompt: str) -> ChatMessage:
        """The one turn a clip becomes, in whichever transport was chosen."""
        frames = self._clip_images(clip)
        if self.transport == "video":
            return ChatMessage(role="user", text=prompt, video=frames)
        return ChatMessage(role="user", text=prompt, images=frames)

    def _caption_one(self, clip: Clip, prompt: str) -> str:
        """Caption one clip and bill it; called from the worker threads."""
        message = self._message(clip, prompt)
        reply = self.chat.complete([message], max_tokens=self.max_tokens)
        with self._lock:
            self.usage = self.usage + reply.usage
        # The local captioner strips too; an empty reply stays empty, since
        # the chat adapter already retried it once.
        return reply.text.strip()

    def caption(
        self, clips: Sequence[Clip], prompts: Sequence[str] | None = None
    ) -> list[str]:
        """Return one caption per clip, aligned by index.

        ``prompts`` is one instruction per clip — what a run with context
        segments needs, since the question depends on where the described
        segment sits in the clip — and ``None`` sends ``self.prompt`` to all
        of them.
        """
        texts = resolve_prompts(self.prompt, clips, prompts)
        if not clips:
            return []
        if self.workers == 1:
            return [
                self._caption_one(clip, text) for clip, text in zip(clips, texts)
            ]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            # ``map`` yields in submission order, so completion order cannot
            # shuffle the captions away from their clips.
            return list(pool.map(self._caption_one, clips, texts))

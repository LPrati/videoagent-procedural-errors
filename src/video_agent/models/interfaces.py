"""Protocols for the pluggable models behind memory construction and inference.

All methods are batched: inputs map to outputs by index, and implementations
own their internal chunking and hardware usage. In the listings below,
"reference" marks the model used by VideoAgent (see ``memory.preprocess``).
"""

from typing import Protocol, Sequence

import numpy as np

from video_agent.media import Clip, Frame, Image
from video_agent.models.types import ChatMessage, ChatReply, Detection, Tracklet


class Captioner(Protocol):
    """Describes short clips in natural language.

    Implementations: LaViLa (reference), Video-LLaVA, BLIP-2.
    """

    def caption(
        self, clips: Sequence[Clip], prompts: Sequence[str] | None = None
    ) -> list[str]:
        """Return one caption per clip, aligned by index.

        ``prompts`` gives one instruction per clip and defaults to ``None``,
        which asks the implementation's own prompt of every clip. It exists
        because a clip carrying its neighbours has to say which part of
        itself the sentence is about (``memory.temporal``), and an
        implementation that only ever asks one question may ignore it.
        """
        ...


class TextEmbedder(Protocol):
    """Embeds text into a single vector space.

    The same model must embed captions at build time and queries at
    inference time. Implementations: SentenceTransformers, OpenAI embeddings.
    """

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return embeddings of shape (len(texts), D)."""
        ...


class VideoTextAligner(Protocol):
    """Contrastive dual encoder mapping video and text to a shared space.

    Implementations: ViCLIP (reference), X-CLIP, PE-Core and SigLIP2 frame
    means — see ``adapters.aligner``, whose registry maps a model id to one
    of them and to the artifact suffix its embeddings are stored under.
    """

    def encode_video(self, clips: Sequence[Clip]) -> np.ndarray:
        """Return clip embeddings of shape (len(clips), D)."""
        ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return text embeddings of shape (len(texts), D), same space as videos."""
        ...


class ObjectDetector(Protocol):
    """Detects objects independently per frame.

    Implementations: RT-DETR (reference), YOLO, Grounding DINO.
    """

    def detect(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        """Return detections per frame, aligned with ``frames``."""
        ...


class Tracker(Protocol):
    """Associates per-frame detections into tracklets.

    Frames are provided for appearance-based trackers (e.g. BoT-SORT);
    motion-only trackers such as ByteTrack (reference) may ignore them.
    """

    def track(
        self, frames: Sequence[Frame], detections: Sequence[Sequence[Detection]]
    ) -> list[Tracklet]:
        """Return tracklets associated from per-frame detections."""
        ...


class ImageEmbedder(Protocol):
    """Embeds image crops for re-identification.

    Implementations: CLIP + DINOv2 ensemble (reference), SigLIP2, DINOv2.
    Only text-aligned embedders support querying objects by name at
    inference time.
    """

    def embed(self, images: Sequence[Image]) -> np.ndarray:
        """Return embeddings of shape (len(images), D)."""
        ...


class TextAlignedImageEmbedder(ImageEmbedder, Protocol):
    """Image embedder whose text tower shares the crop space.

    Needed by the object memory's open-vocabulary retrieval (reference: CLIP
    ViT-B/32 text of ``"a photo of a {description}."``). SigLIP2 satisfies
    it; DINOv2 does not.
    """

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return text embeddings of shape (len(texts), D), same space as crops."""
        ...


class ReIDMerger(Protocol):
    """Merges tracklets of the same physical object under a global object id."""

    def merge(
        self, tracklets: Sequence[Tracklet], embeddings: np.ndarray
    ) -> dict[int, int]:
        """Return a ``Tracklet.track_id -> object_id`` mapping."""
        ...


class ChatModel(Protocol):
    """Chat completion for the orchestrator and its sub-agents.

    Reference: GPT-4 in the paper, ``gpt-4o-2024-05-13`` in the released
    benchmark code. Sampling settings (temperature, seed) are adapter
    configuration so a run is reproducible from its config alone.
    Implementations: OpenAI-compatible HTTP APIs (OpenRouter, vLLM), local
    ``transformers`` checkpoints.
    """

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] = (),
        max_tokens: int | None = None,
    ) -> ChatReply:
        """Return the completion for ``messages``, truncated at any ``stop`` string."""
        ...


class VideoQA(Protocol):
    """Answers free-form questions about short clips.

    Reference: Video-LLaVA-7B served in a separate process; the released code
    also offers ``gpt-4-vision-preview`` over four frames. Implementations:
    Qwen3-VL (local), any multimodal ``ChatModel`` over sampled frames.
    """

    def answer(self, clips: Sequence[Clip], questions: Sequence[str]) -> list[str]:
        """Return one answer per ``(clip, question)`` pair, aligned by index."""
        ...

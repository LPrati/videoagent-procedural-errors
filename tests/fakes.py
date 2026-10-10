"""Duck-typed stand-ins for the models and memories the agent layer needs.

Every fake is deterministic and answers from arithmetic, so a test can state
the exact string or ranking it expects. The embedding fakes agree on one
convention: text embeds to the one-hot of the first integer it mentions, and
``make_temporal_memory`` builds row ``i`` as the one-hot of ``i``, so
"segment 2" is the query that retrieves segment 2.
"""

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from video_agent.media import Clip, Frame, Image
from video_agent.memory.types import (
    ObjectMemory,
    ObjectRecord,
    Segment,
    TemporalMemory,
)
from video_agent.models.types import ChatMessage, ChatReply, Detection, Usage


def _one_hot(index: int, dim: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index % dim] = 1.0
    return vector


def _first_integer(text: str) -> int:
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0


@dataclass(frozen=True)
class ScriptedReply:
    """A scripted completion carrying what a bare string cannot.

    Most scripts only care about the text, so ``FakeChat`` takes strings; a
    test about the thinking block the server split out, or about a turn the
    token cap truncated, scripts one of these instead.
    """

    text: str
    reasoning: str | None = None
    finish_reason: str = "stop"


class FakeChat:
    """Chat model replaying a script, recording everything it was asked.

    Replies are strings, or ``ScriptedReply`` when a turn has to carry a
    reasoning block or a finish reason of its own. Running out of replies is
    a test bug, not a model behavior, so it fails loudly instead of returning
    something the agent would act on.
    """

    def __init__(
        self, replies: list[str | ScriptedReply], *, model: str = "fake-chat"
    ) -> None:
        self.replies = list(replies)
        self.model = model
        self.messages: list[list[ChatMessage]] = []
        self.prompts: list[str] = []
        self.stops: list[tuple[str, ...]] = []
        self.max_tokens: list[int | None] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] = (),
        max_tokens: int | None = None,
    ) -> ChatReply:
        if not self.replies:
            raise AssertionError("FakeChat ran out of scripted replies")
        self.messages.append(list(messages))
        self.prompts.append(messages[-1].text)
        self.stops.append(tuple(stop))
        self.max_tokens.append(max_tokens)
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            reply = ScriptedReply(reply)
        return ChatReply(
            text=reply.text,
            usage=Usage(llm_calls=1, tokens_in=1, tokens_out=1),
            model=self.model,
            finish_reason=reply.finish_reason,
            reasoning=reply.reasoning,
        )


class RecordingChat:
    """Chat model replaying canned replies and recording every call verbatim.

    ``FakeChat``'s sibling for the adapters: it keeps the whole call —
    messages, stop strings, token budget — because what an adapter *sent* is
    exactly what those tests are about.
    """

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[ChatMessage], tuple[str, ...], int | None]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] = (),
        max_tokens: int | None = None,
    ) -> ChatReply:
        self.calls.append((list(messages), tuple(stop), max_tokens))
        if not self.replies:
            raise AssertionError("more calls than canned replies")
        return ChatReply(
            text=self.replies.pop(0), usage=Usage(llm_calls=1), model="fake"
        )


class FakeTextEmbedder:
    """Caption embedder: one-hot of the first integer in the text."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.texts: list[str] = []

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.texts.extend(texts)
        return np.stack([_one_hot(_first_integer(t), self.dim) for t in texts])


class FakeAligner:
    """Video/text aligner sharing ``FakeTextEmbedder``'s one-hot convention."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.texts: list[str] = []

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        self.texts.extend(texts)
        return np.stack([_one_hot(_first_integer(t), self.dim) for t in texts])

    def encode_video(self, clips: Sequence[Clip]) -> np.ndarray:
        return np.stack([_one_hot(i, self.dim) for i in range(len(clips))])


class FakeSiglip:
    """Text-aligned crop embedder over a fixed keyword axis.

    ``embed_text`` returns the one-hot of the first keyword the text mentions
    (the first axis when it mentions none), which lets a test place object
    features at chosen angles from a query and check the threshold.
    """

    def __init__(self, keywords: Sequence[str] = ("cup", "dog")) -> None:
        self.keywords = tuple(keywords)
        self.embedding_dim = len(self.keywords)
        self.texts: list[str] = []

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        self.texts.extend(texts)
        return np.stack([self._vector(text) for text in texts])

    def embed(self, images: Sequence[Image]) -> np.ndarray:
        return np.stack(
            [_one_hot(int(np.asarray(i).mean()), self.embedding_dim) for i in images]
        )

    def _vector(self, text: str) -> np.ndarray:
        for index, keyword in enumerate(self.keywords):
            if keyword in text:
                return _one_hot(index, self.embedding_dim)
        return _one_hot(0, self.embedding_dim)


class FakeVideoQA:
    """Video QA echoing each question and the clip it was asked about.

    The clips are kept so a test can assert which frames were decoded.
    """

    def __init__(self) -> None:
        self.clips: list[Clip] = []
        self.questions: list[str] = []

    def answer(self, clips: Sequence[Clip], questions: Sequence[str]) -> list[str]:
        if len(clips) != len(questions):
            raise ValueError(f"{len(clips)} clips but {len(questions)} questions")
        self.clips.extend(clips)
        self.questions.extend(questions)
        return [
            f"answer to '{question}' over {len(clip.frames)} frames"
            for clip, question in zip(clips, questions)
        ]




class FakeDetector:
    """Object detector answering one box per frame, cycling through categories.

    Deterministic on the frame index so a test can state which category a
    frame got: the detect stage saves detections keyed by frame, and a stage
    that dropped or reordered a window would otherwise look correct.

    ``score_threshold`` and ``batch_size`` are the attributes the stage reads
    off a detector, and ``lineage`` is the dict it copies into ``meta.json``.
    """

    def __init__(
        self,
        categories: Sequence[str] = ("cup",),
        *,
        box: tuple[float, float, float, float] = (1.0, 1.0, 20.0, 20.0),
    ) -> None:
        self.categories = tuple(categories)
        self.box = tuple(float(v) for v in box)
        self.score_threshold = 0.5
        self.batch_size = 1
        self.seen: list[int] = []

    def detect(self, frames: Sequence[Frame]) -> list[list[Detection]]:
        self.seen.extend(frame.index for frame in frames)
        return [[self._detection(frame)] for frame in frames]

    def _detection(self, frame: Frame) -> Detection:
        return Detection(
            frame_index=frame.index,
            box_xyxy=self.box,
            category=self.categories[frame.index % len(self.categories)],
            score=0.9,
        )

    def lineage(self) -> dict:
        return {
            "checkpoint": "fake-detector",
            "hf_snapshot": None,
            "score_threshold": self.score_threshold,
            "dtype": "float32",
            "batch_size": self.batch_size,
        }


class FakeCropEmbedder:
    """Concatenated crop embedder: one one-hot per member, from the crop's mean.

    Mirrors ``ConcatImageEmbedder``'s layout — ``widths`` says where one
    member's columns end and the next's begin — because that is what the
    embed stage records and what re-ID slices the feature by. Each block is
    a unit vector, so two crops of the same brightness are at cosine 1 and
    the merger has a real decision to make.
    """

    def __init__(self, widths: Sequence[int] = (2, 2)) -> None:
        self.widths = tuple(int(width) for width in widths)
        self.embedding_dim = sum(self.widths)

    def embed(self, images: Sequence[Image]) -> np.ndarray:
        return np.stack([self._vector(image) for image in images])

    def _vector(self, image: Image) -> np.ndarray:
        value = int(np.asarray(image, dtype=np.float64).mean())
        blocks = []
        for width in self.widths:
            block = np.zeros(width, dtype=np.float32)
            block[value % width] = 1.0
            blocks.append(block)
        return np.concatenate(blocks)



def make_temporal_memory(n: int, dim: int = 4) -> TemporalMemory:
    """Build an ``n``-segment memory on the 2-second, 30 fps grid.

    Both channels hold the one-hot of the segment index (wrapping when
    ``n > dim``) and captions are ``"caption i"``, so retrieval is
    predictable without a model.
    """
    return TemporalMemory(
        segments=[
            Segment(
                index=i,
                start_s=2.0 * i,
                end_s=2.0 * i + 2.0,
                frame_range=(60 * i, 60 * i + 60),
            )
            for i in range(n)
        ],
        captions=[f"caption {i}" for i in range(n)],
        caption_embeddings=np.stack([_one_hot(i, dim) for i in range(n)]),
        visual_embeddings=np.stack([_one_hot(i, dim) for i in range(n)]),
    )


def object_record(
    object_id: int, category: str, segments: Sequence[int], feature: Sequence[float]
) -> ObjectRecord:
    """Build one record; tracklets stay empty since no query reads them."""
    return ObjectRecord(
        object_id=object_id,
        category=category,
        segment_indices=tuple(segments),
        feature=np.asarray(feature, dtype=np.float32),
        tracklets=(),
    )


def make_object_memory(*records: ObjectRecord) -> ObjectMemory:
    return ObjectMemory(records=list(records))

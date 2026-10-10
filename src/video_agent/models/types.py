"""Data types exchanged with perception models (see ``interfaces``)."""

from dataclasses import asdict, dataclass

from video_agent.media import Image


@dataclass(frozen=True)
class Detection:
    """One detected object in a single frame."""

    frame_index: int  # Frame.index in the source video
    box_xyxy: tuple[float, float, float, float]
    category: str
    score: float


@dataclass(frozen=True)
class Tracklet:
    """One object instance tracked over consecutive frames."""

    track_id: int
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class ChatMessage:
    """One turn of a chat conversation.

    Frames ride along for multimodal models in one of two shapes, and the
    difference is not cosmetic: ``images`` are independent stills, while
    ``video`` is the frame sequence of *one* clip, to be sent as a single
    video input — which is what buys temporal patch merging, and with it a
    prompt several times shorter than the same frames as stills.

    Text-only adapters must reject either kind rather than silently drop it.
    A message may carry both at once only if an adapter documents that it
    sends both; ours never need to.
    """

    role: str  # "system" | "user" | "assistant"
    text: str
    images: tuple[Image, ...] = ()
    video: tuple[Image, ...] = ()  # frames of one clip, sent as one video


@dataclass
class Usage:
    """Cost counters for one or more model calls.

    Adapters fill ``llm_calls``/``tokens_*`` per call; agent runners add
    ``tool_calls``, error counts and wall time while they accumulate.
    """

    llm_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    parse_errors: int = 0
    tool_errors: int = 0
    wall_s: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            llm_calls=self.llm_calls + other.llm_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            parse_errors=self.parse_errors + other.parse_errors,
            tool_errors=self.tool_errors + other.tool_errors,
            wall_s=self.wall_s + other.wall_s,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Usage":
        return cls(**data)


@dataclass(frozen=True)
class ChatReply:
    """A chat completion and what it cost.

    ``reasoning`` is the thinking block a server with a reasoning parser
    returns beside the answer. It is never folded into ``text`` — the ReAct
    scratchpad concatenates ``text`` and would drown in it — but it is kept
    so a transcript can show what the model deliberated. Its tokens are
    already inside ``usage.tokens_out``, which is what the server billed.
    """

    text: str
    usage: Usage
    model: str  # model id as reported by the provider (snapshot, not alias)
    finish_reason: str | None = None
    reasoning: str | None = None  # thinking block, when the server split it out
    raw: dict | None = None  # provider payload, kept for transcripts

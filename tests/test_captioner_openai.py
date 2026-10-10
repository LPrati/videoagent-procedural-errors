"""Clip captioning over a chat model: one call per clip, in clip order.

The transport is what the message carries: ``video`` puts the clip's frames
in ``ChatMessage.video`` (one video input, temporal patch merging on the
server), ``images`` puts them in ``ChatMessage.images`` (one still each).
Only the field differs here; turning either into a request part is the chat
adapter's job (``test_chat_openai``).
"""

import threading
from typing import Sequence

import numpy as np
import pytest
from tests.fakes import RecordingChat

from video_agent.adapters.captioner import DEFAULT_PROMPT
from video_agent.adapters.captioner_openai import OpenAIVisionCaptioner
from video_agent.media import Clip, Frame
from video_agent.models.types import ChatMessage, ChatReply, Usage

STATE_PROMPT = "Describe the action and the state of the ingredients."


def make_clip(first: int, frame_count: int = 10) -> Clip:
    """Clip whose frames carry ``first + offset`` in every pixel.

    The first pixel of the first frame identifies the clip, which is what
    lets a test see a caption that came back on the wrong one.
    """
    return Clip(
        frames=tuple(
            Frame(
                index=index,
                time_s=index / 30,
                data=np.full((4, 4, 3), first + index, dtype=np.uint8),
            )
            for index in range(frame_count)
        )
    )


class EchoingChat:
    """Chat model whose reply names the clip it saw, with sloppy whitespace.

    Replies are derived from the first pixel of the first frame — whichever
    transport carried it — so a misalignment between clips and captions is
    visible in the output rather than only in the call log. ``tokens_out``
    counts the frames, so a test can check that every call was billed, and
    the calling thread is recorded so a test can see whether a pool was used
    at all.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[ChatMessage], tuple[str, ...], int | None]] = []
        self.threads: list[str] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] = (),
        max_tokens: int | None = None,
    ) -> ChatReply:
        self.calls.append((list(messages), tuple(stop), max_tokens))
        self.threads.append(threading.current_thread().name)
        frames = messages[0].images or messages[0].video
        return ChatReply(
            text=f"  clip {int(frames[0][0, 0, 0])}\n",
            usage=Usage(llm_calls=1, tokens_in=3, tokens_out=len(frames)),
            model="fake",
        )


def test_sends_one_completion_per_clip_with_the_prompt_as_the_text() -> None:
    chat = RecordingChat("a person stirs a pot", "a person washes a bowl")
    captioner = OpenAIVisionCaptioner(chat, prompt=STATE_PROMPT)

    captions = captioner.caption([make_clip(0), make_clip(100)])

    assert captions == ["a person stirs a pot", "a person washes a bowl"]
    assert len(chat.calls) == 2
    for messages, stop, max_tokens in chat.calls:
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].text == STATE_PROMPT
        assert stop == ()
        assert max_tokens == 64


def test_the_default_prompt_is_the_adapters_default_instruction() -> None:
    chat = RecordingChat("a caption")

    OpenAIVisionCaptioner(chat).caption([make_clip(0)])

    assert chat.calls[0][0][0].text == DEFAULT_PROMPT


class TestTransport:
    """Which field the clip lands in, and which frames of it."""

    def test_a_clip_travels_as_one_video_by_default(self) -> None:
        chat = RecordingChat("a caption")

        OpenAIVisionCaptioner(chat).caption([make_clip(0, frame_count=10)])

        message = chat.calls[0][0][0]
        assert message.images == ()
        assert [int(frame[0, 0, 0]) for frame in message.video] == list(range(10))

    def test_the_frame_budget_applies_to_the_video_transport_too(self) -> None:
        chat = RecordingChat("a caption")

        OpenAIVisionCaptioner(chat, frames_per_clip=4).caption(
            [make_clip(0, frame_count=10)]
        )

        message = chat.calls[0][0][0]
        assert [int(frame[0, 0, 0]) for frame in message.video] == [0, 2, 5, 7]

    def test_the_images_transport_sends_one_still_per_frame(self) -> None:
        chat = RecordingChat("a caption")

        OpenAIVisionCaptioner(chat, transport="images").caption(
            [make_clip(0, frame_count=10)]
        )

        message = chat.calls[0][0][0]
        assert message.video == ()
        assert [int(image[0, 0, 0]) for image in message.images] == list(range(10))

    def test_the_frame_budget_applies_to_the_images_transport_too(self) -> None:
        chat = RecordingChat("a caption")

        OpenAIVisionCaptioner(chat, transport="images", frames_per_clip=4).caption(
            [make_clip(0, frame_count=10)]
        )

        images = chat.calls[0][0][0].images
        assert [int(image[0, 0, 0]) for image in images] == [0, 2, 5, 7]

    def test_an_unknown_transport_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError):
            OpenAIVisionCaptioner(RecordingChat(), transport="frames")


def test_forwards_the_token_budget() -> None:
    chat = RecordingChat("a caption")

    OpenAIVisionCaptioner(chat, max_tokens=200).caption([make_clip(0)])

    assert chat.calls[0][2] == 200


def test_captions_come_back_stripped() -> None:
    chat = RecordingChat("  a person stirs a pot \n")

    assert OpenAIVisionCaptioner(chat).caption([make_clip(0)]) == [
        "a person stirs a pot"
    ]


def test_an_empty_reply_stays_empty() -> None:
    chat = RecordingChat("   ")

    assert OpenAIVisionCaptioner(chat).caption([make_clip(0)]) == [""]
    assert len(chat.calls) == 1  # no retry here: the chat adapter already did


def test_captions_stay_aligned_with_their_clips_in_one_thread() -> None:
    chat = EchoingChat()
    clips = [make_clip(first) for first in (0, 40, 80, 120, 160)]

    captions = OpenAIVisionCaptioner(chat).caption(clips)

    assert captions == [f"clip {first}" for first in (0, 40, 80, 120, 160)]


def test_captions_stay_aligned_when_the_requests_run_concurrently() -> None:
    chat = EchoingChat()
    clips = [make_clip(first) for first in (0, 40, 80, 120, 160)]

    captions = OpenAIVisionCaptioner(chat, workers=4).caption(clips)

    assert captions == [f"clip {first}" for first in (0, 40, 80, 120, 160)]
    assert len(chat.calls) == 5


def test_a_single_worker_never_builds_a_thread_pool() -> None:
    chat = EchoingChat()
    main_thread = threading.current_thread().name

    OpenAIVisionCaptioner(chat).caption([make_clip(0), make_clip(40)])

    assert chat.threads == [main_thread, main_thread]


def test_several_workers_do_the_requests_off_the_calling_thread() -> None:
    chat = EchoingChat()
    main_thread = threading.current_thread().name

    OpenAIVisionCaptioner(chat, workers=4).caption(
        [make_clip(20 * i) for i in range(8)]
    )

    # A pool spawns its threads lazily, so a fake this fast may well be
    # served by one of them; what matters is that it is not this one.
    assert main_thread not in chat.threads
    assert len(chat.threads) == 8


def test_usage_accumulates_over_every_call() -> None:
    chat = EchoingChat()
    captioner = OpenAIVisionCaptioner(chat, frames_per_clip=4)

    captioner.caption([make_clip(0), make_clip(40), make_clip(80)])

    assert captioner.usage.llm_calls == 3
    assert captioner.usage.tokens_in == 9
    assert captioner.usage.tokens_out == 12  # 4 frames billed per call


def test_usage_accumulates_across_batches_and_workers() -> None:
    chat = EchoingChat()
    captioner = OpenAIVisionCaptioner(chat, frames_per_clip=2, workers=4)

    captioner.caption([make_clip(20 * i) for i in range(8)])
    captioner.caption([make_clip(20 * i) for i in range(8)])

    assert captioner.usage.llm_calls == 16
    assert captioner.usage.tokens_out == 32


def test_no_clips_means_no_call() -> None:
    chat = RecordingChat()

    assert OpenAIVisionCaptioner(chat, workers=4).caption([]) == []
    assert chat.calls == []


def test_rejects_a_worker_count_below_one() -> None:
    with pytest.raises(ValueError):
        OpenAIVisionCaptioner(RecordingChat(), workers=0)


class TestPerClipPrompts:
    """One instruction per clip: what a run with context segments needs."""

    def test_each_message_carries_the_prompt_of_its_own_clip(self) -> None:
        chat = RecordingChat("first", "middle", "last")

        captions = OpenAIVisionCaptioner(chat, prompt=STATE_PROMPT).caption(
            [make_clip(0), make_clip(40), make_clip(80)],
            prompts=["ask first", "ask middle", "ask last"],
        )

        assert captions == ["first", "middle", "last"]
        assert [messages[0].text for messages, _, _ in chat.calls] == [
            "ask first",
            "ask middle",
            "ask last",
        ]

    def test_the_prompts_stay_with_their_clips_across_workers(self) -> None:
        chat = EchoingChat()
        clips = [make_clip(first) for first in (0, 40, 80, 120)]

        captions = OpenAIVisionCaptioner(chat, workers=3).caption(
            clips, prompts=[f"ask {first}" for first in (0, 40, 80, 120)]
        )

        assert captions == [f"clip {first}" for first in (0, 40, 80, 120)]
        sent = {
            int((messages[0].video or messages[0].images)[0][0, 0, 0]): messages[0].text
            for messages, _, _ in chat.calls
        }
        assert sent == {first: f"ask {first}" for first in (0, 40, 80, 120)}

    def test_no_prompts_at_all_keeps_the_captioners_own_instruction(self) -> None:
        chat = RecordingChat("a caption", "another")

        OpenAIVisionCaptioner(chat, prompt=STATE_PROMPT).caption(
            [make_clip(0), make_clip(40)], prompts=None
        )

        assert [messages[0].text for messages, _, _ in chat.calls] == [
            STATE_PROMPT,
            STATE_PROMPT,
        ]

    def test_a_prompt_list_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="prompts"):
            OpenAIVisionCaptioner(RecordingChat()).caption(
                [make_clip(0), make_clip(40)], prompts=["only one"]
            )

    def test_a_prompt_for_no_clip_at_all_is_refused(self) -> None:
        # Checked before the empty-clip shortcut: a caller that lost its
        # clips must not be told the pass succeeded.
        with pytest.raises(ValueError, match="prompts"):
            OpenAIVisionCaptioner(RecordingChat()).caption([], prompts=["a prompt"])

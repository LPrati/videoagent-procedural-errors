"""The OpenAI-compatible chat adapter, driven by a stub SDK client.

The stub stands in for ``openai.OpenAI``: injecting it must keep the adapter
from importing the SDK at all, so these tests run without the agent extra.
"""

import base64
import os
import tempfile

import cv2
import numpy as np
import pytest

from video_agent.adapters.chat_openai import (
    OpenAICompatibleChat,
    encode_video_data_url,
    to_openai_message,
)
from video_agent.models.types import ChatMessage


class StubMessage:
    def __init__(self, content: str | None, reasoning: str | None = None) -> None:
        self.content = content
        if reasoning is not None:
            # Only a server started with a reasoning parser sets the field;
            # everyone else's message object simply has no such attribute.
            self.reasoning_content = reasoning


class StubChoice:
    def __init__(
        self, content: str | None, finish_reason: str, reasoning: str | None = None
    ) -> None:
        self.message = StubMessage(content, reasoning)
        self.finish_reason = finish_reason


class StubUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class StubResponse:
    """Shaped like ``openai.types.chat.ChatCompletion``."""

    def __init__(
        self,
        content: str | None,
        *,
        model: str = "openai/gpt-4o-2024-05-13",
        finish_reason: str = "stop",
        prompt_tokens: int = 11,
        completion_tokens: int = 7,
        reasoning: str | None = None,
    ) -> None:
        self.choices = [StubChoice(content, finish_reason, reasoning)]
        self.usage = StubUsage(prompt_tokens, completion_tokens)
        self.model = model

    def model_dump(self) -> dict:
        return {"model": self.model, "content": self.choices[0].message.content}


class StubCompletions:
    def __init__(self, responses: list[StubResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs) -> StubResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("more requests than canned responses")
        return self.responses.pop(0)


class StubNamespace:
    def __init__(self, completions: StubCompletions) -> None:
        self.completions = completions


class StubClient:
    def __init__(self, *responses: StubResponse) -> None:
        self.completions = StubCompletions(list(responses))
        self.chat = StubNamespace(self.completions)


def solid_image(color: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def make_frames(count: int = 6, size: int = 16) -> tuple[np.ndarray, ...]:
    """Frames of one clip, each carrying its own index in every pixel.

    16x16 rather than 8x8: an MPEG-4 macroblock is 16 pixels wide, and a
    frame smaller than one is not something every encoder will take.
    """
    return tuple(
        np.full((size, size, 3), index, dtype=np.uint8) for index in range(count)
    )


def read_back(data_url: str) -> tuple[int, float]:
    """Decode a video ``data:`` URL to a file and probe it: frames, fps.

    The frame count is counted by reading, not read off the container's
    metadata, so a header that lies cannot pass the test.
    """
    payload = base64.b64decode(data_url.split(",", 1)[1])
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        handle.write(payload)
        path = handle.name
    try:
        capture = cv2.VideoCapture(path)
        assert capture.isOpened()
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = 0
        while capture.read()[0]:
            count += 1
        capture.release()
        return count, fps
    finally:
        os.unlink(path)


def test_sends_one_user_message_and_maps_the_reply() -> None:
    client = StubClient(StubResponse("Thought: I should look", finish_reason="stop"))
    chat = OpenAICompatibleChat(client=client)

    reply = chat.complete([ChatMessage(role="user", text="Question: why?")])

    request = client.completions.calls[0]
    assert request["messages"] == [{"role": "user", "content": "Question: why?"}]
    assert request["model"] == "openai/gpt-4o-2024-05-13"
    assert reply.text == "Thought: I should look"
    assert reply.model == "openai/gpt-4o-2024-05-13"
    assert reply.finish_reason == "stop"
    assert reply.raw == {"model": "openai/gpt-4o-2024-05-13", "content": reply.text}
    assert reply.usage.llm_calls == 1
    assert (reply.usage.tokens_in, reply.usage.tokens_out) == (11, 7)


def test_sends_frames_as_jpeg_data_urls_after_the_text() -> None:
    client = StubClient(StubResponse("a pan on the stove"))
    chat = OpenAICompatibleChat(client=client)
    red, blue = solid_image((200, 20, 20)), solid_image((20, 20, 200))

    chat.complete([ChatMessage(role="user", text="what is this?", images=(red, blue))])

    content = client.completions.calls[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert [part["type"] for part in content[1:]] == ["image_url", "image_url"]
    urls = [part["image_url"]["url"] for part in content[1:]]
    assert all(url.startswith("data:image/jpeg;base64,") for url in urls)
    # The frames survive the round trip as RGB (JPEG is lossy, hence the
    # tolerance) and keep their order.
    decoded = [
        cv2.cvtColor(
            cv2.imdecode(
                np.frombuffer(base64.b64decode(url.split(",", 1)[1]), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            ),
            cv2.COLOR_BGR2RGB,
        )
        for url in urls
    ]
    assert np.allclose(decoded[0].mean(axis=(0, 1)), (200, 20, 20), atol=12)
    assert np.allclose(decoded[1].mean(axis=(0, 1)), (20, 20, 200), atol=12)


def test_passes_stop_max_tokens_temperature_and_seed() -> None:
    client = StubClient(StubResponse("ok"))
    chat = OpenAICompatibleChat(client=client, temperature=0.2, seed=17)

    chat.complete(
        [ChatMessage(role="user", text="hi")], stop=["\nObservation"], max_tokens=64
    )

    request = client.completions.calls[0]
    assert request["stop"] == ["\nObservation"]
    assert request["max_tokens"] == 64
    assert request["temperature"] == 0.2
    assert request["seed"] == 17


def test_omits_optional_fields_that_are_not_configured() -> None:
    client = StubClient(StubResponse("ok"))
    chat = OpenAICompatibleChat(client=client)

    chat.complete([ChatMessage(role="user", text="hi")])

    request = client.completions.calls[0]
    assert "stop" not in request and "max_tokens" not in request
    assert "seed" not in request and "extra_headers" not in request
    assert request["temperature"] == 0.0


def test_forwards_provider_specific_headers_and_body() -> None:
    client = StubClient(StubResponse("ok"))
    chat = OpenAICompatibleChat(
        client=client,
        extra_headers={"HTTP-Referer": "https://example.org"},
        extra_body={"provider": {"order": ["OpenAI"]}},
    )

    chat.complete([ChatMessage(role="user", text="hi")])

    request = client.completions.calls[0]
    assert request["extra_headers"] == {"HTTP-Referer": "https://example.org"}
    assert request["extra_body"] == {"provider": {"order": ["OpenAI"]}}


def test_a_reasoning_block_is_recorded_but_never_joins_the_completion() -> None:
    """``--reasoning-parser qwen3`` splits the thinking block off ``content``.

    The scratchpad concatenates ``text`` turn after turn and would drown in
    it, so it stays apart; its tokens are already inside ``completion_tokens``
    and are not counted twice.
    """
    client = StubClient(
        StubResponse(
            "Thought: I now know\nFinal Answer: 2",
            reasoning="The user wants a count. There are two boats.",
            completion_tokens=40,
        )
    )
    chat = OpenAICompatibleChat(client=client)

    reply = chat.complete([ChatMessage(role="user", text="how many?")])

    assert reply.text == "Thought: I now know\nFinal Answer: 2"
    assert reply.reasoning == "The user wants a count. There are two boats."
    assert reply.usage.tokens_out == 40


def test_a_reply_without_a_reasoning_block_has_none() -> None:
    client = StubClient(StubResponse("Final Answer: 2"))

    reply = OpenAICompatibleChat(client=client).complete(
        [ChatMessage(role="user", text="how many?")]
    )

    assert reply.reasoning is None


def test_retries_once_on_empty_content_and_bills_both_calls() -> None:
    client = StubClient(
        StubResponse("   ", prompt_tokens=10, completion_tokens=0),
        StubResponse("Final Answer: 2", prompt_tokens=10, completion_tokens=5),
    )
    chat = OpenAICompatibleChat(client=client)

    reply = chat.complete([ChatMessage(role="user", text="hi")])

    assert len(client.completions.calls) == 2
    assert reply.text == "Final Answer: 2"
    assert reply.usage.llm_calls == 2
    assert (reply.usage.tokens_in, reply.usage.tokens_out) == (20, 5)


def test_returns_the_second_empty_reply_rather_than_looping() -> None:
    client = StubClient(StubResponse(None), StubResponse(""))
    chat = OpenAICompatibleChat(client=client)

    reply = chat.complete([ChatMessage(role="user", text="hi")])

    assert reply.text == ""
    assert len(client.completions.calls) == 2


def test_rejects_an_empty_conversation() -> None:
    chat = OpenAICompatibleChat(client=StubClient())

    with pytest.raises(ValueError):
        chat.complete([])


def test_rejects_a_response_without_choices() -> None:
    class Empty:
        choices: list = []

    client = StubClient()
    client.completions.responses.append(Empty())
    chat = OpenAICompatibleChat(client=client)

    with pytest.raises(ValueError):
        chat.complete([ChatMessage(role="user", text="hi")])


class TestEncodeVideoDataUrl:
    def test_the_frames_come_back_as_a_readable_mp4_at_the_given_rate(self) -> None:
        url = encode_video_data_url(make_frames(6), 5.0)

        assert url.startswith("data:video/mp4;base64,")
        assert read_back(url) == (6, 5.0)

    def test_the_frame_rate_is_the_one_the_clip_was_decoded_at(self) -> None:
        assert read_back(encode_video_data_url(make_frames(4), 2.0)) == (4, 2.0)

    def test_an_empty_clip_is_not_a_video(self) -> None:
        with pytest.raises(ValueError):
            encode_video_data_url([], 5.0)

    def test_frames_of_different_shapes_cannot_form_one_video(self) -> None:
        frames = [
            np.zeros((16, 16, 3), dtype=np.uint8),
            np.zeros((16, 32, 3), dtype=np.uint8),
        ]

        with pytest.raises(ValueError):
            encode_video_data_url(frames, 5.0)


class TestVideoInTheMessage:
    def test_a_clip_becomes_one_video_part_after_the_text(self) -> None:
        message = ChatMessage(role="user", text="describe it", video=make_frames(6))

        content = to_openai_message(message, video_fps=5.0)["content"]

        assert content[0] == {"type": "text", "text": "describe it"}
        assert [part["type"] for part in content[1:]] == ["video_url"]
        assert read_back(content[1]["video_url"]["url"]) == (6, 5.0)

    def test_images_come_first_and_the_video_last(self) -> None:
        message = ChatMessage(
            role="user",
            text="and this?",
            images=(solid_image((200, 20, 20)),),
            video=make_frames(4),
        )

        content = to_openai_message(message)["content"]

        assert [part["type"] for part in content] == [
            "text",
            "image_url",
            "video_url",
        ]

    def test_a_turn_with_neither_stays_a_plain_string(self) -> None:
        message = ChatMessage(role="user", text="no frames here")

        assert to_openai_message(message) == {
            "role": "user",
            "content": "no frames here",
        }


def test_sends_a_clip_as_one_video_part_at_the_configured_rate() -> None:
    client = StubClient(StubResponse("a person stirs a pot"))
    chat = OpenAICompatibleChat(client=client, video_fps=2.0)

    chat.complete([ChatMessage(role="user", text="describe", video=make_frames(6))])

    content = client.completions.calls[0]["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text", "video_url"]
    # The adapter's rate reaches the container, so what the server reads as
    # the clip's timeline is what the clip was decoded at.
    assert read_back(content[1]["video_url"]["url"]) == (6, 2.0)

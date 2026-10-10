"""Frame-sampling VQA over a chat model, against a local recording fake."""

import numpy as np
import pytest
from tests.fakes import RecordingChat

from video_agent.adapters.vqa_openai import GPT4V_PROMPT, OpenAIVisionVideoQA
from video_agent.media import Clip, Frame

DESCRIBE = "what is the video about?"
QUESTION = "what is the person holding?"


def make_clip(frame_count: int = 12) -> Clip:
    """Clip whose frames carry their own index in every pixel."""
    return Clip(
        frames=tuple(
            Frame(
                index=index,
                time_s=index / 30,
                data=np.full((4, 4, 3), index, dtype=np.uint8),
            )
            for index in range(frame_count)
        )
    )


def test_asks_one_question_per_pair_with_four_frames_each() -> None:
    chat = RecordingChat("a kitchen", "a frying pan")
    vqa = OpenAIVisionVideoQA(chat)
    clip = make_clip()

    answers = vqa.answer([clip, clip], [DESCRIBE, QUESTION])

    assert answers == ["a kitchen", "a frying pan"]
    assert len(chat.calls) == 2
    for (messages, _, max_tokens), question in zip(chat.calls, [DESCRIBE, QUESTION]):
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].text == question  # sent as the tool phrased it
        assert len(messages[0].images) == 4
        assert max_tokens == 200


def test_samples_frames_evenly_across_the_window() -> None:
    chat = RecordingChat("a kitchen")
    vqa = OpenAIVisionVideoQA(chat)

    vqa.answer([make_clip(12)], [QUESTION])

    images = chat.calls[0][0][0].images
    assert [int(image[0, 0, 0]) for image in images] == [0, 3, 6, 9]


def test_wraps_the_question_in_the_reference_gpt4v_prompt() -> None:
    chat = RecordingChat("The person picks up a frying pan.")
    vqa = OpenAIVisionVideoQA(chat, question_template=GPT4V_PROMPT)

    answers = vqa.answer([make_clip()], [QUESTION])

    assert len(chat.calls) == 1
    messages, _, max_tokens = chat.calls[0]
    assert messages[0].text == (
        "The images are sequential frames in a 6-second video. Please briefly "
        "describe what happened in the video then briefly provide the answer "
        f"to the question '{QUESTION}'."
    )
    assert len(messages[0].images) == 4
    assert max_tokens == 200
    assert answers == ["The person picks up a frying pan."]


def test_sends_one_call_per_pair_even_with_a_template() -> None:
    chat = RecordingChat("first", "second")
    vqa = OpenAIVisionVideoQA(chat, question_template=GPT4V_PROMPT)
    clip = make_clip()

    answers = vqa.answer([clip, clip], [DESCRIBE, QUESTION])

    assert answers == ["first", "second"]
    assert [call[0][0].text for call in chat.calls] == [
        GPT4V_PROMPT.format(question=DESCRIBE),
        GPT4V_PROMPT.format(question=QUESTION),
    ]


def test_honors_a_smaller_frame_budget() -> None:
    chat = RecordingChat("short")
    vqa = OpenAIVisionVideoQA(chat, frames_per_clip=2)

    vqa.answer([make_clip(12)], [QUESTION])

    images = chat.calls[0][0][0].images
    assert [int(image[0, 0, 0]) for image in images] == [0, 6]


def test_sends_every_frame_when_the_clip_is_shorter_than_the_budget() -> None:
    chat = RecordingChat("short")
    vqa = OpenAIVisionVideoQA(chat)

    vqa.answer([make_clip(2)], [QUESTION])

    assert len(chat.calls[0][0][0].images) == 2


def test_rejects_misaligned_clips_and_questions() -> None:
    vqa = OpenAIVisionVideoQA(RecordingChat())

    with pytest.raises(ValueError):
        vqa.answer([make_clip()], [DESCRIBE, QUESTION])

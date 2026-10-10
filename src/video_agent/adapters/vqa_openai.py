"""Clip question answering by sampling frames into a multimodal ChatModel.

Reproduces the reference's ``gpt-4v`` VQA option: four frames evenly spaced
over the 6-second window are inlined in one chat completion capped at 200
tokens (see ``tools.py::gpt4v_VQA``).
"""

from typing import Sequence

from video_agent.adapters.frames import evenly_spaced
from video_agent.media import Clip
from video_agent.models.interfaces import ChatModel
from video_agent.models.types import ChatMessage

GPT4V_PROMPT = (
    "The images are sequential frames in a 6-second video. Please briefly "
    "describe what happened in the video then briefly provide the answer to "
    "the question '{question}'."
)
"""Verbatim from the reference ``gpt4v_VQA``.

It asks for the description and the answer at once, so it belongs with the
single-question tool style (``ToolConfig.vqa_style="single"``), where the
whole reply becomes the observation.
"""


class OpenAIVisionVideoQA:
    """VideoQA over any multimodal ``ChatModel`` (OpenRouter, Azure, vLLM).

    One chat completion per ``(clip, question)`` pair, always.
    ``question_template`` wraps each question first — pass ``GPT4V_PROMPT``
    to reproduce the reference prompt, or leave it ``None`` to send the
    question as the tool phrased it.
    """

    def __init__(
        self,
        chat: ChatModel,
        *,
        frames_per_clip: int = 4,
        question_template: str | None = None,
        max_tokens: int = 200,
    ) -> None:
        self.chat = chat
        self.frames_per_clip = frames_per_clip
        self.question_template = question_template
        self.max_tokens = max_tokens

    def _prompt(self, question: str) -> str:
        if self.question_template is None:
            return question
        return self.question_template.format(question=question)

    def answer(self, clips: Sequence[Clip], questions: Sequence[str]) -> list[str]:
        """Return one answer per ``(clip, question)`` pair, aligned by index."""
        if len(clips) != len(questions):
            raise ValueError(f"{len(clips)} clips != {len(questions)} questions")
        answers: list[str] = []
        for clip, question in zip(clips, questions):
            frames = tuple(
                frame.data for frame in evenly_spaced(clip.frames, self.frames_per_clip)
            )
            message = ChatMessage(
                role="user", text=self._prompt(question), images=frames
            )
            reply = self.chat.complete([message], max_tokens=self.max_tokens)
            answers.append(reply.text)
        return answers

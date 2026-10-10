"""Clip question answering with Qwen3-VL instruct models.

Local stand-in for the reference Video-LLaVA-7B server: the reference ran it
in a separate process behind a unix socket and asked one question per
forward pass, which is exactly this adapter's batched ``answer``.
"""

from typing import Any, Sequence

from video_agent.adapters.captioner import generate_video_texts
from video_agent.adapters.frames import evenly_spaced
from video_agent.media import Clip, Image


class Qwen3VLVideoQA:
    """VideoQA backed by a Qwen3-VL instruct checkpoint.

    ``model``/``processor`` inject already-loaded weights so a script can
    share one checkpoint between captioning and question answering instead
    of paying for a second copy on the GPU.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        max_frames_per_clip: int | None = 8,
        max_new_tokens: int = 128,
        batch_size: int = 4,
        device_map: str = "cuda",  # all on the GPU: fail fast rather than
        dtype: str = "bfloat16",  # silently offload to CPU or disk
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        """Answers are short by default (``max_new_tokens=128``).

        The observation is pasted into the orchestrator's scratchpad, so a
        long answer costs tokens in every later ReAct step.
        """
        self.max_frames_per_clip = max_frames_per_clip
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        if (model is None) != (processor is None):
            raise ValueError("pass both model= and processor=, or neither")
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
            return
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"  # batched generation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, device_map=device_map, dtype=dtype
        )
        self.model.eval()

    def _clip_frames(self, clip: Clip) -> list[Image]:
        frames = list(clip.frames)
        if self.max_frames_per_clip is not None:
            frames = evenly_spaced(frames, self.max_frames_per_clip)
        return [frame.data for frame in frames]

    def answer(self, clips: Sequence[Clip], questions: Sequence[str]) -> list[str]:
        """Return one answer per ``(clip, question)`` pair, aligned by index."""
        if len(clips) != len(questions):
            raise ValueError(f"{len(clips)} clips != {len(questions)} questions")
        answers: list[str] = []
        for start in range(0, len(clips), self.batch_size):
            batch = clips[start : start + self.batch_size]
            answers.extend(
                generate_video_texts(
                    self.processor,
                    self.model,
                    [self._clip_frames(clip) for clip in batch],
                    list(questions[start : start + self.batch_size]),
                    max_new_tokens=self.max_new_tokens,
                )
            )
        return answers

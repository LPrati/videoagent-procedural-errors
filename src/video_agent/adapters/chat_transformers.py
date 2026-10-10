"""Chat completion with a local ``transformers`` causal LM.

Text-only orchestrator: the ReAct loop only ever sends text, so an open
checkpoint can replace the reference's hosted GPT-4o without touching the
agent layer. Messages carrying frames are rejected rather than silently
stripped (see ``models.types.ChatMessage``).
"""

import re
from typing import Any, Sequence

from video_agent.models.types import ChatMessage, ChatReply, Usage

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
"""Qwen3 thinking checkpoints wrap their reasoning in this block."""


def strip_thinking(text: str) -> str:
    """Drop a Qwen3 ``<think>`` block, including one left open by the template.

    Chat templates prefill ``<think>`` for thinking checkpoints, so the
    decoded completion can carry a closing tag without its opening tag.
    """
    text = THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text


def cut_at_stop(text: str, stop: Sequence[str]) -> tuple[str, bool]:
    """Truncate ``text`` at the earliest stop string; also report whether one hit.

    ``generate(stop_strings=...)`` stops *after* emitting the string, and
    some backends ignore the argument entirely, so the cut is applied here
    as well — the ReAct scratchpad must never contain an "Observation".
    """
    positions = [text.find(s) for s in stop if s and s in text]
    if not positions:
        return text, False
    return text[: min(positions)], True


class TransformersChat:
    """ChatModel backed by a local instruct checkpoint.

    Greedy by default (``temperature=0.0``), mirroring the reference's
    ``temperature=0`` orchestrator.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,  # 0 = greedy decoding
        device_map: str = "cuda",  # all on the GPU: fail fast rather than
        dtype: str = "bfloat16",  # silently offload to CPU or disk
        enable_thinking: bool = False,
        stop_strings_supported: bool = True,
    ) -> None:
        """``stop_strings_supported=False`` falls back to truncating locally.

        ``generate(stop_strings=..., tokenizer=...)`` exists since
        transformers 4.39 but a few model classes still reject it.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.stop_strings_supported = stop_strings_supported
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, dtype=dtype
        )
        self.model.eval()

    def _render(self, messages: Sequence[ChatMessage]) -> str:
        turns = [{"role": m.role, "content": m.text} for m in messages]
        try:
            return self.tokenizer.apply_chat_template(
                turns,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Templates without a thinking switch reject the keyword.
            return self.tokenizer.apply_chat_template(
                turns, tokenize=False, add_generation_prompt=True
            )

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] = (),
        max_tokens: int | None = None,
    ) -> ChatReply:
        """Return the completion for ``messages``, truncated at any ``stop`` string."""
        import torch

        if not messages:
            raise ValueError("at least one message is required")
        with_images = [m.role for m in messages if m.images]
        if with_images:
            raise ValueError(
                f"{type(self).__name__} is text-only but {len(with_images)} "
                f"message(s) carry images: {with_images}"
            )
        with_video = [m.role for m in messages if m.video]
        if with_video:
            raise ValueError(
                f"{type(self).__name__} is text-only but {len(with_video)} "
                f"message(s) carry video: {with_video}"
            )
        inputs = self.tokenizer([self._render(messages)], return_tensors="pt").to(
            self.model.device
        )
        budget = self.max_new_tokens if max_tokens is None else max_tokens
        generation: dict[str, Any] = {
            "max_new_tokens": budget,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation["temperature"] = self.temperature
        if stop and self.stop_strings_supported:
            generation["stop_strings"] = list(stop)
            generation["tokenizer"] = self.tokenizer
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        prompt_length = inputs["input_ids"].shape[1]
        completion = generated[0][prompt_length:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        text, hit_stop = cut_at_stop(strip_thinking(text), stop)
        return ChatReply(
            text=text,
            usage=Usage(
                llm_calls=1,
                tokens_in=int(prompt_length),
                tokens_out=int(completion.shape[0]),
            ),
            model=self.model_name,
            finish_reason=(
                "stop" if hit_stop or completion.shape[0] < budget else "length"
            ),
        )

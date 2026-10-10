"""Chat completion over any OpenAI-compatible HTTP API (OpenRouter, vLLM, ...).

The reference orchestrator talked to Azure OpenAI through LangChain; this
adapter keeps the same wire format (one chat completion per ReAct step, the
provider-side ``stop`` sequence doing the "\\nObservation" truncation) while
making sampling settings explicit so a run is reproducible from its config.
"""

import base64
import os
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np

from video_agent.media import Image
from video_agent.models.types import ChatMessage, ChatReply, Usage


def encode_image_data_url(image: Image) -> str:
    """Encode an RGB frame as a base64 JPEG ``data:`` URL.

    Reference behavior: frames are JPEG-encoded and inlined in the request
    (the reference ``gpt4v_VQA`` also wrote them to disk first; this adapter
    encodes them in memory).
    """
    array = np.asarray(image)
    success, buffer = cv2.imencode(".jpg", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
    if not success:
        raise ValueError(f"cannot JPEG-encode image of shape {array.shape}")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def encode_video_data_url(frames: Sequence[Image], fps: float) -> str:
    """Encode RGB frames as one base64 MP4 ``data:`` URL, at ``fps``.

    MP4 and not something lossless: a ``data:`` URL is decoded by the
    server's own media loader, and MP4 (H.264/MPEG-4) is the one container
    such a loader is guaranteed to read — an uncompressed AVI would be
    several times the payload and is not universally accepted. The encoding
    is therefore lossy, but the frames are 360p stills of a kitchen and the
    difference is invisible in a caption.

    OpenCV writes through a filename only, so the clip goes to a temporary
    file, is read back as bytes and the file is removed.
    """
    if not frames:
        raise ValueError("cannot encode an empty frame sequence as a video")
    arrays = [np.asarray(frame) for frame in frames]
    shapes = {array.shape for array in arrays}
    if len(shapes) > 1:
        raise ValueError(f"frames of a video must share one shape, got {shapes}")
    height, width = arrays[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        path = handle.name
    try:
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise ValueError(f"cannot open an MP4 writer for {width}x{height}")
        try:
            for array in arrays:
                writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        with open(path, "rb") as encoded_file:
            encoded = base64.b64encode(encoded_file.read()).decode("ascii")
    finally:
        os.unlink(path)
    return f"data:video/mp4;base64,{encoded}"


def to_openai_message(
    message: ChatMessage, *, video_fps: float = 5.0
) -> dict[str, Any]:
    """Convert one ``ChatMessage`` to the OpenAI chat format.

    Text-only turns keep a plain string content (accepted by every
    OpenAI-compatible server); turns carrying frames use the multimodal
    content list, text first, as in the reference payload. ``message.video``
    becomes a single ``video_url`` part after the images: the same frames
    read as one clip rather than as a series of stills.

    ``video_fps`` only writes the container's timestamps — the frames sent
    are the frames given, whatever rate they are stamped with.
    """
    if not message.images and not message.video:
        return {"role": message.role, "content": message.text}
    content: list[dict[str, Any]] = [{"type": "text", "text": message.text}]
    content.extend(
        {"type": "image_url", "image_url": {"url": encode_image_data_url(image)}}
        for image in message.images
    )
    if message.video:
        content.append(
            {
                "type": "video_url",
                "video_url": {"url": encode_video_data_url(message.video, video_fps)},
            }
        )
    return {"role": message.role, "content": content}


class OpenAICompatibleChat:
    """ChatModel backed by an OpenAI-compatible endpoint.

    ``client`` injects an already-built SDK client (tests, or sharing one
    connection pool); only when it is ``None`` is ``openai`` imported and a
    client constructed, so the adapter works without the ``agent`` extra.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4o-2024-05-13",
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        temperature: float = 0.0,
        seed: int | None = None,
        video_fps: float = 5.0,
        max_retries: int = 5,
        timeout_s: float = 120.0,
        extra_headers: dict | None = None,
        extra_body: dict | None = None,
        client: Any | None = None,
    ) -> None:
        """HTTP errors are retried by the SDK (``max_retries``); see ``complete``.

        ``temperature=0.0`` and an explicit ``seed`` reproduce the reference
        setting (``temperature=0``, non-streaming); the seed is best-effort,
        as most providers only honor it for some backends.

        ``video_fps`` is the rate the clips were decoded at
        (``preprocess_temporal.CLIP_SAMPLE_FPS``). It stamps the MP4 a
        ``ChatMessage.video`` is written into; no frame is added or dropped
        by it, only the timestamps the server reads change.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.video_fps = video_fps
        self.extra_headers = extra_headers
        self.extra_body = extra_body
        if client is not None:
            self.client = client
            return
        from openai import OpenAI

        key = api_key if api_key is not None else os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"no API key: pass api_key= or set ${api_key_env}")
        self.client = OpenAI(
            base_url=base_url,
            api_key=key,
            max_retries=max_retries,
            timeout=timeout_s,
        )

    def _request(
        self,
        messages: Sequence[ChatMessage],
        stop: Sequence[str],
        max_tokens: int | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                to_openai_message(message, video_fps=self.video_fps)
                for message in messages
            ],
            "temperature": self.temperature,
        }
        if stop:
            request["stop"] = list(stop)
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if self.seed is not None:
            request["seed"] = self.seed
        if self.extra_headers:
            request["extra_headers"] = self.extra_headers
        if self.extra_body:
            request["extra_body"] = self.extra_body
        return request

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] = (),
        max_tokens: int | None = None,
    ) -> ChatReply:
        """Return the completion for ``messages``, truncated at any ``stop`` string.

        Empty content is retried once: providers occasionally return a blank
        choice, and an empty ReAct step would otherwise burn an iteration.
        Both attempts are billed, so their usage is accumulated.
        """
        if not messages:
            raise ValueError("at least one message is required")
        request = self._request(messages, stop, max_tokens)
        reply = self._to_reply(self.client.chat.completions.create(**request))
        if reply.text.strip():
            return reply
        retry = self._to_reply(self.client.chat.completions.create(**request))
        return ChatReply(
            text=retry.text,
            usage=reply.usage + retry.usage,
            model=retry.model,
            finish_reason=retry.finish_reason,
            reasoning=retry.reasoning,
            raw=retry.raw,
        )

    def _to_reply(self, response: Any) -> ChatReply:
        choices = getattr(response, "choices", None) or ()
        if not choices:
            raise ValueError(f"no choices in response from {self.model_name}")
        choice = choices[0]
        # Content is kept verbatim: the ReAct scratchpad concatenates it and
        # the reference never trimmed the completion.
        text = getattr(choice.message, "content", None) or ""
        # A server started with a reasoning parser (``--reasoning-parser
        # qwen3``) moves the thinking block to ``reasoning_content`` and
        # leaves ``content`` clean. It stays out of the scratchpad but is
        # recorded, and its tokens are already inside ``completion_tokens``.
        reasoning = getattr(choice.message, "reasoning_content", None) or None
        counts = getattr(response, "usage", None)
        usage = Usage(
            llm_calls=1,
            tokens_in=int(getattr(counts, "prompt_tokens", 0) or 0),
            tokens_out=int(getattr(counts, "completion_tokens", 0) or 0),
        )
        dump = getattr(response, "model_dump", None)
        return ChatReply(
            text=text,
            usage=usage,
            model=str(getattr(response, "model", None) or self.model_name),
            finish_reason=getattr(choice, "finish_reason", None),
            reasoning=reasoning,
            raw=dump() if callable(dump) else None,
        )

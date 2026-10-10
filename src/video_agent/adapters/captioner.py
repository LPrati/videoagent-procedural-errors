"""Clip captioning with Qwen3-VL instruct models.

Hosts ``generate_video_texts``, the batched video+text generate path shared
with ``adapters.vqa_qwen`` (same checkpoint family, same chat-template
mechanics, different prompts), and ``load_prompt_file``, the instruction
file a run can be handed instead of one string on the command line.
"""

import re
from pathlib import Path
from typing import Any, Sequence

from video_agent.adapters.frames import evenly_spaced
from video_agent.media import Clip, Image

DEFAULT_PROMPT = (
    "Describe what the person is doing in this short video clip in one "
    "concise sentence. Mention the objects being manipulated."
)

CAPTION_INSTRUCTIONS: dict[str, str] = {
    "action": DEFAULT_PROMPT,
    "state": (
        "In one concise sentence, describe the action in this short clip AND "
        "the state of the ingredients or objects involved (e.g. chopped, "
        "mixed, raw, in the bowl, on the pan)."
    ),
    "order": (
        "In one concise sentence, state which cooking action is being "
        "performed in this clip, on which ingredients, and whether it is "
        "starting, ongoing, or finishing."
    ),
}
"""The three caption instructions a batch script can select by name.

``action`` is the adapter default and asks for the action alone; ``state``
also asks for the state of the objects involved, and ``order`` for the phase
of the action (starting, ongoing or finishing) — the two signals a procedural
task needs and a bare action caption drops. Keeping them here lets a script
name an instruction instead of pasting its text.
"""


PROMPT_POSITIONS: tuple[str, ...] = ("middle", "first", "last")
"""Where the described segment sits inside the clip that carries it.

With ``--context-segments k`` a clip covers its neighbours too, and the
instruction has to say which part of it the sentence is about; at the two
ends of a recording the clip is shorter and the wording differs, which is
why a position is a prompt and not a formatting detail.
"""

DEFAULT_POSITION = "middle"
"""The position an untagged ``## user`` block writes, and the only one a run
without context ever uses."""

_PROMPT_BLOCK = re.compile(r"^##\s+user\s*(?:\[\s*([a-zA-Z]+)\s*\])?\s*$")
_HEADING = re.compile(r"^##\s+")


def load_prompt_file(path: Path) -> dict[str, str]:
    """Read a markdown instruction file into one prompt per clip position.

    A block starts at a ``## user [middle|first|last]`` heading (an untagged
    ``## user`` is the middle one) and runs to the next ``## `` heading or to
    the end of the file; its lines are stripped and joined with a space, so
    the file can wrap a prompt over several lines and the model still gets
    one paragraph. Everything before the first block (a title, a version
    note, a description of the prompt) is documentation and never reaches
    the model.

    The ``middle`` block is required because it is what every other position
    falls back to; a file without it would caption part of a recording with
    an instruction nobody wrote.
    """
    path = Path(path)
    lines_by_position: dict[str, list[str]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PROMPT_BLOCK.match(line)
        if match is not None:
            current = (match.group(1) or DEFAULT_POSITION).lower()
            if current not in PROMPT_POSITIONS:
                raise ValueError(
                    f"{path}: unknown prompt position {current!r}; expected "
                    f"one of {list(PROMPT_POSITIONS)}"
                )
            if current in lines_by_position:
                raise ValueError(f"{path}: two '## user [{current}]' blocks")
            lines_by_position[current] = []
        elif _HEADING.match(line):
            current = None  # any other '## ' heading closes the block
        elif current is not None:
            lines_by_position[current].append(line)
    prompts = {}
    for position, lines in lines_by_position.items():
        text = " ".join(part for part in (line.strip() for line in lines) if part)
        if text:
            prompts[position] = text
    if DEFAULT_POSITION not in prompts:
        raise ValueError(
            f"{path}: no '## user' block; the {DEFAULT_POSITION!r} one is "
            "the instruction every other position falls back to"
        )
    return prompts


def resolve_prompts(
    default: str, clips: Sequence[Clip], prompts: Sequence[str] | None
) -> list[str]:
    """One prompt per clip: the given ones, or the default repeated.

    Shared by the two captioners so a mismatched list fails the same way on
    both paths. It has to fail loudly: captioning clip *i* with prompt *i−1*
    is the one error that leaves no trace in the captions it writes.
    """
    if prompts is None:
        return [default] * len(clips)
    texts = [str(prompt) for prompt in prompts]
    if len(texts) != len(clips):
        raise ValueError(f"{len(clips)} clips != {len(texts)} prompts")
    return texts


def generate_video_texts(
    processor: Any,
    model: Any,
    clips_frames: Sequence[Sequence[Image]],
    texts: Sequence[str],
    *,
    max_new_tokens: int,
    do_sample: bool = False,
    top_p: float = 0.95,
    temperature: float = 0.7,
    num_candidates: int = 1,
) -> list[str]:
    """Generate one string per clip from ``texts[i]`` asked about ``clips_frames[i]``.

    One batched ``generate`` call; ``num_candidates`` only applies when
    sampling, and the longest candidate wins (reference LaViLa mechanics).
    """
    import torch

    if len(clips_frames) != len(texts):
        raise ValueError(f"{len(clips_frames)} clips != {len(texts)} texts")
    messages = [
        [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": text}],
            }
        ]
        for text in texts
    ]
    rendered = [
        processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        for message in messages
    ]
    inputs = processor(
        text=rendered,
        videos=[list(frames) for frames in clips_frames],
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,  # clips are already sampled upstream
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    candidates = num_candidates if do_sample else 1
    generation: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation.update(
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=candidates,
        )
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation)
    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # num_return_sequences groups candidates contiguously per input;
    # reference behavior: keep the longest candidate.
    grouped = [
        decoded[i * candidates : (i + 1) * candidates] for i in range(len(texts))
    ]
    return [max(group, key=len).strip() for group in grouped]


class Qwen3VLCaptioner:
    """Captioner backed by a Qwen3-VL instruct checkpoint.

    ``max_frames_per_clip`` caps the frames sent to the model per clip
    (evenly spaced); ``None`` sends every frame the clip carries.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        prompt: str = DEFAULT_PROMPT,
        max_frames_per_clip: int | None = None,
        max_new_tokens: int = 64,
        batch_size: int = 8,
        do_sample: bool = False,
        top_p: float = 0.95,  # only used when do_sample
        temperature: float = 0.7,  # only used when do_sample
        num_candidates: int = 1,  # >1: sample candidates, keep the longest
        device_map: str = "cuda",  # all on the GPU: fail fast rather than
        dtype: str = "bfloat16",  # silently offload to CPU or disk
    ) -> None:
        """Generation defaults are greedy/1-candidate.

        ``do_sample`` + ``num_candidates`` reproduce the reference VideoAgent
        captioning mechanics (LaViLa: nucleus top_p=0.95, T=0.7, 5 candidates,
        keep the longest); the reference narrator takes no prompt at all, so
        the prompt is a degree of freedom introduced by the model swap.
        """
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.prompt = prompt
        self.max_frames_per_clip = max_frames_per_clip
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature
        self.num_candidates = num_candidates
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"  # batched generation
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, device_map=device_map, dtype=dtype
        )
        self.model.eval()

    def _clip_frames(self, clip: Clip) -> list:
        frames = list(clip.frames)
        if self.max_frames_per_clip is not None:
            frames = evenly_spaced(frames, self.max_frames_per_clip)
        return [frame.data for frame in frames]

    def _caption_batch(
        self, clips: Sequence[Clip], prompts: Sequence[str]
    ) -> list[str]:
        return generate_video_texts(
            self.processor,
            self.model,
            [self._clip_frames(clip) for clip in clips],
            list(prompts),
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            top_p=self.top_p,
            temperature=self.temperature,
            num_candidates=self.num_candidates,
        )

    def caption(
        self, clips: Sequence[Clip], prompts: Sequence[str] | None = None
    ) -> list[str]:
        """Return one caption per clip, aligned by index.

        ``prompts`` asks a different question of each clip — what the clip
        positions of ``memory.temporal`` need — and ``None`` asks
        ``self.prompt`` of every one. ``generate_video_texts`` already takes
        one text per clip, so the batch carries the slice of the list its
        clips came from.
        """
        texts = resolve_prompts(self.prompt, clips, prompts)
        captions: list[str] = []
        for start in range(0, len(clips), self.batch_size):
            stop = start + self.batch_size
            captions.extend(self._caption_batch(clips[start:stop], texts[start:stop]))
        return captions

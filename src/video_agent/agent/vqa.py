"""Visual question answering over a segment and its neighbours.

The reference cut a 6-second window around the target segment — one segment
on each side, clamped to the video — wrote it to disk as an mp4 and sent it
to a Video-LLaVA server, asking two questions: what the video is about and
the agent's own question. The window and the two-question scheme are what
matter for fidelity; the decoding is ours (``media.load_clips``) and the
model is any ``VideoQA``.
"""

from typing import Literal, Sequence

from video_agent.media import Video, load_clips
from video_agent.memory.store import segment_interval
from video_agent.memory.types import Segment
from video_agent.models.interfaces import VideoQA

VQAStyle = Literal["two_questions", "single"]
"""How the window is questioned: the Video-LLaVA scheme or the gpt-4v one."""


def vqa_window(
    segments: Sequence[Segment], segment_id: int, context: int
) -> tuple[float, float]:
    """Time span of ``segment_id`` plus ``context`` segments on each side.

    Clamped to the grid, so a target at either end yields a shorter window
    rather than an invalid one — the reference dropped out-of-range
    neighbours the same way.
    """
    return segment_interval(segments, segment_id - context, segment_id + context)


def answer_about_segment(
    video: Video,
    segments: Sequence[Segment],
    segment_id: int,
    question: str,
    model: VideoQA,
    *,
    context: int,
    sample_fps: float,
    describe_question: str,
    style: VQAStyle = "two_questions",
) -> str:
    """Describe the window around ``segment_id`` and answer ``question`` about it.

    Under ``"two_questions"`` (the Video-LLaVA path) both questions go to the
    model in one batched call over the same clip and the answers are laid out
    as the reference's observation was: the description first, because the
    prompt warns the agent to trust it over the answer. Under ``"single"``
    (the gpt-4v path) one reply already describes the window and answers the
    question, and the reference passed it on verbatim.
    """
    start_s, end_s = vqa_window(segments, segment_id, context)
    clip = load_clips(video, [(start_s, end_s)], sample_fps)[0]
    if style == "single":
        answers = model.answer([clip], [question])
        if len(answers) != 1:
            raise ValueError(f"VideoQA returned {len(answers)} answers, expected 1")
        return answers[0]
    answers = model.answer([clip, clip], [describe_question, question])
    if len(answers) != 2:
        raise ValueError(f"VideoQA returned {len(answers)} answers, expected 2")
    return f"Segment description: {answers[0]}\nAnswer to the question: {answers[1]}"

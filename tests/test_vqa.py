from video_agent.agent.vqa import answer_about_segment, vqa_window
from video_agent.media import Video
from video_agent.memory.preprocess import get_video_segments

from tests.fakes import FakeVideoQA, make_temporal_memory

DESCRIBE = "what is the video about?"
QUESTION = "is there a boat?"


def test_the_window_spans_the_neighbours_and_stops_at_the_grid() -> None:
    segments = make_temporal_memory(5).segments

    assert vqa_window(segments, 2, 1) == (2.0, 8.0)  # 6 seconds, the reference's
    assert vqa_window(segments, 0, 1) == (0.0, 4.0)  # clamped at the start
    assert vqa_window(segments, 4, 1) == (6.0, 10.0)  # clamped at the end
    assert vqa_window(segments, 2, 0) == (4.0, 6.0)  # no context at all
    assert vqa_window(segments, 2, 9) == (0.0, 10.0)  # more context than video


def test_the_answer_lays_out_the_description_before_the_answer(
    synthetic_video: Video,
) -> None:
    segments = get_video_segments(synthetic_video)  # 2 segments of 2 seconds
    model = FakeVideoQA()

    observation = answer_about_segment(
        synthetic_video,
        segments,
        0,
        QUESTION,
        model,
        context=1,
        sample_fps=5.0,
        describe_question=DESCRIBE,
    )

    assert observation == (
        "Segment description: answer to 'what is the video about?' over 20 frames\n"
        "Answer to the question: answer to 'is there a boat?' over 20 frames"
    )
    # Both questions are asked about the same decoded clip, description first.
    assert model.questions == [DESCRIBE, QUESTION]
    assert model.clips[0] is model.clips[1]
    # Segment 0 has no left neighbour: the window is the whole 4-second video,
    # sampled at 5 fps from a 30 fps source.
    assert [f.index for f in model.clips[0].frames] == list(range(0, 120, 6))


def test_the_window_of_the_last_segment_stops_at_the_end(
    synthetic_video: Video,
) -> None:
    segments = get_video_segments(synthetic_video)
    model = FakeVideoQA()

    answer_about_segment(
        synthetic_video,
        segments,
        1,
        QUESTION,
        model,
        context=0,
        sample_fps=5.0,
        describe_question=DESCRIBE,
    )

    assert [f.index for f in model.clips[0].frames] == list(range(60, 120, 6))


def test_the_single_style_returns_the_one_reply_verbatim(
    synthetic_video: Video,
) -> None:
    segments = get_video_segments(synthetic_video)
    model = FakeVideoQA()

    observation = answer_about_segment(
        synthetic_video,
        segments,
        0,
        QUESTION,
        model,
        context=1,
        sample_fps=5.0,
        describe_question=DESCRIBE,
        style="single",
    )

    assert observation == "answer to 'is there a boat?' over 20 frames"
    # One clip, one question: the description is part of the reply, not a call.
    assert model.questions == [QUESTION]
    assert len(model.clips) == 1
    assert [f.index for f in model.clips[0].frames] == list(range(0, 120, 6))

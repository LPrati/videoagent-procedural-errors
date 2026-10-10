"""``scripts/run_single_call.py`` offline: a synthetic video, a fake server,
real annotations.

Nothing here needs a GPU or a network. The videos are a few dozen flat frames
written with OpenCV into ``tmp_path``, each frame's colour spelling out its
own index so that a test can say *which* frames came out the other end; the
server is an object with ``complete`` that records what it was handed; the
SOPs and the task graphs are the real CaptainCook4D annotations (tests that
need them are skipped when they are absent), because a question built
against an invented recipe would only prove that the fake agrees with itself.

Three things are checked here that nothing downstream would notice going
wrong: that the resampled MP4 *declares* the rate its frames were taken at
(the model's clock), that the video comes before the text in the request
(the prefix cache, and the second question's cost), and that every recording
of a run claims one ``config_id`` (``eval.report`` refuses a run that does
not).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pytest

from tests._paths import ANNOTATIONS_DIR, SCRIPTS, load_annotations

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_single_call  # noqa: E402
from run_single_call import (  # noqa: E402
    ARM_CAPTIONS,
    ARM_VIDEO,
    QUESTION_GLOBAL,
    QUESTION_STEPS,
    Completion,
    OneShotClient,
    answer_key,
    build_text_content,
    build_video_content,
    encode_parts,
    format_fps,
    is_retryable,
    parse_global_answer,
    preamble_of,
    sample_plan,
    spans_to_segments,
    split_into_parts,
)

from video_agent.agent.transcript import Transcript  # noqa: E402
from video_agent.eval.report import (  # noqa: E402
    evaluate_run,
    read_verdicts,
    record_segments,
)
from video_agent.task.run import RunStore  # noqa: E402
from video_agent.media import Video, open_video  # noqa: E402
from video_agent.task.annotations import AnnotationSet  # noqa: E402

ERROR_RECORDING = "10_26"
NORMAL_RECORDING = "9_13"
"""Two real recordings, one with errors and one without; their SOPs have
17 and 11 steps, which is what makes an answer covering only some of them a
test of ``normalize`` and not of the fake."""

VIDEO_FPS = 10.0
LONG_FRAMES = 80  # 8 s -> 4 segments of 2 s
SHORT_FRAMES = 60  # 6 s -> 3 segments
FRAME_W, FRAME_H = 32, 24
LOW_STEP, HIGH_STEP, LOW_BASE = 30, 25, 8
"""Frame ``i`` of a synthetic video is a solid colour carrying its own index:
``i % 8`` in one channel and ``i // 8`` in another, spaced far enough apart
that both survive two lossy MP4 encodings (see ``recovered_indices``)."""

PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
"""Any ``{placeholder}`` a template could have left behind.

Lower case and underscores only, so the JSON schema quoted in the question
(``{"steps": [...]}``, ``{"followed": "yes"|"no", ...}``) does not match: its
braces are followed by a quote, and its inner names are quoted too."""


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def annotations() -> AnnotationSet:
    return load_annotations()


def index_frame(index: int) -> np.ndarray:
    """A solid colour that says which frame of the source this is."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :, 0] = (index % LOW_BASE) * LOW_STEP
    frame[:, :, 1] = (index // LOW_BASE) * HIGH_STEP
    return frame


def write_video(path: Path, *, frame_count: int, fps: float = VIDEO_FPS) -> Video:
    """A tiny MP4 whose frames carry their own index; see ``index_frame``."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_W, FRAME_H)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(index_frame(index))
    writer.release()
    return open_video(path)


def recovered_indices(path: Path) -> list[int]:
    """Which source frames an MP4 holds, read back out of their colours.

    Rounded rather than compared exactly: MP4 is lossy and a flat colour
    comes back a few levels off, while the two digits it is written in are
    25 and 30 levels apart — so the reading is exact where it matters.
    """
    capture = cv2.VideoCapture(str(path))
    indices: list[int] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                return indices
            low = round(float(frame[:, :, 0].mean()) / LOW_STEP)
            high = round(float(frame[:, :, 1].mean()) / HIGH_STEP)
            indices.append(high * LOW_BASE + low)
    finally:
        capture.release()


class FakeClient:
    """Stands in for ``OneShotClient``: records the calls, replies by question.

    The reply is chosen from the *last* text item of the content, which is
    the question — so a test that broke the ordering of the message would get
    the wrong answer here before it got the wrong number anywhere else.
    """

    def __init__(
        self,
        *,
        steps: Any = "",
        global_: Any = '{"followed": "yes", "justification": "all good"}',
        tokens_in: int = 1234,
        tokens_out: int = 56,
        finish_reason: str | None = "stop",
    ) -> None:
        self.steps = steps
        self.global_ = global_
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.finish_reason = finish_reason
        self.calls: list[list[dict]] = []

    def question_of(self, content: Sequence[dict]) -> str:
        texts = [item["text"] for item in content if item["type"] == "text"]
        return texts[-1] if texts else ""

    def complete(self, content: Sequence[dict]) -> Completion:
        self.calls.append([dict(item) for item in content])
        question = self.question_of(content)
        reply = (
            self.global_
            if "Was the recipe followed correctly" in question
            else self.steps
        )
        if isinstance(reply, BaseException):
            raise reply
        return Completion(
            text=str(reply),
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            latency_s=0.5,
            model="fake/model",
            finish_reason=self.finish_reason,
        )


def steps_answer(sop, *, key: str = "span_s", spans: dict | None = None) -> str:
    """A Final Answer covering every step of ``sop``, anchored with ``key``."""
    spans = spans or {}
    entries = []
    for position, (step_id, _) in enumerate(sop.steps):
        entries.append(
            {
                "step_id": step_id,
                "executed": "yes",
                key: spans.get(step_id, [position, position + 1]),
                "order": "ok",
                "violated_prerequisite": None,
            }
        )
    return "one line per step\nFinal Answer: " + json.dumps({"steps": entries})


@pytest.fixture
def world(tmp_path: Path) -> dict:
    """A split of two recordings, their videos and an empty artifacts root."""
    videos = tmp_path / "videos"
    videos.mkdir()
    artifacts = tmp_path / "artifacts"
    out = tmp_path / "runs"
    write_video(videos / f"{ERROR_RECORDING}_360p.mp4", frame_count=LONG_FRAMES)
    write_video(videos / f"{NORMAL_RECORDING}_360p.mp4", frame_count=SHORT_FRAMES)
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps({"error": [ERROR_RECORDING], "normal": [NORMAL_RECORDING]}),
        encoding="utf-8",
    )
    return {
        "split": split,
        "videos": videos,
        "artifacts": artifacts,
        "out": out,
        "tmp": tmp_path,
    }


def write_captions(world: dict, recording_id: str, captions: Sequence[str]) -> None:
    directory = world["artifacts"] / f"{recording_id}_360p"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "captions.abst_ctx.json").write_text(
        json.dumps(list(captions)), encoding="utf-8"
    )


def argv(
    world: dict, *extra: str, arm: str = ARM_VIDEO, run_name: str = "t"
) -> list[str]:
    base = [
        "--split",
        str(world["split"]),
        "--arm",
        arm,
        "--out",
        str(world["out"]),
        "--run-name",
        run_name,
        "--artifacts-root",
        str(world["artifacts"]),
        "--videos-dir",
        str(world["videos"]),
        "--annotations-dir",
        str(ANNOTATIONS_DIR),
    ]
    if arm == ARM_VIDEO:
        base += ["--fps", "1", "--width", str(FRAME_W), "--height", str(FRAME_H)]
    return base + list(extra)


def load(
    world: dict, recording_id: str, name: str = "run.json", run: str = "t"
) -> dict:
    return json.loads(
        (world["out"] / run / recording_id / name).read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------
# The sampling plan
# --------------------------------------------------------------------------


def test_sample_plan_spends_the_expected_budget():
    """At 448x256 and 1 fps: 56 visual tokens per second of video."""
    plan = sample_plan(600.0, 1.0, 448, 256, 110_000)
    assert plan.tokens_per_pair == 14 * 8
    assert plan.fps_effective == 1.0
    assert plan.n_frames == 600
    assert plan.est_visual_tokens == 300 * 112
    assert plan.timestamps_s[:3] == (0.0, 1.0, 2.0)
    assert plan.timestamps_s[-1] == 599.0


def test_sample_plan_drops_an_odd_last_frame():
    """The processor merges frames two by two; an odd tail costs a whole pair."""
    plan = sample_plan(7.4, 1.0, 64, 64, 10_000)
    assert plan.n_frames == 8  # 8 frames fall inside 7.4 s; the 9th does not
    assert plan.timestamps_s == tuple(float(k) for k in range(8))


def test_sample_plan_thins_a_recording_that_would_blow_the_budget():
    """A 2186 s recording at 1 fps and 448x256 does not fit 110,000 tokens."""
    budget = 110_000
    plan = sample_plan(2186.0, 1.0, 448, 256, budget)
    assert plan.fps_requested == 1.0
    assert plan.fps_effective < 1.0
    assert plan.est_visual_tokens <= budget
    # Thinned in proportion and not to the floor: the budget is nearly spent.
    assert plan.est_visual_tokens > 0.98 * budget
    assert plan.n_frames % 2 == 0


@pytest.mark.parametrize("duration_s", [263.5, 564.0, 1357.0, 2186.0])
@pytest.mark.parametrize("fps", [0.5, 1.0, 2.0, 4.0])
def test_sample_plan_never_exceeds_its_budget(duration_s: float, fps: float):
    plan = sample_plan(duration_s, fps, 384, 224, 110_000)
    assert plan.est_visual_tokens <= 110_000
    assert plan.n_frames % 2 == 0
    assert plan.fps_effective <= fps
    assert all(t < duration_s for t in plan.timestamps_s)


def test_sample_plan_rates_are_representable_in_a_container():
    """A thinned rate is quantized *before* the timestamps are picked."""
    plan = sample_plan(2186.0, 1.0, 448, 256, 110_000)
    assert plan.fps_effective == round(plan.fps_effective, run_single_call.FPS_DECIMALS)
    assert plan.timestamps_s[1] == 1.0 / plan.fps_effective


def test_sample_plan_refuses_an_impossible_budget():
    with pytest.raises(ValueError, match="cannot fit a budget"):
        sample_plan(3600.0, 4.0, 640, 352, 10)


def test_sample_plan_refuses_a_video_shorter_than_two_frames():
    with pytest.raises(ValueError, match="at least two"):
        sample_plan(1.5, 0.5, 64, 64, 10_000)


def test_split_into_parts_is_one_part_by_default():
    plan = sample_plan(10.0, 1.0, 64, 64, 10_000)
    (part,) = split_into_parts(plan, None)
    assert part.index == 1
    assert part.timestamps_s == plan.timestamps_s
    assert (part.start_s, part.end_s) == (0.0, 10.0)


def test_split_into_parts_keeps_absolute_times_and_even_counts():
    plan = sample_plan(10.0, 1.0, 64, 64, 10_000)
    parts = split_into_parts(plan, 4.0)
    assert [part.index for part in parts] == [1, 2, 3]
    assert parts[0].timestamps_s == (0.0, 1.0, 2.0, 3.0)
    assert parts[1].timestamps_s == (4.0, 5.0, 6.0, 7.0)
    # 8 s and 9 s are two frames, which is even and therefore kept whole.
    assert parts[2].timestamps_s == (8.0, 9.0)
    assert parts[2].end_s == 10.0
    assert "from 4 s to 8 s" in parts[1].label()


def test_split_into_parts_drops_an_odd_frame_of_a_part():
    """Each part is its own video input, so each needs an even frame count."""
    plan = sample_plan(9.0, 1.0, 64, 64, 10_000)  # 8 frames, 0..7 s
    parts = split_into_parts(plan, 3.0)
    assert [part.timestamps_s for part in parts] == [
        (0.0, 1.0),  # 2 s is dropped
        (3.0, 4.0),  # 5 s is dropped
        (6.0, 7.0),
    ]


# --------------------------------------------------------------------------
# The resampled MP4 — the model's clock
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fps", [4.0, 2.0, 1.0, 0.5])
def test_resampled_mp4_declares_the_rate_its_frames_were_taken_at(
    tmp_path: Path, fps: float
):
    """The container's rate is the sampling rate, fractional rates included.

    A container that rounded a fractional rate to 1 fps would move every
    anchor of a 0.5 fps run by a factor of two, and nothing in the record
    would say so — hence reading the file back instead of trusting the writer.
    """
    video = write_video(tmp_path / "source.mp4", frame_count=LONG_FRAMES)
    plan = sample_plan(video.duration_s, fps, FRAME_W, FRAME_H, 10_000)
    encoded = encode_parts(
        video,
        split_into_parts(plan, None),
        width=FRAME_W,
        height=FRAME_H,
        fps=plan.fps_effective,
        directory=tmp_path,
    )
    (part,) = encoded.parts
    assert encoded.n_frames == plan.n_frames
    assert part.size_bytes > 0
    capture = cv2.VideoCapture(str(part.path))
    try:
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(fps)
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == plan.n_frames
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == FRAME_W
    finally:
        capture.release()


def test_resampled_mp4_holds_the_planned_frames(tmp_path: Path):
    """One frame per second means source frames 0, 10, 20, ... at 10 fps."""
    video = write_video(tmp_path / "source.mp4", frame_count=LONG_FRAMES)
    plan = sample_plan(video.duration_s, 1.0, FRAME_W, FRAME_H, 10_000)
    encoded = encode_parts(
        video,
        split_into_parts(plan, None),
        width=FRAME_W,
        height=FRAME_H,
        fps=plan.fps_effective,
        directory=tmp_path,
    )
    (part,) = encoded.parts
    assert recovered_indices(part.path) == [0, 10, 20, 30, 40, 50, 60, 70]


def test_resampled_mp4_is_one_file_per_part(tmp_path: Path):
    video = write_video(tmp_path / "source.mp4", frame_count=LONG_FRAMES)
    plan = sample_plan(video.duration_s, 1.0, FRAME_W, FRAME_H, 10_000)
    encoded = encode_parts(
        video,
        split_into_parts(plan, 4.0),
        width=FRAME_W,
        height=FRAME_H,
        fps=plan.fps_effective,
        directory=tmp_path,
    )
    assert [entry.n_frames for entry in encoded.parts] == [4, 4]
    assert encoded.n_frames == 8
    assert len({entry.path for entry in encoded.parts}) == 2
    assert recovered_indices(encoded.parts[1].path) == [40, 50, 60, 70]


# --------------------------------------------------------------------------
# Seconds into segments
# --------------------------------------------------------------------------


def test_spans_to_segments_floors_onto_the_two_second_grid():
    payload = {"steps": [{"step_id": 1, "executed": "yes", "span_s": [3.5, 9.0]}]}
    converted = spans_to_segments(payload, 10)
    assert converted["steps"][0]["segments"] == [1, 4]
    assert converted["steps"][0]["span_s"] == [3.5, 9.0]


def test_spans_to_segments_clamps_to_the_grid():
    payload = {
        "steps": [
            {"step_id": 1, "span_s": [-5, 4]},
            {"step_id": 2, "span_s": [100, 500]},
        ]
    }
    converted = spans_to_segments(payload, 5)
    assert converted["steps"][0]["segments"] == [0, 2]
    assert converted["steps"][1]["segments"] == [4, 4]


def test_spans_to_segments_refuses_a_reversed_or_unreadable_span():
    payload = {
        "steps": [
            {"step_id": 1, "span_s": [40, 4]},
            {"step_id": 2, "span_s": None},
            {"step_id": 3, "span_s": "from 4 to 8"},
            {"step_id": 4, "span_s": [4]},
            {"step_id": 5, "span_s": [True, False]},
        ]
    }
    converted = spans_to_segments(payload, 10)
    assert [entry["segments"] for entry in converted["steps"]] == [None] * 5
    # The raw span survives beside the refusal, for the error analysis.
    assert converted["steps"][0]["span_s"] == [40, 4]


def test_spans_to_segments_converts_the_violated_prerequisite():
    payload = {
        "steps": [
            {
                "step_id": 1,
                "executed": "yes",
                "span_s": [10, 12],
                "order": "violation",
                "violated_prerequisite": {"step_id": 7, "span_s": [30.0, 31.5]},
            }
        ]
    }
    converted = spans_to_segments(payload, 100)
    prerequisite = converted["steps"][0]["violated_prerequisite"]
    assert prerequisite["segments"] == [15, 15]
    assert prerequisite["step_id"] == 7


def test_spans_to_segments_reads_a_segments_key_as_seconds_too():
    """The model answering in the packaged key while being asked for seconds."""
    payload = {"steps": [{"step_id": 1, "segments": [10, 13]}]}
    converted = spans_to_segments(payload, 100)
    assert converted["steps"][0]["segments"] == [5, 6]
    assert converted["steps"][0]["span_s"] == [10, 13]


def test_spans_to_segments_leaves_an_entry_without_an_anchor_alone():
    payload = {"steps": [{"step_id": 1, "executed": "no"}, "junk"]}
    converted = spans_to_segments(payload, 10)
    assert converted["steps"][0] == {"step_id": 1, "executed": "no"}
    assert converted["steps"][1] == "junk"


def test_spans_to_segments_does_not_touch_its_input():
    payload = {
        "steps": [
            {
                "step_id": 1,
                "span_s": [2, 4],
                "violated_prerequisite": {"step_id": 2, "span_s": [0, 1]},
            }
        ]
    }
    before = json.dumps(payload, sort_keys=True)
    spans_to_segments(payload, 10)
    assert json.dumps(payload, sort_keys=True) == before


def test_spans_to_segments_survives_an_answer_without_steps():
    assert spans_to_segments({"steps": "nope"}, 10) == {"steps": "nope"}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"steps": [{"span_s": [1, 2]}]}, "span_s"),
        ({"steps": [{"segments": [1, 2]}]}, "segments"),
        ({"steps": [{"span_s": [1, 2]}, {"segments": [1, 2]}]}, "mixed"),
        ({"steps": [{"executed": "no"}]}, None),
        (None, None),
    ],
)
def test_answer_key_names_what_the_model_wrote(payload, expected):
    assert answer_key(payload) == expected


# --------------------------------------------------------------------------
# The global answer
# --------------------------------------------------------------------------


def test_parse_global_answer_reads_the_verdict_and_the_justification():
    answer = parse_global_answer(
        'Sure.\n{"followed": "no", "justification": "step 3 was skipped"}'
    )
    assert (answer.followed, answer.status) == ("no", "ok")
    assert answer.justification == "step 3 was skipped"


def test_parse_global_answer_normalizes_the_case_and_takes_the_last_object():
    answer = parse_global_answer(
        '```json\n{"followed": "YES"}\n```\nWait. {"followed": " No "}'
    )
    assert answer.followed == "no"
    assert answer.justification is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "The recipe was followed.",
        '{"followed": "maybe"}',
        '{"followed": true}',
        '{"verdict": "no"}',
    ],
)
def test_parse_global_answer_abstains_on_anything_else(text):
    answer = parse_global_answer(text)
    assert answer.followed is None
    assert answer.status == "parse_failed"


# --------------------------------------------------------------------------
# The questions
# --------------------------------------------------------------------------


KNOWN_PLACEHOLDERS = frozenset(
    run_single_call.VIDEO_PLACEHOLDERS
    + run_single_call.CAPTIONS_PLACEHOLDERS
    + run_single_call.GLOBAL_PLACEHOLDERS
)


def leftover(text: str) -> list[str]:
    """Placeholder names a rendered question still carries.

    Two nets: anything shaped like a placeholder, and the names this script
    knows by heart. Rendering is ``str.replace``, which fails silently, so a
    question that reached the model with ``{steps}`` still in it would read
    as a model that ignored the recipe.
    """
    shaped = set(PLACEHOLDER.findall(text))
    named = {name for name in KNOWN_PLACEHOLDERS if name in text}
    return sorted(shaped | named)


def test_the_video_question_renders_without_a_placeholder_left(annotations):
    sop, graph = annotations.for_recording(ERROR_RECORDING)
    template = run_single_call.read_template(
        run_single_call.PROMPTS_DIR / run_single_call.VIDEO_TEMPLATE_FILE, run_single_call.VIDEO_PLACEHOLDERS
    )
    question = run_single_call.build_video_question(
        sop, graph, template, fps=0.5, duration_s=419.2
    )
    assert leftover(question) == []
    assert "sampled at 0.5 frames per second" in question
    assert "it is 419 seconds long" in question
    assert "span_s" in question and '"segments"' not in question
    assert f"step {sop.steps[0][0]}: " in question


def test_the_captions_question_renders_every_segment(annotations):
    sop, graph = annotations.for_recording(NORMAL_RECORDING)
    template = run_single_call.read_template(
        run_single_call.PROMPTS_DIR / run_single_call.CAPTIONS_TEMPLATE_FILE,
        run_single_call.CAPTIONS_PLACEHOLDERS,
    )
    segments = run_single_call.get_video_segments(
        Video(path=Path("x"), fps=10.0, frame_count=60)
    )
    question = run_single_call.build_captions_question(
        sop,
        graph,
        template,
        segments=segments,
        captions=["a hand\nopens a jar", "", "  "],
    )
    assert leftover(question) == []
    assert "sliced into 3 segments" in question
    assert "[0, 0-2 s] a hand opens a jar" in question
    assert "[1, 2-4 s] (no caption)" in question
    assert "[2, 4-6 s] (no caption)" in question
    assert '"segments"' in question and "span_s" not in question


def test_the_global_question_opens_with_the_arm_preamble(annotations):
    """Both arms: the yes/no question is asked of the same evidence."""
    sop, graph = annotations.for_recording(ERROR_RECORDING)
    templates = {
        ARM_VIDEO: run_single_call.read_template(
            run_single_call.PROMPTS_DIR / run_single_call.VIDEO_TEMPLATE_FILE,
            run_single_call.VIDEO_PLACEHOLDERS,
        ),
        ARM_CAPTIONS: run_single_call.read_template(
            run_single_call.PROMPTS_DIR / run_single_call.CAPTIONS_TEMPLATE_FILE,
            run_single_call.CAPTIONS_PLACEHOLDERS,
        ),
        QUESTION_GLOBAL: run_single_call.read_template(
            run_single_call.PROMPTS_DIR / run_single_call.GLOBAL_TEMPLATE_FILE,
            run_single_call.GLOBAL_PLACEHOLDERS,
        ),
    }
    video_question = run_single_call.build_video_question(
        sop, graph, templates[ARM_VIDEO], fps=1.0, duration_s=100.0
    )
    segments = run_single_call.get_video_segments(
        Video(path=Path("x"), fps=10.0, frame_count=60)
    )
    captions_question = run_single_call.build_captions_question(
        sop, graph, templates[ARM_CAPTIONS], segments=segments, captions=["a", "b", "c"]
    )

    from_video = run_single_call.build_global_question(
        preamble_of(video_question), sop, graph, templates[QUESTION_GLOBAL]
    )
    from_captions = run_single_call.build_global_question(
        preamble_of(captions_question), sop, graph, templates[QUESTION_GLOBAL]
    )
    for question in (from_video, from_captions):
        assert leftover(question) == []
        assert question.startswith("The video shows one person making")
        assert "Was the recipe followed correctly" in question
        assert '{"followed": "yes"|"no"' in question
        # The schema of the per-step answer must not follow the preamble in.
        assert "step_id" not in question
    assert "sampled at 1 frames per second" in from_video
    # In the captions arm the "first paragraph" is the caption block too.
    assert "[2, 4-6 s] c" in from_captions


def test_the_preamble_of_a_question_without_the_marker_is_refused():
    with pytest.raises(ValueError, match="These are the steps"):
        preamble_of("no steps here")


def test_the_parts_note_reaches_both_questions(annotations):
    sop, graph = annotations.for_recording(NORMAL_RECORDING)
    template = run_single_call.read_template(
        run_single_call.PROMPTS_DIR / run_single_call.VIDEO_TEMPLATE_FILE, run_single_call.VIDEO_PLACEHOLDERS
    )
    question = run_single_call.with_parts_note(
        run_single_call.build_video_question(
            sop, graph, template, fps=1.0, duration_s=300.0
        ),
        3,
    )
    global_template = run_single_call.read_template(
        run_single_call.PROMPTS_DIR / run_single_call.GLOBAL_TEMPLATE_FILE, run_single_call.GLOBAL_PLACEHOLDERS
    )
    global_question = run_single_call.build_global_question(
        preamble_of(question), sop, graph, global_template
    )
    for text in (question, global_question):
        assert "sent in 3 consecutive parts" in text
        assert "not from the start of a part" in text
        assert leftover(text) == []
    assert question.index("consecutive parts") < question.index(
        "These are the steps of the recipe:"
    )


@pytest.mark.parametrize(
    ("fps", "expected"), [(1.0, "1"), (0.5, "0.5"), (0.898, "0.898"), (4.0, "4")]
)
def test_format_fps_drops_trailing_zeros(fps, expected):
    assert format_fps(fps) == expected


# --------------------------------------------------------------------------
# The request
# --------------------------------------------------------------------------


def test_the_video_comes_before_the_text(tmp_path: Path):
    """The prefix the second question of the recording reuses."""
    video = write_video(tmp_path / "source.mp4", frame_count=LONG_FRAMES)
    plan = sample_plan(video.duration_s, 1.0, FRAME_W, FRAME_H, 10_000)
    encoded = encode_parts(
        video,
        split_into_parts(plan, None),
        width=FRAME_W,
        height=FRAME_H,
        fps=plan.fps_effective,
        directory=tmp_path,
    )
    content = build_video_content(encoded, "the question")
    assert [item["type"] for item in content] == ["video_url", "text"]
    assert content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
    assert content[-1]["text"] == "the question"


def test_every_part_is_announced_before_its_frames(tmp_path: Path):
    video = write_video(tmp_path / "source.mp4", frame_count=LONG_FRAMES)
    plan = sample_plan(video.duration_s, 1.0, FRAME_W, FRAME_H, 10_000)
    encoded = encode_parts(
        video,
        split_into_parts(plan, 4.0),
        width=FRAME_W,
        height=FRAME_H,
        fps=plan.fps_effective,
        directory=tmp_path,
    )
    content = build_video_content(encoded, "the question")
    assert [item["type"] for item in content] == [
        "text",
        "video_url",
        "text",
        "video_url",
        "text",
    ]
    assert content[0]["text"].startswith("Part 1: from 0 s to 4 s")
    assert content[2]["text"].startswith("Part 2: from 4 s to 8 s")


def test_a_captions_request_is_text_only():
    assert build_text_content("q") == [{"type": "text", "text": "q"}]


def test_the_request_body_carries_the_switches_of_the_arm():
    client = OneShotClient(
        model="Qwen/Qwen3.5-9B",
        temperature=0.0,
        seed=7,
        max_tokens=8192,
        extra_body={**run_single_call.NO_THINKING_BODY, **run_single_call.VIDEO_BODY},
        client=object(),
    )
    body = client.request([{"type": "text", "text": "q"}])
    assert body["model"] == "Qwen/Qwen3.5-9B"
    assert body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "q"}]}
    ]
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 8192
    assert body["seed"] == 7
    assert body["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["extra_body"]["mm_processor_kwargs"] == {"do_sample_frames": False}


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = _Message(content)
        self.finish_reason = finish_reason


class _Usage:
    prompt_tokens = 80_000
    completion_tokens = 900


class _Response:
    model = "Qwen/Qwen3.5-9B"
    usage = _Usage()

    def __init__(self, content: str = "hi", finish_reason: str = "stop") -> None:
        self.choices = [_Choice(content, finish_reason)]


class _Completions:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def create(self, **_: Any):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _SDK:
    def __init__(self, outcomes: list) -> None:
        self.chat = type("Chat", (), {"completions": _Completions(outcomes)})()


class APIConnectionError(Exception):
    """Same name as the SDK's, which is what ``is_retryable`` looks at."""


class BadRequestError(Exception):
    status_code = 400


class InternalError(Exception):
    status_code = 503


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (APIConnectionError(), True),
        (InternalError(), True),
        (BadRequestError(), False),
        (ValueError("nope"), False),
    ],
)
def test_is_retryable_only_where_retrying_could_help(error, expected):
    assert is_retryable(error) is expected


def test_the_client_retries_a_dropped_connection_once():
    sdk = _SDK([APIConnectionError(), _Response("hello")])
    client = OneShotClient(client=sdk, retry_wait_s=0.0)
    completion = client.complete([{"type": "text", "text": "q"}])
    assert completion.text == "hello"
    assert completion.tokens_in == 80_000
    assert completion.tokens_out == 900
    assert completion.finish_reason == "stop"
    assert sdk.chat.completions.calls == 2


def test_the_client_does_not_retry_a_bad_request():
    sdk = _SDK([BadRequestError(), _Response()])
    client = OneShotClient(client=sdk, retry_wait_s=0.0)
    with pytest.raises(BadRequestError):
        client.complete([{"type": "text", "text": "q"}])
    assert sdk.chat.completions.calls == 1


# --------------------------------------------------------------------------
# The records
# --------------------------------------------------------------------------


def test_a_video_run_writes_records_the_evaluation_can_read(world, annotations):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    client = FakeClient(steps=steps_answer(sop, spans={sop.steps[0][0]: [2.0, 5.9]}))
    assert run_single_call.main(argv(world), client=client) == run_single_call.EXIT_OK

    record = load(world, ERROR_RECORDING)
    assert set(record) == {
        "schema_version",
        "config",
        "recording_id",
        "activity_id",
        "started_at",
        "finished_at",
        "wall_s",
        "status",
        "error",
        "n_segments",
        "video",
        "question",
        "final_answer_text",
        "n_answer_candidates",
        "verdicts",
        "transcript",
        "usage",
        "usage_total",
        "input",
    }
    assert record["status"] == "ok"
    assert record["activity_id"] == sop.activity_id
    assert record["n_segments"] == 4
    assert record["video"] == {
        "fps": VIDEO_FPS,
        "frame_count": LONG_FRAMES,
        "n_segments": 4,
    }
    assert record["n_answer_candidates"] == 1

    # What the evaluation does with a record, done here.
    verdicts = read_verdicts(record)
    assert verdicts.recording_id == ERROR_RECORDING
    assert len(verdicts.steps) == len(sop.steps)
    assert verdicts.status == "ok"
    assert [segment.index for segment in record_segments(record)] == [0, 1, 2, 3]
    transcript = Transcript.from_dict(record["transcript"])
    assert len(transcript.steps) == 1
    assert transcript.steps[0].turn.usage.tokens_in == 1234
    assert transcript.steps[0].action is None

    # Seconds became segment ids: [2.0, 5.9] s is the grid's segments 1 to 2.
    first = record["verdicts"][0]
    assert first["execution"] == "executado"
    assert first["anchor"]["segments"] == [1, 2]
    assert first["anchor"]["start_s"] == pytest.approx(2.0)
    assert first["anchor"]["end_s"] == pytest.approx(6.0)
    assert first["raw"]["span_s"] == [2.0, 5.9]

    assert record["usage"] == record["usage_total"]
    assert record["usage"]["llm_calls"] == 1
    assert record["input"]["n_frames"] == 8
    assert record["input"]["fps_effective"] == 1.0
    assert record["input"]["est_visual_tokens"] == 4
    assert record["input"]["mp4_bytes"] > 0
    assert record["input"]["duration_s"] == pytest.approx(8.0)
    assert record["input"]["n_captions"] is None
    assert record["input"]["answer_key"] == "span_s"
    assert record["input"]["prompt_chars"] == len(record["question"])


def test_a_whole_run_scores_without_touching_the_package(world, annotations):
    """The end of the contract: ``evaluate_run`` reads this run like any other.

    Not just the loader — the scoring, which refuses a run whose records
    claim two configurations, counts the statuses and matches every step
    instance of the annotations against the verdicts.
    """
    class PerRecipe(FakeClient):
        """Answers with the step ids of whichever recipe it was asked about."""

        def complete(self, content):
            question = self.question_of(content)
            ids = [
                int(line.split()[1].rstrip(":"))
                for line in question.splitlines()
                if line.startswith("step ") and ": " in line
            ]
            self.steps = "notes\nFinal Answer: " + json.dumps(
                {
                    "steps": [
                        {
                            "step_id": step_id,
                            "executed": "yes",
                            "span_s": [2.0 * position, 2.0 * position + 1.5],
                            "order": "ok",
                            "violated_prerequisite": None,
                        }
                        for position, step_id in enumerate(ids)
                    ]
                }
            )
            return super().complete(content)

    client = PerRecipe()
    assert run_single_call.main(argv(world), client=client) == run_single_call.EXIT_OK

    store = RunStore(world["out"], "t")
    evaluation = evaluate_run(
        store,
        [ERROR_RECORDING, NORMAL_RECORDING],
        annotations,
        n_bootstrap=20,
    )
    assert evaluation.status_counts == {"ok": 2}
    assert len(evaluation.config_ids) == 1
    assert evaluation.instances > 0
    assert evaluation.cost.mean_llm_calls == 1.0
    assert evaluation.cost.mean_tokens_in == 1234


def test_the_configuration_is_one_id_for_the_whole_run(world, annotations):
    """``eval.report`` refuses a run whose records claim two configurations."""
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    client = FakeClient(steps=steps_answer(sop))
    run_single_call.main(argv(world), client=client)

    first = load(world, ERROR_RECORDING)["config"]
    second = load(world, NORMAL_RECORDING)["config"]
    assert first == second  # different durations, one configuration
    assert first["config_id"] == second["config_id"]
    assert first["arm"] == ARM_VIDEO
    assert first["question"] == QUESTION_STEPS
    assert first["system"] == "single_call"
    assert "experiment" not in first
    assert first["chat_backend"] == "vllm"
    assert first["thinking"] is False
    assert first["seed"] == run_single_call.DEFAULT_SEED
    assert first["caption_variant"] is None
    assert (first["fps"], first["width"], first["height"]) == (1.0, FRAME_W, FRAME_H)
    # The per-recording facts live in ``input``, where they cannot rename a run.
    assert load(world, ERROR_RECORDING)["input"]["n_frames"] == 8
    assert load(world, NORMAL_RECORDING)["input"]["n_frames"] == 6

    global_config = load(world, ERROR_RECORDING, "global.json")["config"]
    assert global_config["question"] == QUESTION_GLOBAL
    assert global_config["config_id"] != first["config_id"]
    assert (
        global_config["question_template_sha256"]
        != first["question_template_sha256"]
    )


def config_kwargs(**overrides) -> dict:
    """The arguments ``build_config`` takes, with the video arm's values."""
    return {
        "arm": ARM_VIDEO,
        "question": QUESTION_GLOBAL,
        "question_template_sha256": "global-sha",
        "model": "Qwen/Qwen3.5-9B",
        "temperature": 0.0,
        "seed": 20260919,
        "max_tokens": 8192,
        "fps": 1.0,
        "width": 448,
        "height": 256,
        "visual_token_budget": 56_000,
        "video_parts_s": None,
        "caption_variant": None,
        **overrides,
    }


def test_the_global_configuration_names_both_templates_it_is_made_of(
    world, annotations
):
    """The global question quotes the arm's preamble, so it has two digests.

    Without this, rewording the first paragraph of the video arm would change
    what the model is asked and leave the global configuration answering to
    the same id, the mix-up ``config_id`` exists to prevent.
    """
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    run_single_call.main(argv(world), client=FakeClient(steps=steps_answer(sop)))

    steps_config = load(world, ERROR_RECORDING)["config"]
    global_config = load(world, ERROR_RECORDING, "global.json")["config"]
    # The preamble of the global question came from the arm's own template.
    assert (
        global_config["preamble_template_sha256"]
        == steps_config["question_template_sha256"]
    )
    assert global_config["preamble_template_sha256"] == run_single_call.template_sha256(
        (run_single_call.PROMPTS_DIR / run_single_call.VIDEO_TEMPLATE_FILE).read_text(encoding="utf-8")
    )
    # The per-step question is made of one template and says so.
    assert "preamble_template_sha256" not in steps_config

    # Moving the arm's wording moves the global id; so does moving its own.
    arm_sha256 = global_config["preamble_template_sha256"]
    here = run_single_call.build_config(**config_kwargs(preamble_template_sha256=arm_sha256))
    reworded_arm = run_single_call.build_config(
        **config_kwargs(preamble_template_sha256="another-arm-wording")
    )
    reworded_global = run_single_call.build_config(
        **config_kwargs(
            question_template_sha256="another-global-wording",
            preamble_template_sha256=global_config["preamble_template_sha256"],
        )
    )
    assert here["config_id"] != reworded_arm["config_id"]
    assert here["config_id"] != reworded_global["config_id"]


def test_the_captions_arm_puts_its_own_template_in_the_global_configuration(
    world, annotations
):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    write_captions(world, ERROR_RECORDING, ["chop", "stir", "serve", "plate"])
    run_single_call.main(
        argv(world, "--limit", "1", arm=ARM_CAPTIONS),
        client=FakeClient(steps=steps_answer(sop, key="segments")),
    )
    global_config = load(world, ERROR_RECORDING, "global.json")["config"]
    assert global_config["preamble_template_sha256"] == run_single_call.template_sha256(
        (run_single_call.PROMPTS_DIR / run_single_call.CAPTIONS_TEMPLATE_FILE).read_text(
            encoding="utf-8"
        )
    )


def test_the_global_answer_is_written_beside_the_record(world, annotations):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    client = FakeClient(
        steps=steps_answer(sop),
        global_='{"followed": "no", "justification": "the jar is never opened"}',
    )
    run_single_call.main(argv(world), client=client)

    record = load(world, ERROR_RECORDING, "global.json")
    assert set(record) == {
        "schema_version",
        "config",
        "recording_id",
        "started_at",
        "finished_at",
        "wall_s",
        "status",
        "error",
        "question",
        "answer_text",
        "followed",
        "justification",
        "input",
        "usage",
        "turn",
    }
    assert record["status"] == "ok"
    assert record["followed"] == "no"
    assert record["justification"] == "the jar is never opened"
    assert record["turn"]["usage"]["tokens_in"] == 1234
    assert record["input"]["n_frames"] == 8
    assert "Was the recipe followed correctly" in record["question"]


def test_the_global_question_is_asked_first(world, annotations):
    """It must not be written after the model gave its per-step verdicts."""
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    client = FakeClient(steps=steps_answer(sop))
    run_single_call.main(argv(world, "--limit", "1"), client=client)
    asked = [client.question_of(call) for call in client.calls]
    assert len(asked) == 2
    assert "Was the recipe followed correctly" in asked[0]
    assert "Final Answer:" in asked[1]
    # Both calls carry the same video prefix, which is what the ordering buys.
    assert client.calls[0][:-1] == client.calls[1][:-1]


def test_a_captions_run_keeps_the_answer_in_segment_ids(world, annotations):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    write_captions(world, ERROR_RECORDING, ["chop", "stir", "serve", "plate"])
    client = FakeClient(
        steps=steps_answer(sop, key="segments", spans={sop.steps[0][0]: [1, 2]})
    )
    assert (
        run_single_call.main(
            argv(world, "--limit", "1", arm=ARM_CAPTIONS),
            client=client,
        )
        == run_single_call.EXIT_OK
    )
    record = load(world, ERROR_RECORDING)
    assert record["status"] == "ok"
    assert record["n_segments"] == 4
    assert record["config"]["caption_variant"] == "abst_ctx"
    assert record["config"]["fps"] is None
    assert record["input"]["n_captions"] == 4
    assert record["input"]["n_frames"] is None
    assert record["input"]["answer_key"] == "segments"
    # Segment ids stay segment ids in this arm: no conversion happens.
    assert record["verdicts"][0]["anchor"]["segments"] == [1, 2]
    assert "[1, 2-4 s] stir" in record["question"]
    assert all(item["type"] == "text" for item in client.calls[0])


def test_an_unreadable_answer_is_parse_failed(world, annotations):
    client = FakeClient(steps="I could not watch the video.")
    run_single_call.main(argv(world, "--limit", "1"), client=client)
    record = load(world, ERROR_RECORDING)
    assert record["status"] == "parse_failed"
    assert record["error"] is None
    assert {verdict["execution"] for verdict in record["verdicts"]} == {"indeterminado"}
    assert record["transcript"] is not None


def test_a_truncated_answer_is_length(world, annotations):
    client = FakeClient(
        steps='Final Answer: {"steps": [{"step_id": 1,', finish_reason="length"
    )
    run_single_call.main(argv(world, "--limit", "1"), client=client)
    record = load(world, ERROR_RECORDING)
    assert record["status"] == "length"
    assert record["transcript"]["steps"][0]["turn"]["finish_reason"] == "length"


def test_a_truncated_answer_that_still_parses_is_ok(world, annotations):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    client = FakeClient(steps=steps_answer(sop), finish_reason="length")
    run_single_call.main(argv(world, "--limit", "1"), client=client)
    assert load(world, ERROR_RECORDING)["status"] == "ok"


def test_an_exception_becomes_a_record_and_the_batch_goes_on(world, annotations):
    """One broken recording leaves an error record and the batch goes on."""
    calls: list[str] = []

    class Exploding(FakeClient):
        def complete(self, content):
            calls.append("x")
            if len(calls) <= 2:  # both questions of the first recording
                raise ValueError("the server exploded")
            return super().complete(content)

    sop, _ = annotations.for_recording(NORMAL_RECORDING)
    client = Exploding(steps=steps_answer(sop))
    # Exit 3 and not 1: a calling script must be able to tell a batch that
    # finished with failed recordings from a crash.
    assert run_single_call.main(argv(world), client=client) == run_single_call.EXIT_FAILED_RECORDINGS

    broken = load(world, ERROR_RECORDING)
    assert broken["status"] == "error"
    assert "ValueError: the server exploded" in broken["error"]
    assert broken["transcript"] is None
    assert broken["usage"]["llm_calls"] == 0
    assert {verdict["execution"] for verdict in broken["verdicts"]} == {"indeterminado"}
    assert read_verdicts(broken).status == "error"
    # The video block survives a failed call, so the grid is still rebuildable.
    assert record_segments(broken) is not None
    assert load(world, ERROR_RECORDING, "global.json")["status"] == "error"
    assert load(world, NORMAL_RECORDING)["status"] == "ok"


def test_a_missing_video_becomes_a_record_too(world, annotations):
    (world["videos"] / f"{ERROR_RECORDING}_360p.mp4").unlink()
    sop, _ = annotations.for_recording(NORMAL_RECORDING)
    client = FakeClient(steps=steps_answer(sop))
    assert run_single_call.main(argv(world), client=client) == run_single_call.EXIT_FAILED_RECORDINGS
    broken = load(world, ERROR_RECORDING)
    assert broken["status"] == "error"
    assert broken["error"]
    assert load(world, NORMAL_RECORDING)["status"] == "ok"


# --------------------------------------------------------------------------
# Selection: resuming, sharding, which questions
# --------------------------------------------------------------------------


def test_a_second_pass_skips_what_is_already_answered(world, annotations):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    first = FakeClient(steps=steps_answer(sop))
    run_single_call.main(argv(world), client=first)
    assert len(first.calls) == 4  # two recordings, two questions each

    again = FakeClient(steps=steps_answer(sop))
    run_single_call.main(argv(world), client=again)
    assert again.calls == []

    forced = FakeClient(steps=steps_answer(sop))
    run_single_call.main(argv(world, "--force"), client=forced)
    assert len(forced.calls) == 4


def test_only_the_missing_question_is_asked_again(world, annotations):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    client = FakeClient(steps=steps_answer(sop))
    run_single_call.main(argv(world, "--questions", "global"), client=client)
    assert len(client.calls) == 2
    assert not (world["out"] / "t" / ERROR_RECORDING / "run.json").exists()

    second = FakeClient(steps=steps_answer(sop))
    run_single_call.main(argv(world), client=second)
    assert len(second.calls) == 2
    assert all(
        "Final Answer:" in second.question_of(call) for call in second.calls
    )


def test_retry_errors_redoes_only_the_failed_question(world, annotations):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    broken = FakeClient(steps=ValueError("boom"))
    run_single_call.main(argv(world), client=broken)
    assert load(world, ERROR_RECORDING)["status"] == "error"
    assert load(world, ERROR_RECORDING, "global.json")["status"] == "ok"

    retried = FakeClient(steps=steps_answer(sop))
    run_single_call.main(argv(world, "--retry-errors"), client=retried)
    assert len(retried.calls) == 2  # the per-step question of both recordings
    assert all("Final Answer:" in retried.question_of(c) for c in retried.calls)
    assert load(world, ERROR_RECORDING)["status"] == "ok"


def test_shards_split_the_sorted_split(world, annotations):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    client = FakeClient(steps=steps_answer(sop))
    run_single_call.main(argv(world, "--shard", "0/2", "--questions", "steps"), client=client)
    # sorted(["10_26", "9_13"]) == ["10_26", "9_13"]; shard 0 of 2 is the first.
    assert (world["out"] / "t" / ERROR_RECORDING / "run.json").exists()
    assert not (world["out"] / "t" / NORMAL_RECORDING / "run.json").exists()

    other = FakeClient(steps=steps_answer(sop))
    run_single_call.main(argv(world, "--shard", "1/2", "--questions", "steps"), client=other)
    assert (world["out"] / "t" / NORMAL_RECORDING / "run.json").exists()


def test_a_bad_shard_is_refused(world, annotations):
    with pytest.raises(ValueError, match="not a valid I/N"):
        run_single_call.main(argv(world, "--shard", "2/2"), client=FakeClient())


def test_dry_run_writes_nothing_even_when_a_recording_cannot_be_built(
    world, capsys, annotations
):
    """A rehearsal that left an error record would poison the real run.

    The resume check reads a record as an answer, so the run that followed
    would skip exactly the recording the rehearsal could not build. The
    failure is still counted, and the exit code still says so.
    """
    (world["videos"] / f"{ERROR_RECORDING}_360p.mp4").unlink()
    code = run_single_call.main(argv(world, "--dry-run"), client=FakeClient())
    assert code == run_single_call.EXIT_FAILED_RECORDINGS
    assert not (world["out"] / "t" / ERROR_RECORDING).exists()
    assert f"{ERROR_RECORDING}: error" in capsys.readouterr().out


def test_dry_run_builds_everything_and_calls_nothing(world, capsys, annotations):
    client = FakeClient()
    run_single_call.main(argv(world, "--dry-run"), client=client)
    assert client.calls == []
    assert not (world["out"] / "t" / ERROR_RECORDING / "run.json").exists()
    printed = capsys.readouterr().out
    assert f"{ERROR_RECORDING}: n_frames=8 fps=1 est_visual_tokens=4" in printed
    assert "mp4_bytes=" in printed
    assert "steps_chars=" in printed and "global_chars=" in printed


def test_the_video_arm_demands_its_geometry(world):
    with pytest.raises(SystemExit):
        run_single_call.main(
            [
                "--split",
                str(world["split"]),
                "--arm",
                ARM_VIDEO,
                "--out",
                str(world["out"]),
                "--run-name",
                "t",
                "--artifacts-root",
                str(world["artifacts"]),
                "--videos-dir",
                str(world["videos"]),
            ],
            client=FakeClient(),
        )


def test_an_unknown_question_is_refused(world):
    with pytest.raises(SystemExit):
        run_single_call.main(argv(world, "--questions", "steps,summary"), client=FakeClient())


# --------------------------------------------------------------------------
# Choosing the per-step template
# --------------------------------------------------------------------------

REWORDED = "An order violation is whatever this sentence says it is."


def reworded_template(tmp: Path, arm: str) -> Path:
    """The arm's default template with one sentence added, in ``tmp``."""
    text = run_single_call.default_template_path(arm).read_text(encoding="utf-8")
    marker = "These are the ordering constraints of the recipe."
    assert marker in text
    path = tmp / f"reworded_{arm}.txt"
    path.write_text(text.replace(marker, f"{REWORDED} {marker}"), encoding="utf-8")
    return path


def test_the_packaged_templates_are_the_default(world, annotations):
    """No flag and the default file named explicitly are the same run."""
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    run_single_call.main(argv(world), client=FakeClient(steps=steps_answer(sop)))
    default = load(world, ERROR_RECORDING)["config"]
    default_global = load(world, ERROR_RECORDING, "global.json")["config"]
    assert "question_template_file" not in default
    assert "preamble_template_file" not in default_global

    explicit = run_single_call.default_template_path(ARM_VIDEO)
    run_single_call.main(
        argv(world, "--video-template", str(explicit), run_name="explicit"),
        client=FakeClient(steps=steps_answer(sop)),
    )
    named = load(world, ERROR_RECORDING, run="explicit")["config"]
    assert named["config_id"] == default["config_id"]
    assert (
        load(world, ERROR_RECORDING, "global.json", run="explicit")["config"][
            "config_id"
        ]
        == default_global["config_id"]
    )


def test_build_config_without_a_template_file_keeps_its_id():
    """The template-file keys are absent unless given, so the id is stable."""
    before = run_single_call.build_config(**config_kwargs())
    after = run_single_call.build_config(
        **config_kwargs(question_template_file=None, preamble_template_file=None)
    )
    assert before == after
    named = run_single_call.build_config(**config_kwargs(question_template_file="x.txt"))
    assert named["question_template_file"] == "x.txt"
    assert named["config_id"] != before["config_id"]


def test_a_video_template_reaches_the_question_and_the_configuration(
    world, annotations
):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    run_single_call.main(argv(world), client=FakeClient(steps=steps_answer(sop)))
    default = load(world, ERROR_RECORDING)["config"]
    default_global = load(world, ERROR_RECORDING, "global.json")["config"]

    template = reworded_template(world["tmp"], ARM_VIDEO)
    client = FakeClient(steps=steps_answer(sop))
    run_single_call.main(
        argv(world, "--video-template", str(template), run_name="new"),
        client=client,
    )
    record = load(world, ERROR_RECORDING, run="new")
    config = record["config"]
    sha256 = run_single_call.template_sha256(template.read_text(encoding="utf-8"))
    assert REWORDED in record["question"]
    assert config["question_template_sha256"] == sha256
    assert config["question_template_file"] == template.resolve().as_posix()
    assert config["config_id"] != default["config_id"]

    # The global question quotes only the preamble (before the step list), so
    # a sentence reworded after it leaves the global text as it was; the
    # configuration still names the arm's template, and its id moves.
    global_record = load(world, ERROR_RECORDING, "global.json", run="new")
    global_config = global_record["config"]
    assert REWORDED not in global_record["question"]
    assert (
        global_record["question"]
        == load(world, ERROR_RECORDING, "global.json")["question"]
    )
    assert global_config["preamble_template_sha256"] == sha256
    assert global_config["preamble_template_file"] == config["question_template_file"]
    assert global_config["config_id"] != default_global["config_id"]
    # The global template itself did not change.
    assert (
        global_config["question_template_sha256"]
        == default_global["question_template_sha256"]
    )
    steps_calls = [
        call for call in client.calls if "Final Answer:" in client.question_of(call)
    ]
    assert len(steps_calls) == 2
    assert all(REWORDED in client.question_of(call) for call in steps_calls)


def test_a_captions_template_reaches_the_question_and_the_configuration(
    world, annotations
):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    write_captions(world, ERROR_RECORDING, ["chop", "stir", "serve", "plate"])
    template = reworded_template(world["tmp"], ARM_CAPTIONS)
    run_single_call.main(
        argv(
            world,
            "--limit",
            "1",
            "--captions-template",
            str(template),
            arm=ARM_CAPTIONS,
        ),
        client=FakeClient(steps=steps_answer(sop, key="segments")),
    )
    record = load(world, ERROR_RECORDING)
    assert REWORDED in record["question"]
    assert record["config"]["question_template_sha256"] == run_single_call.template_sha256(
        template.read_text(encoding="utf-8")
    )
    assert record["config"]["question_template_file"] == template.resolve().as_posix()


def test_the_other_arm_template_does_not_touch_the_run(world, annotations):
    """A captions template given to a video run leaves the video run alone."""
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    run_single_call.main(argv(world), client=FakeClient(steps=steps_answer(sop)))
    default = load(world, ERROR_RECORDING)["config"]
    template = reworded_template(world["tmp"], ARM_CAPTIONS)
    run_single_call.main(
        argv(world, "--captions-template", str(template), run_name="other"),
        client=FakeClient(steps=steps_answer(sop)),
    )
    assert load(world, ERROR_RECORDING, run="other")["config"] == default


def test_a_template_inside_the_repository_is_named_relative_to_it():
    label = run_single_call.template_label(run_single_call.default_template_path(ARM_VIDEO))
    assert label == "prompts/single_call/video_question.txt"


@pytest.mark.parametrize(
    "flag,arm,placeholder",
    [
        ("--video-template", ARM_VIDEO, "{fps}"),
        ("--captions-template", ARM_CAPTIONS, "{captions}"),
    ],
)
def test_a_template_missing_a_placeholder_is_refused_before_anything_runs(
    world, flag, arm, placeholder
):
    text = run_single_call.default_template_path(arm).read_text(encoding="utf-8")
    broken = world["tmp"] / "broken.txt"
    broken.write_text(text.replace(placeholder, "nothing"), encoding="utf-8")
    client = FakeClient()
    with pytest.raises(SystemExit):
        run_single_call.main(argv(world, flag, str(broken), arm=arm), client=client)
    assert client.calls == []
    assert not (world["out"] / "t").exists()


# --------------------------------------------------------------------------
# The provider
# --------------------------------------------------------------------------

TEST_KEY_ENV = "RUN_SINGLE_CALL_TEST_API_KEY"


def provider_argv(world: dict, *extra: str, arm: str = ARM_CAPTIONS) -> list[str]:
    return argv(
        world,
        "--provider",
        "openai-compatible",
        "--api-key-env",
        TEST_KEY_ENV,
        "--model",
        "hosted/model",
        *extra,
        arm=arm,
    )


def built_client(monkeypatch, args) -> tuple[OneShotClient, dict]:
    """``build_client`` with the SDK client injected, and the arguments it used.

    The injected client keeps the test free of the ``openai`` package and of
    any network; the captured keyword arguments show the API key it chose.
    """
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return OneShotClient(**kwargs, client=object())

    monkeypatch.setattr(run_single_call, "OneShotClient", fake)
    return run_single_call.build_client(args), captured


def test_the_vllm_provider_is_the_default(world, monkeypatch):
    args = run_single_call.parse_args(argv(world, arm=ARM_CAPTIONS))
    assert args.provider == run_single_call.PROVIDER_VLLM
    client, _ = built_client(monkeypatch, args)
    body = client.request([{"type": "text", "text": "q"}])
    assert body["seed"] == run_single_call.DEFAULT_SEED
    assert body["extra_body"] == run_single_call.NO_THINKING_BODY


def test_the_openai_compatible_request_has_no_extra_body_and_no_seed(
    world, monkeypatch
):
    monkeypatch.setenv(TEST_KEY_ENV, "test-key")
    args = run_single_call.parse_args(provider_argv(world))
    client, kwargs = built_client(monkeypatch, args)
    body = client.request([{"type": "text", "text": "q"}])
    assert "extra_body" not in body
    assert "seed" not in body
    assert body["model"] == "hosted/model"
    assert body["temperature"] == 0.0
    assert kwargs["api_key"] == "test-key"


def test_the_openai_compatible_provider_refuses_the_video_arm(world, capsys):
    with pytest.raises(SystemExit):
        run_single_call.parse_args(provider_argv(world, arm=ARM_VIDEO))
    assert "supports only --arm captions" in capsys.readouterr().err


def test_the_openai_compatible_provider_needs_its_api_key(world, monkeypatch):
    monkeypatch.delenv(TEST_KEY_ENV, raising=False)
    args = run_single_call.parse_args(provider_argv(world))
    with pytest.raises(SystemExit, match=TEST_KEY_ENV):
        run_single_call.build_client(args)


def test_the_openai_compatible_configuration_records_no_thinking_or_seed():
    config = run_single_call.build_config(
        **config_kwargs(
            arm=ARM_CAPTIONS,
            fps=None,
            width=None,
            height=None,
            visual_token_budget=None,
            caption_variant="abst_ctx",
            provider=run_single_call.PROVIDER_OPENAI_COMPATIBLE,
        )
    )
    assert config["chat_backend"] == "openai-compatible"
    assert config["thinking"] is None
    assert config["seed"] is None
    vllm = run_single_call.build_config(
        **config_kwargs(
            arm=ARM_CAPTIONS,
            fps=None,
            width=None,
            height=None,
            visual_token_budget=None,
            caption_variant="abst_ctx",
        )
    )
    assert vllm["config_id"] != config["config_id"]


def test_an_openai_compatible_captions_run_writes_its_backend(world, annotations):
    sop, _ = annotations.for_recording(ERROR_RECORDING)
    write_captions(world, ERROR_RECORDING, ["chop", "stir", "serve", "plate"])
    client = FakeClient(steps=steps_answer(sop, key="segments"))
    assert (
        run_single_call.main(provider_argv(world, "--limit", "1"), client=client)
        == run_single_call.EXIT_OK
    )
    for name in ("run.json", "global.json"):
        config = load(world, ERROR_RECORDING, name)["config"]
        assert config["chat_backend"] == "openai-compatible"
        assert config["thinking"] is None
        assert config["seed"] is None
        assert config["orchestrator_model"] == "hosted/model"

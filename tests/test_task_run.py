"""One recording end to end: the run record it produces."""

import json
from pathlib import Path

import pytest
from tests.fakes import FakeAligner, FakeChat, FakeTextEmbedder, make_temporal_memory

from video_agent.agent.config import ReActConfig, ToolConfig
from video_agent.agent.prompts import load_prompt
from video_agent.agent.react import ReActRunner
from video_agent.agent.tools import ToolKit
from video_agent.agent.transcript import Transcript
from video_agent.media import Video
from video_agent.models.types import Usage
from video_agent.task.annotations import Sop, TaskGraph
from video_agent.task.prompt import build_question
from video_agent.task.run import (
    RUN_SCHEMA_VERSION,
    RunStore,
    TaskConfig,
    config_id,
    make_record,
    object_memory_names,
    run_recording,
    total_usage,
    visual_memory_names,
)
from video_agent.task.verdicts import Execution, Order

SOP = Sop(
    activity_id=1,
    recipe_name="Toast",
    steps=((10, "Slice the bread"), (11, "Toast the bread"), (12, "Spread the butter")),
)
GRAPH = TaskGraph(
    activity_id=1,
    prerequisites={10: frozenset(), 11: frozenset({10}), 12: frozenset({11})},
    edges=frozenset({(10, 11), (11, 12)}),
)

ANSWER = {
    "steps": [
        {"step_id": 10, "executed": "yes", "segments": [0, 1], "order": "ok"},
        {"step_id": 11, "executed": "yes", "segments": [2, 3], "order": "ok"},
        {
            "step_id": 12,
            "executed": "yes",
            "segments": [4, 5],
            "order": "violation",
            "violated_prerequisite": {"step_id": 11, "segments": [8, 9]},
        },
    ]
}

FENCED_ANSWER = (
    "Thought: I now know the final answer\n"
    "Final Answer: ```json\n" + json.dumps(ANSWER) + "\n```"
)
TOOL_CALL = (
    "Thought: I should look at the captions\n"
    "Action: caption_retrieval\n"
    'Action Input: (0, 2)'
)


def make_config(**overrides) -> TaskConfig:
    defaults = dict(
        orchestrator_model="fake-chat",
        chat_backend="openrouter",
        thinking=False,
        vqa_backend="none",
        vqa_style="two_questions",
        caption_variant="default",
        use_objects=False,
        seed=None,
        temperature=0.0,
        react=ReActConfig(max_iterations=4),
        tools=ToolConfig(),
        prompt_sha256="a" * 64,
        question_template_sha256="b" * 64,
        code_version="deadbee",
    )
    return TaskConfig(**{**defaults, **overrides})


def make_toolkit(n_segments: int = 10) -> ToolKit:
    return ToolKit(
        video=Video(path=Path("1_7_360p.mp4"), fps=30.0, frame_count=60 * n_segments),
        temporal=make_temporal_memory(n_segments),
        text_embedder=FakeTextEmbedder(),
        aligner=FakeAligner(),
    )


def run(replies: list[str], **overrides) -> dict:
    toolkit = make_toolkit()
    runner = ReActRunner(FakeChat(replies), ReActConfig(max_iterations=4))
    return run_recording(
        recording_id="1_7",
        sop=SOP,
        graph=GRAPH,
        toolkit=toolkit,
        runner=runner,
        prompt_template=load_prompt("task_prompt"),
        question_template=load_prompt("task_question"),
        config=make_config(**overrides),
    )


@pytest.fixture(scope="module")
def record() -> dict:
    return run([TOOL_CALL, FENCED_ANSWER])


def test_a_parsed_answer_is_a_successful_run(record: dict) -> None:
    assert record["status"] == "ok"
    assert record["error"] is None
    assert record["schema_version"] == RUN_SCHEMA_VERSION
    assert record["recording_id"] == "1_7" and record["activity_id"] == 1


def test_the_record_holds_one_verdict_per_sop_step_in_sop_order(record: dict) -> None:
    assert [verdict["step_id"] for verdict in record["verdicts"]] == [10, 11, 12]
    assert record["verdicts"][0]["execution"] == Execution.EXECUTED.value
    assert record["verdicts"][2]["order"] == Order.VIOLATION.value


def test_anchors_are_recorded_in_seconds(record: dict) -> None:
    anchor = record["verdicts"][2]["anchor"]

    assert (anchor["start_s"], anchor["end_s"]) == (8.0, 12.0)
    assert anchor["segments"] == [4, 5]
    assert record["verdicts"][2]["prerequisite_anchor"]["start_s"] == 16.0


def test_the_record_carries_the_question_and_the_raw_answer(record: dict) -> None:
    assert "step 10: Slice the bread" in record["question"]
    assert "Toast" in record["question"]
    assert record["final_answer_text"].startswith("```json")


def test_the_record_counts_the_answers_the_model_wrote(record: dict) -> None:
    assert record["n_answer_candidates"] == 1


def test_a_model_that_answers_twice_is_counted_twice() -> None:
    """The accepted answer alone hides how many there were."""
    answer = json.dumps(ANSWER)
    record = run([f"Thought: done\nFinal Answer: {answer}\n</think>\n\n{answer}"])

    assert record["status"] == "ok"
    assert record["n_answer_candidates"] == 2


def test_an_empty_answer_is_a_parse_failure_that_was_still_written() -> None:
    record = run(['Thought: done\nFinal Answer: {"steps": []}'])

    assert record["status"] == "parse_failed"
    assert record["n_answer_candidates"] == 1
    assert [verdict["execution"] for verdict in record["verdicts"]] == [
        Execution.INDETERMINATE.value
    ] * 3


def test_the_record_describes_the_segment_grid_it_anchored_against(
    record: dict,
) -> None:
    assert record["n_segments"] == 10
    assert record["video"] == {"fps": 30.0, "frame_count": 600, "n_segments": 10}


def test_the_record_carries_the_transcript_and_its_cost(record: dict) -> None:
    transcript = Transcript.from_dict(record["transcript"])

    assert len(transcript.steps) == 2
    assert transcript.steps[0].action.name == "caption_retrieval"
    assert record["usage"]["llm_calls"] == 2
    # No sub-agent ran, so the total equals the parent's own usage.
    assert record["usage_total"]["llm_calls"] >= record["usage"]["llm_calls"]
    assert record["usage_total"]["tokens_in"] >= record["usage"]["tokens_in"]


def test_the_record_is_json_serializable_as_is(record: dict) -> None:
    assert json.loads(json.dumps(record))["status"] == "ok"


def test_the_record_stamps_the_configuration_and_its_id(record: dict) -> None:
    config = make_config()

    assert record["config"]["config_id"] == config_id(config)
    assert record["config"]["orchestrator_model"] == "fake-chat"
    assert record["config"]["react"]["max_iterations"] == 4


def test_the_record_names_the_visual_memory_the_verdicts_came_from(
    record: dict,
) -> None:
    """A recording's directory may hold one embeddings file per aligner.

    Which of them was scored is not something a reader can infer from the
    model id alone, so the record spells the slug and the file name out.
    """
    config = record["config"]

    assert config["tools"]["aligner_model"] == "timm/PE-Core-L-14-336"
    assert config["aligner_slug"] == "pe-core-l"
    assert config["visual_embeddings_file"] == (
        "segment_visual_embeddings.pe-core-l.npy"
    )


def test_the_default_aligner_slug_names_the_unsuffixed_file() -> None:
    # X-CLIP (unsuffixed artifact name) carries no suffix; the record says so.
    config = make_config(
        tools=ToolConfig(aligner_model="microsoft/xclip-base-patch16")
    )

    assert visual_memory_names(config) == {
        "aligner_slug": "xclip-base-patch16",
        "visual_embeddings_file": "segment_visual_embeddings.npy",
    }


def test_the_record_names_the_object_memory_the_sub_agent_read(
    record: dict,
) -> None:
    """A recording's directory may hold one object memory per detector.

    Which of them answered the sub-agent's queries decides which names could
    appear in an observation at all, so the record spells it out next to the
    aligner's.
    """
    config = record["config"]

    assert config["tools"]["object_memory_slug"] is None
    assert config["object_memory_slug"] == "rtdetr-v2-r50"
    assert config["object_memory_file"] == "object_memory.pkl"


def test_another_detectors_memory_names_its_own_file() -> None:
    config = make_config(
        tools=ToolConfig(object_memory_slug="other-detector")
    )

    assert object_memory_names(config) == {
        "object_memory_slug": "other-detector",
        "object_memory_file": "object_memory.other-detector.pkl",
    }
    # The detector decides which names exist to be found, so two runs that
    # differ here are not comparable by config_id either.
    assert config_id(config) != config_id(make_config())


def test_the_two_memories_are_named_by_two_independent_functions() -> None:
    # The aligner's keys follow from ``tools.aligner_model`` and the
    # detector's from ``tools.object_memory_slug``; neither may move when
    # the other changes.
    config = make_config(tools=ToolConfig(object_memory_slug="other-detector"))

    assert visual_memory_names(config) == visual_memory_names(make_config())


def test_a_prose_answer_is_a_parse_failure_that_still_yields_verdicts() -> None:
    record = run(["Final Answer: the cook forgot to slice the bread"])

    assert record["status"] == "parse_failed"
    assert record["n_answer_candidates"] == 0
    assert record["final_answer_text"] == "the cook forgot to slice the bread"
    assert [verdict["execution"] for verdict in record["verdicts"]] == [
        Execution.INDETERMINATE.value
    ] * 3


def test_an_agent_that_never_finishes_keeps_its_status() -> None:
    record = run([TOOL_CALL] * 4)

    assert record["status"] == "max_iterations"
    assert record["final_answer_text"] is None


def test_a_runner_that_raises_is_recorded_as_an_error() -> None:
    class ExplodingRunner:
        def run(self, question, tools, *, prompt_template):
            raise RuntimeError("the GPU melted")

    record = run_recording(
        recording_id="1_7",
        sop=SOP,
        graph=GRAPH,
        toolkit=make_toolkit(),
        runner=ExplodingRunner(),
        prompt_template=load_prompt("task_prompt"),
        question_template=load_prompt("task_question"),
        config=make_config(),
    )

    assert record["status"] == "error"
    assert record["error"] == "RuntimeError: the GPU melted"
    assert record["n_answer_candidates"] == 0
    assert record["transcript"] is None
    assert len(record["verdicts"]) == 3


def test_a_backend_that_dies_keeps_the_episode_it_had_run() -> None:
    """A run that fails is still a record of what it did before it failed.

    Without the partial transcript, a request timeout would leave
    ``transcript`` ``None`` and the usage zeroed, so nothing would say how
    far the agent had got or what it had spent.
    """

    class DyingChat(FakeChat):
        """A backend that answers twice and then times out."""

        def complete(self, messages, *, stop=(), max_tokens=None):
            if not self.replies:
                raise RuntimeError("the server timed out")
            return super().complete(messages, stop=stop, max_tokens=max_tokens)

    toolkit = make_toolkit()
    chat = DyingChat([TOOL_CALL, TOOL_CALL])
    runner = ReActRunner(chat, ReActConfig(max_iterations=4))
    record = run_recording(
        recording_id="1_7",
        sop=SOP,
        graph=GRAPH,
        toolkit=toolkit,
        runner=runner,
        prompt_template=load_prompt("task_prompt"),
        question_template=load_prompt("task_question"),
        config=make_config(),
    )

    assert record["status"] == "error"
    assert record["error"] == "ReActError: RuntimeError: the server timed out"
    assert record["final_answer_text"] is None

    transcript = Transcript.from_dict(record["transcript"])

    assert len(transcript.steps) == 2
    assert transcript.steps[0].action.name == "caption_retrieval"
    assert record["usage"]["llm_calls"] == 2
    assert record["usage_total"]["llm_calls"] == 2
    # The verdicts of a failed run are still one per step, all unclassified.
    assert [verdict["execution"] for verdict in record["verdicts"]] == [
        Execution.INDETERMINATE.value
    ] * 3


def test_an_episode_that_has_not_finished_is_recorded_as_running() -> None:
    """A caller stepping the loop by hand can record an unfinished episode.

    A user stepping the loop by hand has to be able to save what they have
    (the prompt, the steps, the cost) before deciding what to do next.
    Nothing was answered, so every verdict is unclassified, and the status
    says ``running`` rather than borrowing one of the final run outcomes.
    """
    toolkit = make_toolkit()
    question = build_question(
        SOP, GRAPH, toolkit.n_segments, load_prompt("task_question")
    )
    session = ReActRunner(FakeChat([TOOL_CALL]), ReActConfig(max_iterations=4)).session(
        question, toolkit.main_tools(), prompt_template=load_prompt("task_prompt")
    )
    session.step()
    result = session.result()
    assert result.status == "running" and result.answer is None

    record = make_record(
        recording_id="1_7",
        sop=SOP,
        segments=toolkit.temporal.segments,
        video=toolkit.video,
        question=question,
        config=make_config(),
        started_at="2025-01-01T12:00:00+00:00",
        wall_s=1.5,
        result=result,
        error=None,
        partial=None,
    )

    assert record["status"] == "running"
    assert record["error"] is None
    assert record["final_answer_text"] is None
    assert record["n_answer_candidates"] == 0
    assert [verdict["execution"] for verdict in record["verdicts"]] == [
        Execution.INDETERMINATE.value
    ] * 3
    # Everything a finished record carries is there, the episode included.
    assert record["started_at"] == "2025-01-01T12:00:00+00:00"
    assert record["wall_s"] == 1.5
    assert record["n_segments"] == 10
    assert record["config"]["config_id"] == config_id(make_config())
    assert record["usage"]["llm_calls"] == 1
    transcript = Transcript.from_dict(record["transcript"])
    assert len(transcript.steps) == 1
    assert transcript.steps[0].action.name == "caption_retrieval"
    assert json.loads(json.dumps(record))["status"] == "running"


class TestConfigId:
    def test_is_stable_under_dict_key_order(self) -> None:
        assert config_id(make_config()) == config_id(make_config())

    def test_changes_with_the_question_template(self) -> None:
        assert config_id(make_config()) != config_id(
            make_config(question_template_sha256="c" * 64)
        )

    def test_changes_with_the_loop_configuration(self) -> None:
        assert config_id(make_config()) != config_id(
            make_config(react=ReActConfig(max_iterations=5))
        )

    def test_changes_with_the_per_turn_token_cap(self) -> None:
        assert config_id(make_config()) != config_id(
            make_config(react=ReActConfig(max_iterations=4, max_tokens=16384))
        )

    def test_changes_when_the_orchestrator_is_allowed_to_reason(self) -> None:
        assert config_id(make_config()) != config_id(make_config(thinking=True))

    def test_changes_with_the_aligner_of_the_visual_channel(self) -> None:
        # Different tower, different embeddings file, different scores: the
        # two runs must not answer to the same configuration id.
        assert config_id(make_config()) != config_id(
            make_config(tools=ToolConfig(aligner_model="microsoft/xclip-base-patch16"))
        )


class TestShuffledRecord:
    """What a record of the shuffled condition says about its own memory."""

    def permutation(self):
        from video_agent.memory.ablation import segment_permutation

        return segment_permutation(10, ordinal=3)

    def test_the_permutation_is_named_in_the_record_by_digest(self) -> None:
        from video_agent.task.run import permutation_name

        permutation = self.permutation()
        # The same configuration, recorded for another recording of the run —
        # another permutation, or none handed over at all.
        sibling = run([TOOL_CALL, FENCED_ANSWER], memory="shuffled")
        shuffled = run_recording(
            recording_id="1_7",
            sop=SOP,
            graph=GRAPH,
            toolkit=make_toolkit(),
            runner=ReActRunner(
                FakeChat([TOOL_CALL, FENCED_ANSWER]), ReActConfig(max_iterations=4)
            ),
            prompt_template=load_prompt("task_prompt"),
            question_template=load_prompt("task_question"),
            config=make_config(memory="shuffled"),
            permutation=permutation,
        )

        assert shuffled["config"]["memory"] == "shuffled"
        assert shuffled["config"]["memory_permutation_sha256"] == permutation.sha256()
        assert permutation_name(None) == {}
        # One digest per recording, one id for the run: the digest is written
        # into the record and kept out of the hash.
        assert "memory_permutation_sha256" not in sibling["config"]
        assert shuffled["config"]["config_id"] == sibling["config"]["config_id"]

    def test_a_full_record_says_nothing_about_a_memory_it_did_not_shuffle(
        self, record: dict
    ) -> None:
        assert "memory" not in record["config"]
        assert "memory_permutation_sha256" not in record["config"]

    def test_the_condition_of_the_run_is_part_of_its_id(self) -> None:
        ids = {
            config_id(make_config(memory=mode))
            for mode in ("full", "empty", "shuffled")
        }

        assert len(ids) == 3

    def test_the_permutation_is_stored_beside_the_record(
        self, tmp_path: Path
    ) -> None:
        store = RunStore(tmp_path, "agent-shuffled")
        permutation = self.permutation()

        store.save_permutation("1_7", permutation)
        store.save("1_7", {"status": "ok"})

        assert (tmp_path / "agent-shuffled" / "1_7" / "permutation.json").is_file()
        assert store.has_permutation("1_7")
        assert store.load_permutation("1_7") == permutation
        assert store.load_permutation("1_7").sha256() == permutation.sha256()
        # A run with no shuffle leaves no file, and asking creates nothing.
        assert not store.has_permutation("1_1")
        assert not (tmp_path / "agent-shuffled" / "1_1").exists()
        assert store.recording_ids() == ["1_7"]


def test_total_usage_adds_every_nested_sub_agent_episode() -> None:
    from video_agent.agent.transcript import LLMTurn, ReActStep, ToolCall

    def turn(index: int) -> LLMTurn:
        return LLMTurn(
            index=index,
            completion="",
            usage=Usage(llm_calls=1),
            latency_s=0.0,
            model="fake",
            finish_reason=None,
        )

    def transcript(usage: Usage, steps: list[ReActStep]) -> Transcript:
        return Transcript(
            prompt="", steps=steps, final_answer=None, status="final_answer", usage=usage
        )

    grandchild = transcript(Usage(llm_calls=1, tokens_in=5), [])
    child = transcript(
        Usage(llm_calls=2, tokens_in=7),
        [
            ReActStep(
                turn=turn(0),
                action=ToolCall(
                    name="database_querying",
                    raw_input="",
                    args=[],
                    observation="",
                    error=None,
                    latency_s=0.0,
                    sub_transcript=grandchild,
                ),
            )
        ],
    )
    parent = transcript(
        Usage(llm_calls=3, tokens_in=11),
        [
            ReActStep(
                turn=turn(0),
                action=ToolCall(
                    name="object_memory_querying",
                    raw_input="",
                    args=[],
                    observation="",
                    error=None,
                    latency_s=0.0,
                    sub_transcript=child,
                ),
            )
        ],
    )

    assert total_usage(parent) == Usage(llm_calls=6, tokens_in=23)


class TestRunStore:
    def test_knows_which_recordings_it_holds(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path, "agent")

        assert not store.has("1_7")
        assert store.recording_ids() == []

        store.save("1_7", {"status": "ok"})
        store.save("1_1", {"status": "error"})

        assert store.has("1_7")
        assert store.recording_ids() == ["1_1", "1_7"]

    def test_reads_back_what_it_wrote(self, tmp_path: Path, record: dict) -> None:
        store = RunStore(tmp_path, "agent")

        store.save("1_7", record)

        assert store.load("1_7") == json.loads(json.dumps(record))

    def test_asking_about_a_missing_record_creates_nothing(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path, "agent")

        assert not store.has("1_7")
        assert not (tmp_path / "agent" / "1_7").exists()

    def test_writes_under_run_name_and_recording_id(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path, "agent")

        store.save("1_7", {"status": "ok"})

        assert (tmp_path / "agent" / "1_7" / "run.json").is_file()

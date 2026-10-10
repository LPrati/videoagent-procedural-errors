import json
from pathlib import Path

import pytest

from video_agent.agent.config import ReActConfig, ToolConfig
from video_agent.agent.react import ReActRunner
from video_agent.agent.runner import (
    AGENT_STOPPED_MESSAGE,
    ToolResult,
    render_tool_descriptions,
)
from video_agent.agent.tools import (
    MAIN_TOOL_NAMES,
    OPEN_VOCABULARY_UNAVAILABLE,
    ToolKit,
)
from video_agent.media import Video
from video_agent.memory.object_db import text_columns_for

from tests.fakes import (
    FakeAligner,
    FakeChat,
    FakeSiglip,
    FakeTextEmbedder,
    FakeVideoQA,
    make_object_memory,
    make_temporal_memory,
    object_record,
)

PREAMBLE = "There are 3 segments in total, ranging from 0 to 2. "
WIDTHS = (2, 1)  # a text-aligned member and an appearance-only one
TEMPLATE = (
    "{tools}\n\nOne of [{tool_names}].\n\nQuestion: {input}\nThought: {agent_scratchpad}"
)


def objects():
    return make_object_memory(
        object_record(1, "person", (0, 1), [1.0, 0.0, 9.0]),
        object_record(2, "person", (2,), [0.2, 1.0, 9.0]),
        object_record(3, "cup", (0,), [0.5, 0.5, 9.0]),
    )


def make_kit(**overrides) -> ToolKit:
    memory = make_temporal_memory(3)
    memory.captions = ["a", "b", "c"]
    arguments = {
        "video": Video(path=Path("unused.mp4"), fps=30, frame_count=180),
        "temporal": memory,
        "text_embedder": FakeTextEmbedder(),
        "aligner": FakeAligner(),
    }
    arguments.update(overrides)
    return ToolKit(**arguments)


def idle_runner() -> ReActRunner:
    """A runner nothing in the test will call."""
    return ReActRunner(FakeChat([]))


# --- caption retrieval ----------------------------------------------------


def test_caption_retrieval_prefixes_the_captions_with_the_segment_count() -> None:
    kit = make_kit()

    assert kit.caption_retrieval(0, 1) == (
        "There are 3 segments in total, ranging from 0 to 2. {0: 'a', 1: 'b'}"
    )
    assert kit.n_segments == 3


def test_caption_retrieval_clamps_its_window_to_the_grid() -> None:
    kit = make_kit()

    assert kit.caption_retrieval(-5, 99) == PREAMBLE + "{0: 'a', 1: 'b', 2: 'c'}"
    assert kit.caption_retrieval(2, 2) == PREAMBLE + "{2: 'c'}"
    assert kit.caption_retrieval(2, 1) == PREAMBLE + "Invalid start and end segment IDs!"
    assert kit.caption_retrieval(5, 9) == PREAMBLE + "Invalid start and end segment IDs!"


def test_the_advertised_caption_window_is_enforced_only_on_request() -> None:
    # The reference promised 15 captions at most and never enforced it.
    assert make_kit().caption_retrieval(0, 2) == PREAMBLE + "{0: 'a', 1: 'b', 2: 'c'}"

    strict = make_kit(config=ToolConfig(enforce_caption_window=True, caption_window_max=2))
    assert strict.caption_retrieval(0, 2) == PREAMBLE + "{0: 'a', 1: 'b'}"


# --- segment localization -------------------------------------------------


def test_segment_localization_reports_the_ranked_segments_with_their_captions() -> None:
    kit = make_kit()

    assert kit.segment_localization("segment 2") == (
        "There are 3 segments in total, ranging from 0 to 2. "
        "The most relevant segments are: {2: 'c', 1: 'b', 0: 'a'}."
    )
    # Both channels are queried with the description, as in the reference.
    assert kit.text_embedder.texts == ["segment 2"]
    assert kit.aligner.texts == ["segment 2"]


def test_the_localization_depth_comes_from_the_config() -> None:
    kit = make_kit(config=ToolConfig(localization_top_k=1))

    assert kit.segment_localization("segment 1") == (
        PREAMBLE + "The most relevant segments are: {1: 'b'}."
    )


# --- visual question answering -------------------------------------------


def test_visual_question_answering_refuses_a_segment_outside_the_grid() -> None:
    kit = make_kit(vqa=FakeVideoQA())

    assert kit.visual_question_answering("is there a boat?", 3) == (
        "Segment ID 3 not in range 0-2."
    )
    assert kit.visual_question_answering("is there a boat?", -1) == (
        "Segment ID -1 not in range 0-2."
    )


def test_visual_question_answering_decodes_the_window_around_the_segment(
    synthetic_video: Video,
) -> None:
    model = FakeVideoQA()
    kit = make_kit(video=synthetic_video, temporal=make_temporal_memory(2), vqa=model)

    observation = kit.visual_question_answering("is there a boat?", 0)

    assert observation == (
        "Segment description: answer to 'what is the video about?' over 20 frames\n"
        "Answer to the question: answer to 'is there a boat?' over 20 frames"
    )
    assert model.questions == ["what is the video about?", "is there a boat?"]
    assert [f.index for f in model.clips[0].frames] == list(range(0, 120, 6))


# --- the object memory ----------------------------------------------------


def test_database_querying_hands_the_sql_to_the_object_database() -> None:
    with make_kit(objects=objects()) as kit:
        assert kit.database_querying(
            "SELECT COUNT(DISTINCT object_id) FROM Objects WHERE category = 'person'"
        ) == "[(2,)]"
        assert "syntax error" in kit.database_querying("SELEC 1")


def test_open_vocabulary_retrieval_returns_the_candidate_object_ids() -> None:
    embedder = FakeSiglip()
    # The fake similarities were laid out against the reference threshold;
    # pin it so the test reads the filter, not the calibrated default.
    with make_kit(
        objects=objects(),
        object_text_embedder=embedder,
        object_text_columns=text_columns_for(WIDTHS),
        config=ToolConfig(object_similarity_threshold=0.26),
    ) as kit:
        assert kit.open_vocabulary_object_retrieval("cup") == "[1, 3]"
        assert embedder.texts == ["a photo of a cup."]


def test_open_vocabulary_retrieval_stays_listed_when_it_cannot_run() -> None:
    with make_kit(objects=objects()) as kit:
        assert kit.open_vocabulary_object_retrieval("cup") == OPEN_VOCABULARY_UNAVAILABLE
        assert kit.open_vocabulary_object_retrieval("cup") == (
            "Open-vocabulary retrieval is not available."
        )
        # The sub-agent's prompt names the tool: hiding it would change it.
        assert [tool.name for tool in kit.sub_tools()] == [
            "database_querying",
            "open_vocabulary_object_retrieval",
        ]


def test_the_object_memory_tool_runs_a_nested_episode() -> None:
    chat = FakeChat(
        [
            "Thought: the object memory knows.\nAction: object_memory_querying\n"
            "Action Input: how many people are there in the video?",
            "Thought: count the people.\nAction: database_querying\nAction Input: "
            "SELECT COUNT(DISTINCT object_id) FROM Objects WHERE category = 'person'",
            "Thought: I now know the final answer\nFinal Answer: 2",
            "Thought: I now know the final answer\nFinal Answer: there are 2 people",
        ]
    )
    runner = ReActRunner(chat)

    with make_kit(
        objects=objects(), sub_runner=runner, sub_prompt=TEMPLATE
    ) as kit:
        result = runner.run(
            "how many people are there in the video?",
            kit.main_tools(),
            prompt_template=TEMPLATE,
        )

    assert result.answer == "there are 2 people"
    assert chat.replies == []  # both episodes consumed the same script, in order

    call = result.transcript.steps[0].action
    assert call.name == "object_memory_querying"
    assert call.observation == "2"

    sub = call.sub_transcript
    assert sub is not None
    assert sub.status == "final_answer"
    assert sub.final_answer == "2"
    assert sub.steps[0].action.name == "database_querying"
    assert sub.steps[0].action.observation == "[(2,)]"
    # The nested prompt lists the sub-agent's tools, not the orchestrator's.
    assert chat.prompts[1].startswith(
        "database_querying: database_querying(program) - given a MySQL program"
    )
    assert "[database_querying, open_vocabulary_object_retrieval]" in chat.prompts[1]
    assert "    Observation: [(2,)]" in result.transcript.render()


def test_a_sub_agent_that_runs_out_of_steps_says_so_to_its_caller() -> None:
    chat = FakeChat(["Action: database_querying\nAction Input: SELECT 1"] * 2)
    runner = ReActRunner(chat, ReActConfig(max_iterations=2))

    with make_kit(objects=objects(), sub_runner=runner, sub_prompt=TEMPLATE) as kit:
        result = kit.object_memory_querying("how many people are there?")

    assert isinstance(result, ToolResult)
    assert result.observation == "Agent stopped due to iteration limit or time limit."
    assert result.observation == AGENT_STOPPED_MESSAGE
    assert result.sub_transcript.status == "max_iterations"


# --- which tools exist ----------------------------------------------------


def test_the_tool_list_reproduces_the_references_order() -> None:
    with make_kit(
        objects=objects(),
        vqa=FakeVideoQA(),
        sub_runner=idle_runner(),
        sub_prompt=TEMPLATE,
    ) as kit:
        assert [tool.name for tool in kit.main_tools()] == [
            "caption_retrieval",
            "segment_localization",
            "visual_question_answering",
            "object_memory_querying",
        ]


def test_the_constant_names_the_tools_the_full_toolkit_builds() -> None:
    """``MAIN_TOOL_NAMES`` is read by reports that never build a toolkit."""
    with make_kit(
        objects=objects(),
        vqa=FakeVideoQA(),
        sub_runner=idle_runner(),
        sub_prompt=TEMPLATE,
    ) as kit:
        assert tuple(tool.name for tool in kit.main_tools()) == MAIN_TOOL_NAMES


def test_each_ablation_removes_the_tool_it_disables() -> None:
    assert [tool.name for tool in make_kit().main_tools()] == [
        "caption_retrieval",
        "segment_localization",
    ]

    with make_kit(objects=objects(), vqa=FakeVideoQA()) as no_sub_agent:
        # An object memory with nobody to query it is not a tool.
        assert [tool.name for tool in no_sub_agent.main_tools()] == [
            "caption_retrieval",
            "segment_localization",
            "visual_question_answering",
        ]

    no_objects = make_kit(sub_runner=idle_runner(), sub_prompt=TEMPLATE)
    assert "object_memory_querying" not in [t.name for t in no_objects.main_tools()]


def test_the_tool_descriptions_are_the_reference_ones() -> None:
    # The descriptions are prompt text: the model reads them before every step.
    with make_kit(
        objects=objects(),
        vqa=FakeVideoQA(),
        sub_runner=idle_runner(),
        sub_prompt=TEMPLATE,
    ) as kit:
        described = {
            tool.name: tool.description for tool in kit.main_tools() + kit.sub_tools()
        }

    assert described["caption_retrieval"] == (
        "given an input tuple (start_segment_ID, end_segment_ID), this tool will "
        "retrieve all the captions between the two segments, 15 captions at most. "
        "end_segment_ID < start_segment_ID + 15."
    )
    assert described["segment_localization"] == (
        "Given a textual description, this tool will return the top-5 candidate "
        "segments that are most relevant to the description."
    )
    assert described["visual_question_answering"] == (
        "Given an input tuple (question, segment_ID), this tool will focus on the "
        "video segments starting from segment_ID-1 to segment_ID+1. It will return "
        "the description of the video segment and the answer to the question based "
        "on the segment."
    )
    assert described["object_memory_querying"] == (
        "Given a question about open-vocabulary objects such as 'how many people "
        "are there in the video?' or 'In which segments did the brown dog appear?', "
        "this tool will give the answer based on the object memory."
    )
    assert described["database_querying"] == (
        "given a MySQL program, this tool will query the database and return the "
        "results."
    )
    assert described["open_vocabulary_object_retrieval"] == (
        "given an open-vocabulary description of an object or a person (frying pan, "
        "person in red clothes e.g.), this tool will return the possible candidate "
        "object IDs that satisfy the description."
    )


def test_the_tool_descriptions_state_the_configured_limits() -> None:
    tuned = make_kit(
        config=ToolConfig(
            caption_window_max=8, localization_top_k=3, vqa_context_segments=2
        ),
        vqa=FakeVideoQA(),
    )
    described = {tool.name: tool.description for tool in tuned.main_tools()}

    assert "8 captions at most" in described["caption_retrieval"]
    assert "start_segment_ID + 8." in described["caption_retrieval"]
    assert "top-3 candidate" in described["segment_localization"]
    assert (
        "segment_ID-2 to segment_ID+2" in described["visual_question_answering"]
    )


def test_the_tools_of_a_missing_dependency_refuse_to_run() -> None:
    kit = make_kit()

    with pytest.raises(ValueError):
        kit.visual_question_answering("is there a boat?", 0)
    with pytest.raises(ValueError):
        kit.database_querying("SELECT 1")
    with pytest.raises(ValueError):
        kit.open_vocabulary_object_retrieval("cup")
    with pytest.raises(ValueError):
        kit.object_memory_querying("how many people are there?")
    kit.close()  # nothing to release without an object memory


def test_the_tool_config_records_the_reference_values() -> None:
    data = ToolConfig().to_dict()

    assert json.loads(json.dumps(data)) == data
    assert (data["visual_weight"], data["textual_weight"]) == (18.0, 11.0)
    assert data["localization_top_k"] == 5
    assert data["object_similarity_threshold"] == 0.10
    assert data["object_query_template"] == "a photo of a {}."


def test_the_tool_config_names_the_aligner_of_the_visual_channel() -> None:
    """The aligner, the one reference value replaced, recorded like every other knob.

    ``segment_localization`` reads this model's embeddings and asks this
    model's text tower, so a run that changed it is a different run.
    """
    data = ToolConfig().to_dict()

    assert data["aligner_model"] == "timm/PE-Core-L-14-336"
    assert json.loads(json.dumps(data)) == data


def test_the_rendered_tool_block_names_each_tool_twice() -> None:
    # What the model actually read: LangChain rebuilt every description from
    # the decorated function's signature before the renderer prefixed the name.
    with make_kit(
        objects=objects(),
        vqa=FakeVideoQA(),
        sub_runner=idle_runner(),
        sub_prompt=TEMPLATE,
    ) as kit:
        lines = render_tool_descriptions(kit.main_tools()).split("\n")
        sub_lines = render_tool_descriptions(kit.sub_tools()).split("\n")

    assert lines[0] == (
        "caption_retrieval: caption_retrieval(input_tuple) - given an input tuple "
        "(start_segment_ID, end_segment_ID), this tool will retrieve all the "
        "captions between the two segments, 15 captions at most. end_segment_ID "
        "< start_segment_ID + 15."
    )
    assert lines[1] == (
        "segment_localization: segment_localization(description) - Given a textual "
        "description, this tool will return the top-5 candidate segments that are "
        "most relevant to the description."
    )
    assert lines[2].startswith(
        "visual_question_answering: visual_question_answering(input_tuple) - Given "
        "an input tuple (question, segment_ID),"
    )
    assert lines[3].startswith(
        "object_memory_querying: object_memory_querying(question) - Given a question"
    )
    assert sub_lines[0].startswith(
        "database_querying: database_querying(program) - given a MySQL program"
    )
    assert sub_lines[1].startswith(
        "open_vocabulary_object_retrieval: open_vocabulary_object_retrieval"
        "(description) - given an open-vocabulary description"
    )


def test_the_single_reply_vqa_style_is_passed_on_verbatim(
    synthetic_video: Video,
) -> None:
    model = FakeVideoQA()
    kit = make_kit(
        video=synthetic_video,
        temporal=make_temporal_memory(2),
        vqa=model,
        config=ToolConfig(vqa_style="single"),
    )

    observation = kit.visual_question_answering("is there a boat?", 0)

    # The gpt-4v path answered both halves in one reply, which was the whole
    # observation: no "Segment description:" framing around it.
    assert observation == "answer to 'is there a boat?' over 20 frames"
    assert model.questions == ["is there a boat?"]

import json

import pytest

from video_agent.agent.config import ReActConfig
from video_agent.agent.react import (
    EDITED_FINISH_REASON,
    EDITED_TURN_MODEL,
    FINAL_ANSWER_AND_ACTION_MESSAGE,
    INVALID_RESPONSE_OBSERVATION,
    INVALID_TUPLE_OBSERVATION,
    MISSING_ACTION_INPUT_MESSAGE,
    MISSING_ACTION_MESSAGE,
    SESSION_RUNNING,
    ReActError,
    ReActParseError,
    ReActRunner,
    parse_react_output,
    parse_tool_arguments,
    render_prompt,
)
from video_agent.agent.runner import ParamSpec, ToolResult, ToolSpec
from video_agent.agent.transcript import Transcript
from video_agent.models.types import Usage

from tests.fakes import FakeChat, ScriptedReply

TEMPLATE = (
    "Answer the question.\n\n{tools}\n\n"
    "Action should be one of [{tool_names}].\n\n"
    "Question: {input}\nThought: {agent_scratchpad}"
)
QUESTION = "how many boats are there in the video?"


def make_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="caption_retrieval",
            description="retrieves captions",
            params=(ParamSpec("start", "int"), ParamSpec("end", "int")),
            input_name="input_tuple",
            fn=lambda start, end: f"captions {start}-{end}",
        ),
        ToolSpec(
            name="segment_localization",
            description="ranks segments",
            params=(ParamSpec("description", "str"),),
            input_name="description",
            fn=lambda description: f"localized {description}",
        ),
    ]


def run(replies: list[str], *, tools=None, config=ReActConfig()):
    chat = FakeChat(replies)
    tools = make_tools() if tools is None else tools
    result = ReActRunner(chat, config).run(QUESTION, tools, prompt_template=TEMPLATE)
    return chat, result


def scratchpad_of(chat: FakeChat, index: int) -> str:
    """The scratchpad the runner had built by prompt ``index``."""
    prefix = render_prompt(TEMPLATE, tools=make_tools(), question=QUESTION, scratchpad="")
    assert chat.prompts[index].startswith(prefix)
    return chat.prompts[index][len(prefix) :]


# --- the output parser, branch by branch ---------------------------------


def test_an_action_and_its_input_are_stripped_of_space_and_quotes() -> None:
    parsed = parse_react_output(
        'Thought: I should look.\nAction: segment_localization\nAction Input:  "a boat"  '
    )

    assert parsed.kind == "action"
    assert parsed.tool == "segment_localization"
    assert parsed.tool_input == "a boat"


def test_numbered_actions_and_multiline_inputs_are_accepted() -> None:
    parsed = parse_react_output(
        'Action 1: segment_localization\nAction Input 1: "a man\nopening a jar"'
    )

    assert parsed.tool == "segment_localization"
    assert parsed.tool_input == "a man\nopening a jar"


def test_a_final_answer_is_everything_after_the_marker() -> None:
    parsed = parse_react_output("Thought: I now know.\nFinal Answer:  two boats  ")

    assert parsed.kind == "finish"
    assert parsed.answer == "two boats"


def test_an_answer_next_to_a_parseable_action_is_rejected() -> None:
    text = "Action: caption_retrieval\nAction Input: (0, 1)\nFinal Answer: 42"

    parsed = parse_react_output(text)

    assert parsed.kind == "error"
    assert parsed.error == f"{FINAL_ANSWER_AND_ACTION_MESSAGE}: {text}"
    # The upstream constant already ends with a colon and is interpolated
    # after another one, so the doubled colon is part of the message.
    assert parsed.error == (
        "Parsing LLM output produced both a final answer and a parse-able action:: "
        + text
    )
    assert parsed.send_to_llm is False


def test_a_thought_without_an_action_says_which_line_is_missing() -> None:
    text = "Thought: I am not sure what to do."

    parsed = parse_react_output(text)

    assert parsed.kind == "error"
    assert parsed.error == f"Could not parse LLM output: `{text}`"
    # The unbalanced quote is the reference's typo, reproduced verbatim.
    assert parsed.observation == "Invalid Format: Missing 'Action:' after 'Thought:"
    assert parsed.observation == MISSING_ACTION_MESSAGE
    assert parsed.send_to_llm is True


def test_an_action_without_an_input_says_which_line_is_missing() -> None:
    text = "Thought: look it up.\nAction: caption_retrieval"

    parsed = parse_react_output(text)

    assert parsed.observation == "Invalid Format: Missing 'Action Input:' after 'Action:'"
    assert parsed.observation == MISSING_ACTION_INPUT_MESSAGE
    assert parsed.send_to_llm is True


def test_the_two_halves_in_the_wrong_order_are_not_explained_to_the_model() -> None:
    text = "Action Input: (0, 1)\nAction: caption_retrieval"

    parsed = parse_react_output(text)

    assert parsed.kind == "error"
    assert parsed.error == f"Could not parse LLM output: `{text}`"
    assert parsed.observation is None
    assert parsed.send_to_llm is False


def test_a_parse_error_can_be_raised_by_callers_that_want_it_fatal() -> None:
    error = ReActParseError.from_parsed(parse_react_output("Thought: hmm"))

    assert error.observation == MISSING_ACTION_MESSAGE
    assert error.send_to_llm is True
    assert str(error) == "Could not parse LLM output: `Thought: hmm`"


# --- prompt rendering -----------------------------------------------------


def test_render_prompt_fills_the_four_placeholders() -> None:
    rendered = render_prompt(
        TEMPLATE, tools=make_tools(), question=QUESTION, scratchpad="Thought: "
    )

    assert (
        "caption_retrieval: caption_retrieval(input_tuple) - retrieves captions\n"
        "segment_localization: segment_localization(description) - ranks segments"
    ) in rendered
    assert "[caption_retrieval, segment_localization]" in rendered
    assert rendered.endswith(f"Question: {QUESTION}\nThought: Thought: ")


def test_render_prompt_leaves_braces_in_the_template_alone() -> None:
    template = 'Use single quotes, as in {"answer": 1}. Question: {input}'

    assert (
        render_prompt(template, tools=[], question="q", scratchpad="")
        == 'Use single quotes, as in {"answer": 1}. Question: q'
    )


# --- argument parsing -----------------------------------------------------


def test_single_string_tools_take_the_action_input_verbatim() -> None:
    spec = make_tools()[1]

    assert parse_tool_arguments(spec, "a boat, at sea") == ["a boat, at sea"]


def test_two_argument_tools_read_a_literal_tuple() -> None:
    spec = make_tools()[0]

    assert parse_tool_arguments(spec, "(0, 3)") == [0, 3]
    assert parse_tool_arguments(spec, "(0, '3')") == [0, 3]  # int() cast, as in the reference
    assert parse_tool_arguments(spec, "(0, 1, 2)") is None  # wrong arity
    assert parse_tool_arguments(spec, "from 0 to 3") is None
    assert parse_tool_arguments(spec, "(0, 'three')") is None


def test_lenient_parsing_recovers_a_badly_quoted_tuple() -> None:
    spec = ToolSpec(
        name="visual_question_answering",
        description="asks",
        params=(ParamSpec("question", "str"), ParamSpec("segment_ID", "int")),
        input_name="input_tuple",
        fn=lambda question, segment_id: f"{question}|{segment_id}",
    )
    raw = '("what is the man doing, 2)'

    assert parse_tool_arguments(spec, raw) is None
    assert parse_tool_arguments(spec, raw, lenient=True) == ["what is the man doing", 2]


# --- the loop -------------------------------------------------------------


def test_a_final_answer_ends_the_episode() -> None:
    chat, result = run(["Thought: I know.\nFinal Answer: two boats"])

    assert result.answer == "two boats"
    assert result.status == "final_answer"
    assert result.transcript.status == "final_answer"
    assert result.transcript.final_answer == "two boats"
    assert result.transcript.steps[0].final_answer == "two boats"
    assert result.transcript.steps[0].action is None
    assert result.usage.llm_calls == 1
    assert result.usage.tool_calls == 0
    assert result.usage.wall_s >= 0.0
    assert chat.stops == [("\nObservation",)]
    assert chat.max_tokens == [None]
    assert chat.prompts[0] == result.transcript.prompt


def test_the_scratchpad_grows_by_completion_observation_thought() -> None:
    completion = "Thought: look.\nAction: caption_retrieval\nAction Input: (0, 1)"
    chat, result = run([completion, "Final Answer: two boats"])

    # The reference's tools returned '\n' + text + '\n', so the observation
    # reaches the model padded even though the record keeps it clean.
    assert scratchpad_of(chat, 1) == f"{completion}\nObservation: \ncaptions 0-1\n\nThought: "

    call = result.transcript.steps[0].action
    assert (call.name, call.raw_input, call.args) == ("caption_retrieval", "(0, 1)", [0, 1])
    assert call.observation == "captions 0-1"
    assert call.error is None
    assert call.sub_transcript is None
    assert result.usage.tool_calls == 1
    assert result.usage.tool_errors == 0
    assert result.usage.llm_calls == 2


def test_observation_padding_can_be_turned_off() -> None:
    completion = "Action: caption_retrieval\nAction Input: (0, 1)"
    chat, _ = run(
        [completion, "Final Answer: done"],
        config=ReActConfig(observation_padding=False),
    )

    assert scratchpad_of(chat, 1) == f"{completion}\nObservation: captions 0-1\nThought: "


def test_an_unknown_tool_is_answered_with_the_valid_names() -> None:
    completion = "Action: caption_retreival\nAction Input: (0, 1)"
    chat, result = run([completion, "Final Answer: done"])

    call = result.transcript.steps[0].action
    assert call.observation == (
        "caption_retreival is not a valid tool, "
        "try one of [caption_retrieval, segment_localization]."
    )
    assert call.args is None
    assert call.error == "unknown tool"
    assert result.usage.tool_calls == 1
    assert result.usage.tool_errors == 1
    # The executor's own message never went through a tool wrapper: no padding.
    assert scratchpad_of(chat, 1) == (
        f"{completion}\nObservation: {call.observation}\nThought: "
    )


def test_an_unusable_argument_tuple_is_answered_like_the_reference() -> None:
    completion = "Action: caption_retrieval\nAction Input: from 0 to 1"
    chat, result = run([completion, "Final Answer: done"])

    call = result.transcript.steps[0].action
    assert call.observation == "Invalid input tuple!"
    assert call.observation == INVALID_TUPLE_OBSERVATION
    assert call.args is None
    assert result.usage.tool_errors == 1
    assert scratchpad_of(chat, 1) == (
        f"{completion}\nObservation: \nInvalid input tuple!\n\nThought: "
    )


def test_an_explainable_parse_error_shows_the_model_its_own_text() -> None:
    completion = "Thought: I am not sure what to do."
    chat, result = run([completion, "Final Answer: done"])

    step = result.transcript.steps[0]
    assert step.action is None
    assert step.parse_error == f"Could not parse LLM output: `{completion}`"
    assert scratchpad_of(chat, 1) == (
        f"{completion}\nObservation: Invalid Format: Missing 'Action:' after "
        "'Thought:\nThought: "
    )
    assert result.usage.parse_errors == 1
    assert result.usage.tool_calls == 0


def test_an_unexplainable_parse_error_hides_the_completion() -> None:
    completion = "Action: caption_retrieval\nAction Input: (0, 1)\nFinal Answer: 42"
    chat, result = run([completion, "Final Answer: done"])

    error = f"{FINAL_ANSWER_AND_ACTION_MESSAGE}: {completion}"
    assert result.transcript.steps[0].parse_error == error
    assert scratchpad_of(chat, 1) == (
        f"{error}\nObservation: Invalid or incomplete response\nThought: "
    )
    assert INVALID_RESPONSE_OBSERVATION == "Invalid or incomplete response"


def test_parse_errors_can_stop_the_episode_as_the_reference_did() -> None:
    chat, result = run(
        ["Thought: I am not sure what to do."],
        config=ReActConfig(parse_errors_as_observations=False),
    )

    assert result.status == "parse_failed"
    assert result.answer is None
    assert len(result.transcript.steps) == 1
    assert chat.replies == []  # the loop stopped instead of asking again


def test_the_loop_stops_at_the_iteration_cap() -> None:
    completion = "Action: segment_localization\nAction Input: boats"
    chat, result = run([completion] * 15)

    assert result.status == "max_iterations"
    assert result.answer is None
    assert len(result.transcript.steps) == 15
    assert result.usage.llm_calls == 15
    assert result.usage.tool_calls == 15
    assert chat.replies == []


def test_the_cap_is_configurable() -> None:
    completion = "Action: segment_localization\nAction Input: boats"
    _, result = run([completion] * 2, config=ReActConfig(max_iterations=2))

    assert result.status == "max_iterations"
    assert len(result.transcript.steps) == 2


def test_a_failing_tool_becomes_an_observation_or_is_raised() -> None:
    def boom(description: str) -> str:
        raise RuntimeError("no index for this video")

    tools = [
        ToolSpec(
            name="segment_localization",
            description="ranks segments",
            params=(ParamSpec("description", "str"),),
            input_name="description",
            fn=boom,
        )
    ]
    completion = "Action: segment_localization\nAction Input: boats"

    _, result = run([completion, "Final Answer: done"], tools=tools)
    call = result.transcript.steps[0].action
    assert call.observation == "Tool error: no index for this video"
    assert call.error == "RuntimeError: no index for this video"
    assert call.args == ["boats"]
    assert result.usage.tool_errors == 1

    with pytest.raises(RuntimeError):
        run(
            [completion],
            tools=tools,
            config=ReActConfig(tool_errors_as_observations=False),
        )


def test_a_tool_result_carries_its_sub_transcript_into_the_call() -> None:
    sub = Transcript(
        prompt="sub prompt",
        steps=[],
        final_answer="one dog",
        status="final_answer",
        usage=Usage(llm_calls=2),
    )
    tools = [
        ToolSpec(
            name="segment_localization",
            description="ranks segments",
            params=(ParamSpec("description", "str"),),
            input_name="description",
            fn=lambda description: ToolResult(observation="one dog", sub_transcript=sub),
        )
    ]

    _, result = run(
        ["Action: segment_localization\nAction Input: dog", "Final Answer: one"],
        tools=tools,
    )

    call = result.transcript.steps[0].action
    assert call.observation == "one dog"
    assert call.sub_transcript is sub
    # The sub-agent's own calls stay in its transcript, not in the parent's.
    assert result.usage.llm_calls == 2


def test_max_tokens_reaches_the_model() -> None:
    chat, _ = run(["Final Answer: done"], config=ReActConfig(max_tokens=256))

    assert chat.max_tokens == [256]


def test_a_turn_cut_off_by_the_token_cap_is_observed_like_a_parse_error() -> None:
    """The per-turn token cap turns a runaway turn into a half-written completion.

    Nothing about it is special to the loop: it holds no ``Action:``, so it
    earns the reference's missing-action observation and the episode goes on.
    """
    truncated = ScriptedReply(
        "Thought: let me weigh every step before I decide, starting with",
        finish_reason="length",
    )

    chat, result = run([truncated, "Final Answer: done"])

    assert result.status == "final_answer" and result.answer == "done"
    assert result.transcript.steps[0].turn.finish_reason == "length"
    assert result.transcript.steps[0].parse_error is not None
    assert result.usage.parse_errors == 1
    assert MISSING_ACTION_MESSAGE in scratchpad_of(chat, 1)


def test_a_reasoning_block_is_recorded_beside_its_completion() -> None:
    """With ``--thinking`` the server returns ``reasoning_content`` apart.

    It must not reach the scratchpad — the loop replays completions, and the
    reasoning is not one — but it is the bulk of what the turn cost, so the
    transcript keeps it and a JSON round trip must not drop it.
    """
    thinking = ScriptedReply(
        "Thought: I now know the final answer\nFinal Answer: done",
        reasoning="The user asks for steps. Step 3 is the ramekin one.",
    )

    chat, result = run([thinking])
    turn = result.transcript.steps[0].turn

    assert turn.reasoning == "The user asks for steps. Step 3 is the ramekin one."
    assert turn.reasoning not in result.transcript.render()
    assert turn.reasoning not in chat.prompts[0]

    round_tripped = Transcript.from_dict(
        json.loads(json.dumps(result.transcript.to_dict()))
    )

    assert round_tripped.steps[0].turn.reasoning == turn.reasoning


def test_a_backend_that_dies_hands_over_the_episode_it_had_built() -> None:
    """A timeout on turn three must not erase turns one and two."""

    class DyingChat(FakeChat):
        def complete(self, messages, *, stop=(), max_tokens=None):
            if not self.replies:
                raise RuntimeError("the server timed out")
            return super().complete(messages, stop=stop, max_tokens=max_tokens)

    action = "Action: segment_localization\nAction Input: boats"
    runner = ReActRunner(DyingChat([action, action]), ReActConfig())

    with pytest.raises(ReActError) as failure:
        runner.run(QUESTION, make_tools(), prompt_template=TEMPLATE)

    assert str(failure.value) == "RuntimeError: the server timed out"
    assert isinstance(failure.value.__cause__, RuntimeError)
    transcript = failure.value.transcript
    assert transcript.status == "error" and transcript.final_answer is None
    assert len(transcript.steps) == 2
    assert transcript.usage.llm_calls == 2


def test_the_loop_config_reaches_the_run_record_as_json() -> None:
    data = ReActConfig().to_dict()

    assert json.loads(json.dumps(data)) == data
    assert data["max_iterations"] == 15  # LangChain's default, as the reference ran
    assert data["stop"] == ["\nObservation"]


# --- the loop, one turn at a time ----------------------------------------

# Two actions and an answer: enough for the scratchpad to grow twice.
SCRIPT = [
    "Thought: look.\nAction: caption_retrieval\nAction Input: (0, 1)",
    "Thought: where?\nAction: segment_localization\nAction Input: a boat",
    "Thought: I know.\nFinal Answer: two boats",
]

# One turn of every kind the loop distinguishes, in the order it meets them:
# an action, an unknown tool, an unusable tuple, a parse error the reference
# explains to the model, one it hides, and the answer.
MIXED_SCRIPT = [
    "Thought: look.\nAction: caption_retrieval\nAction Input: (0, 1)",
    "Action: caption_retreival\nAction Input: (0, 1)",
    "Action: caption_retrieval\nAction Input: from 0 to 1",
    "Thought: I am not sure what to do.",
    "Action: caption_retrieval\nAction Input: (0, 1)\nFinal Answer: 42",
    "Thought: I know.\nFinal Answer: two boats",
]


def zeroed(transcript: Transcript) -> dict:
    """The transcript as a dict with every clock reading set to zero.

    Two runs of the same script differ only in how long they took, and that
    is the one thing a comparison of them must ignore.
    """
    data = transcript.to_dict()
    data["usage"]["wall_s"] = 0.0
    for step in data["steps"]:
        step["turn"]["latency_s"] = 0.0
        if step["action"] is not None:
            step["action"]["latency_s"] = 0.0
    return data


def session_of(chat: FakeChat, config: ReActConfig = ReActConfig(), steps=()):
    """A session over the usual tools, resuming ``steps``."""
    return ReActRunner(chat, config).session(
        QUESTION, make_tools(), prompt_template=TEMPLATE, steps=steps
    )


def test_a_session_stepped_to_the_end_matches_run_turn_for_turn() -> None:
    """``run`` is this loop driven to exhaustion, and must stay that."""
    looped_chat, looped = run(list(SCRIPT))

    stepped_chat = FakeChat(list(SCRIPT))
    session = session_of(stepped_chat)
    assert session.status == SESSION_RUNNING
    while session.active:
        session.step()
    stepped = session.result()

    assert stepped_chat.prompts == looped_chat.prompts
    assert zeroed(stepped.transcript) == zeroed(looped.transcript)
    assert (stepped.answer, stepped.status) == (looped.answer, looped.status)
    assert session.next_index == len(SCRIPT)


@pytest.mark.parametrize(
    "config", [ReActConfig(), ReActConfig(observation_padding=False)]
)
def test_a_session_rebuilt_from_a_saved_prefix_asks_what_the_run_asked(
    config: ReActConfig,
) -> None:
    """The prompt is derived from the steps, so a record can be resumed.

    The steps go through JSON on the way, because that is how a reader of a run gets
    them: out of a saved ``run.json`` and not out of the process that
    produced them.
    """
    chat, result = run(list(MIXED_SCRIPT), config=config)
    steps = Transcript.from_dict(
        json.loads(json.dumps(result.transcript.to_dict()))
    ).steps
    runner = ReActRunner(FakeChat([]), config)

    for k in range(len(steps)):
        # No prefix holds the answer: it is the last step of the script.
        assert all(step.final_answer is None for step in steps[:k])
        session = runner.session(
            QUESTION, make_tools(), prompt_template=TEMPLATE, steps=steps[:k]
        )

        assert session.prompt() == chat.prompts[k]


def test_the_cost_of_a_resumed_episode_is_recounted_from_its_steps() -> None:
    """Otherwise a resumed session would report the tokens twice."""
    _, result = run(list(MIXED_SCRIPT))

    session = session_of(FakeChat([]), steps=result.transcript.steps[:5])

    assert session.usage.llm_calls == 5
    assert session.usage.tokens_in == 5 and session.usage.tokens_out == 5
    assert session.usage.tool_calls == 3  # one good call and two bad ones
    assert session.usage.tool_errors == 2
    assert session.usage.parse_errors == 2
    assert session.usage.wall_s == 0.0  # time this process did not spend


def test_editing_the_last_observation_changes_the_next_prompt() -> None:
    """The reason the loop keeps steps instead of a scratchpad string."""
    session = session_of(FakeChat([SCRIPT[0]]))
    session.step()
    assert "captions 0-1" in session.prompt()

    call = session.edit_observation("the cook slices the bread")

    assert call.observation == "the cook slices the bread"
    assert session.prompt().endswith(
        "Observation: \nthe cook slices the bread\n\nThought: "
    )
    assert "captions 0-1" not in session.prompt()


def test_an_injected_completion_skips_the_model_and_says_where_it_came_from() -> None:
    """``FakeChat([])`` raises if asked, so the tool ran on the typed action."""
    session = session_of(FakeChat([]))

    step = session.step(
        "Thought: mine.\nAction: caption_retrieval\nAction Input: (0, 1)"
    )

    assert step.turn.model == EDITED_TURN_MODEL == "human-edit"
    assert step.turn.finish_reason == EDITED_FINISH_REASON == "edited"
    assert step.turn.usage == Usage()  # a turn nobody was billed for
    assert step.action.observation == "captions 0-1"
    assert session.usage.llm_calls == 0 and session.usage.tool_calls == 1
    assert session.active


class TestASessionRefusesToFalsifyTheRecord:
    def test_by_resuming_an_episode_that_already_answered(self) -> None:
        _, result = run(["Thought: I know.\nFinal Answer: two boats"])

        with pytest.raises(ValueError):
            session_of(FakeChat([]), steps=result.transcript.steps)

    def test_by_stepping_one_that_has_finished(self) -> None:
        session = session_of(FakeChat(["Thought: I know.\nFinal Answer: two boats"]))
        session.step()

        assert not session.active and session.status == "final_answer"
        with pytest.raises(RuntimeError):
            session.step()

    def test_by_stepping_a_prefix_that_is_already_at_the_cap(self) -> None:
        completion = "Action: segment_localization\nAction Input: boats"
        config = ReActConfig(max_iterations=2)
        _, result = run([completion] * 2, config=config)

        session = session_of(FakeChat([]), config=config, steps=result.transcript.steps)

        assert not session.active and session.status == "max_iterations"
        assert session.result().status == "max_iterations"

    def test_by_editing_an_observation_no_tool_produced(self) -> None:
        session = session_of(FakeChat([]))
        session.step("Thought: I am not sure what to do.")

        with pytest.raises(RuntimeError):
            session.edit_observation("captions 0-1")


def test_a_dead_backend_ends_the_session_and_hands_over_its_steps() -> None:
    """The failure of ``run``, seen a turn at a time: the episode survives."""

    class DyingChat(FakeChat):
        def complete(self, messages, *, stop=(), max_tokens=None):
            if not self.replies:
                raise RuntimeError("the server timed out")
            return super().complete(messages, stop=stop, max_tokens=max_tokens)

    session = session_of(DyingChat([SCRIPT[0]]))
    session.step()

    with pytest.raises(ReActError) as failure:
        session.step()

    assert str(failure.value) == "RuntimeError: the server timed out"
    transcript = failure.value.transcript
    assert transcript.status == "error" and len(transcript.steps) == 1
    assert transcript.usage.llm_calls == 1
    assert session.status == "error" and not session.active
    assert session.result().status == "error"

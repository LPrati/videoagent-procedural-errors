import json

from video_agent.agent.transcript import LLMTurn, ReActStep, ToolCall, Transcript
from video_agent.models.types import Usage


def turn(index: int, completion: str) -> LLMTurn:
    return LLMTurn(
        index=index,
        completion=completion,
        usage=Usage(llm_calls=1, tokens_in=10, tokens_out=5),
        latency_s=0.5,
        model="fake-chat",
        finish_reason="stop",
    )


def sub_transcript() -> Transcript:
    return Transcript(
        prompt="sub prompt",
        steps=[
            ReActStep(
                turn=turn(
                    0,
                    "Thought: count them.\nAction: database_querying\n"
                    "Action Input: SELECT COUNT(*) FROM Objects",
                ),
                action=ToolCall(
                    name="database_querying",
                    raw_input="SELECT COUNT(*) FROM Objects",
                    args=["SELECT COUNT(*) FROM Objects"],
                    observation="[(2,)]",
                    error=None,
                    latency_s=0.01,
                ),
            ),
            ReActStep(
                turn=turn(1, "Thought: I now know.\nFinal Answer: 2"), final_answer="2"
            ),
        ],
        final_answer="2",
        status="final_answer",
        usage=Usage(llm_calls=2, tool_calls=1),
    )


def transcript() -> Transcript:
    return Transcript(
        prompt="main prompt",
        steps=[
            ReActStep(
                turn=turn(
                    0,
                    "Thought: ask the object memory.\nAction: object_memory_querying\n"
                    "Action Input: how many people are there?",
                ),
                action=ToolCall(
                    name="object_memory_querying",
                    raw_input="how many people are there?",
                    args=["how many people are there?"],
                    observation="2",
                    error=None,
                    latency_s=1.5,
                    sub_transcript=sub_transcript(),
                ),
            ),
            ReActStep(
                turn=turn(1, "Thought: hmm"),
                parse_error="Could not parse LLM output: `Thought: hmm`",
            ),
            ReActStep(turn=turn(2, "Final Answer: 2"), final_answer="2"),
        ],
        final_answer="2",
        status="final_answer",
        usage=Usage(llm_calls=3, tool_calls=1, parse_errors=1, wall_s=4.0),
    )


def test_a_transcript_survives_a_json_round_trip() -> None:
    original = transcript()

    restored = Transcript.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original
    assert restored.steps[0].action.sub_transcript.final_answer == "2"
    assert restored.usage == Usage(llm_calls=3, tool_calls=1, parse_errors=1, wall_s=4.0)


def test_to_dict_nests_the_sub_agents_episode() -> None:
    data = transcript().to_dict()

    assert data["status"] == "final_answer"
    assert data["usage"]["llm_calls"] == 3
    assert data["steps"][0]["action"]["sub_transcript"]["prompt"] == "sub prompt"
    assert data["steps"][1]["action"] is None
    assert data["steps"][1]["parse_error"] == "Could not parse LLM output: `Thought: hmm`"


def test_render_rebuilds_the_log_with_the_sub_agent_indented() -> None:
    assert transcript().render() == (
        "Thought: ask the object memory.\n"
        "Action: object_memory_querying\n"
        "Action Input: how many people are there?\n"
        "    Thought: count them.\n"
        "    Action: database_querying\n"
        "    Action Input: SELECT COUNT(*) FROM Objects\n"
        "    Observation: [(2,)]\n"
        "    Thought: I now know.\n"
        "    Final Answer: 2\n"
        "Observation: 2\n"
        "Thought: hmm\n"
        "Parse error: Could not parse LLM output: `Thought: hmm`\n"
        "Final Answer: 2"
    )

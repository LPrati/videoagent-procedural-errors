"""Full record of one agent episode: what was asked, answered and observed.

The reference recovered its log by capturing ``AgentExecutor``'s verbose
stdout and stripping ANSI codes, which loses everything a study needs (token
counts, latencies, which tool produced which observation, the nested
sub-agent's own episode). A transcript keeps it structured instead:
``to_dict``/``from_dict`` persist it next to the run record and ``render``
rebuilds the plain-text log the reference printed.
"""

from dataclasses import dataclass
from textwrap import indent

from video_agent.models.types import Usage

NESTED_INDENT = "    "
"""Prefix for a sub-agent's log inside its caller's, so nesting is visible."""


@dataclass
class LLMTurn:
    """One completion: what the model wrote and what it cost.

    ``reasoning`` is the thinking block the server split out of the
    completion (``reasoning_content``), ``None`` when the model did not think
    or the server did not separate it. It is kept because with thinking on
    almost every token a turn was billed for is in there and in no other
    field, so a transcript without it cannot explain its own cost.
    """

    index: int  # position in the loop, from 0
    completion: str
    usage: Usage
    latency_s: float
    model: str
    finish_reason: str | None
    reasoning: str | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "completion": self.completion,
            "usage": self.usage.to_dict(),
            "latency_s": self.latency_s,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMTurn":
        # ``get``: the key is optional, so records without it still load.
        return cls(
            index=int(data["index"]),
            completion=data["completion"],
            usage=Usage.from_dict(data["usage"]),
            latency_s=float(data["latency_s"]),
            model=data["model"],
            finish_reason=data["finish_reason"],
            reasoning=data.get("reasoning"),
        )


@dataclass
class ToolCall:
    """One tool dispatch: the raw action input, the typed args and the result.

    ``raw_input`` is kept next to ``args`` because the model writes the
    argument tuple as free text and the two disagree exactly when parsing
    failed (``args is None``). ``sub_transcript`` holds the episode of a tool
    that is itself an agent.
    """

    name: str
    raw_input: str
    args: list | None
    observation: str
    error: str | None
    latency_s: float
    sub_transcript: "Transcript | None" = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "raw_input": self.raw_input,
            "args": None if self.args is None else list(self.args),
            "observation": self.observation,
            "error": self.error,
            "latency_s": self.latency_s,
            "sub_transcript": (
                None if self.sub_transcript is None else self.sub_transcript.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        sub = data.get("sub_transcript")
        return cls(
            name=data["name"],
            raw_input=data["raw_input"],
            args=None if data["args"] is None else list(data["args"]),
            observation=data["observation"],
            error=data["error"],
            latency_s=float(data["latency_s"]),
            sub_transcript=None if sub is None else Transcript.from_dict(sub),
        )


@dataclass
class ReActStep:
    """One iteration of the loop: a completion and what the agent did with it.

    Exactly one of ``action``, ``final_answer`` and ``parse_error`` is set.
    """

    turn: LLMTurn
    action: ToolCall | None = None
    final_answer: str | None = None
    parse_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "turn": self.turn.to_dict(),
            "action": None if self.action is None else self.action.to_dict(),
            "final_answer": self.final_answer,
            "parse_error": self.parse_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReActStep":
        action = data.get("action")
        return cls(
            turn=LLMTurn.from_dict(data["turn"]),
            action=None if action is None else ToolCall.from_dict(action),
            final_answer=data["final_answer"],
            parse_error=data["parse_error"],
        )


@dataclass
class Transcript:
    """One agent episode, from the first prompt to the answer or the failure.

    Only the initial prompt is stored: every later one is the same template
    with the scratchpad rebuilt from the steps, so keeping them all would
    duplicate the whole episode once per iteration.
    """

    prompt: str
    steps: list[ReActStep]
    final_answer: str | None
    status: str  # "final_answer" | "max_iterations" | "parse_failed" | "error"
    usage: Usage

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "status": self.status,
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Transcript":
        return cls(
            prompt=data["prompt"],
            steps=[ReActStep.from_dict(step) for step in data["steps"]],
            final_answer=data["final_answer"],
            status=data["status"],
            usage=Usage.from_dict(data["usage"]),
        )

    def render(self) -> str:
        """Return the episode as the plain-text log the reference printed.

        Each step shows the completion followed by the observation it earned;
        a sub-agent's own log is indented above the observation it produced.
        """
        blocks: list[str] = []
        for step in self.steps:
            block = step.turn.completion
            if step.action is not None:
                if step.action.sub_transcript is not None:
                    nested = step.action.sub_transcript.render()
                    block += "\n" + indent(nested, NESTED_INDENT)
                block += f"\nObservation: {step.action.observation}"
            elif step.parse_error is not None:
                block += f"\nParse error: {step.parse_error}"
            blocks.append(block)
        return "\n".join(blocks)

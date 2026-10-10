"""The seam between the tools and whatever agent framework drives them.

The tools of ``agent.tools`` describe themselves as ``ToolSpec``s with typed
parameters and are called with typed positional arguments. How the model is
asked to produce those arguments is the runner's business: ``react`` packs
them into the one string tuple the reference used, while a structured
framework (function calling, ADK) would pass them straight through. Keeping
that decision out of the tools is what makes the orchestrator swappable.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

from video_agent.agent.transcript import Transcript
from video_agent.models.types import Usage

AGENT_STOPPED_MESSAGE = "Agent stopped due to iteration limit or time limit."
"""What LangChain's executor returned when it ran out of iterations.

An ``AgentResult`` with no answer means the same thing here; a caller that
has to show *something* to another agent uses this string, as the reference
did when its object-memory sub-agent gave up.
"""


@dataclass(frozen=True)
class ParamSpec:
    """One tool parameter: its name for the prompt, its kind for the cast."""

    name: str
    kind: Literal["str", "int"]


@dataclass(frozen=True)
class ToolResult:
    """A tool's observation plus, for tools that are agents, their episode."""

    observation: str
    sub_transcript: Transcript | None = None


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the agent sees it: name, description, parameters, callable.

    ``fn`` receives the parameters as positional arguments in ``params``
    order, already cast to their ``kind``, and returns either the observation
    or a ``ToolResult``. ``input_name`` is the single free-text argument the
    reference's Python function declared — ``input_tuple`` for the tools that
    packed two values into one string — and exists because it reached the
    model inside the rendered tool description.
    """

    name: str
    description: str
    params: tuple[ParamSpec, ...]
    input_name: str
    fn: Callable[..., "str | ToolResult"]


@dataclass
class AgentResult:
    """Outcome of one episode: the answer if there is one, plus its record."""

    answer: str | None
    status: str
    transcript: Transcript
    usage: Usage


class AgentRunner(Protocol):
    """Drives one question to an answer using ``tools`` and a prompt template.

    A runner that dies mid-episode may attach the ``Transcript`` it had built
    to the exception it raises, under the attribute ``transcript``; callers
    read it with ``getattr`` and record the partial episode rather than
    losing it (``react.ReActError``).
    """

    def run(
        self, question: str, tools: Sequence[ToolSpec], *, prompt_template: str
    ) -> AgentResult:
        """Answer ``question``, recording every step in the result's transcript."""
        ...


def render_tool_descriptions(tools: Sequence[ToolSpec]) -> str:
    """Render the ``{tools}`` block, naming each tool twice as the reference did.

    ``render_text_description`` emitted ``f"{name}: {description}"``, but the
    ``@tool`` decorator had already replaced the docstring with
    ``f"{name}{signature} - {docstring}"`` (langchain-core 0.1.14,
    ``StructuredTool.from_function``), so every line carried the name, the
    signature and only then the prose the author wrote.
    """
    return "\n".join(
        f"{tool.name}: {tool.name}({tool.input_name}) - {tool.description}"
        for tool in tools
    )


def render_tool_names(tools: Sequence[ToolSpec]) -> str:
    """Render the ``{tool_names}`` placeholder, as the reference joined it."""
    return ", ".join(tool.name for tool in tools)

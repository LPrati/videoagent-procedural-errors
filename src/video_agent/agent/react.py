"""ReAct orchestration, reproduced from LangChain 0.1.x without LangChain.

The reference drove the agent with ``create_react_agent`` + ``AgentExecutor``
and read the episode back from captured stdout. Everything that decided an
answer therefore lived in a dependency: how the prompt was rendered, which
regex accepted the model's action, what the executor said back when it did
not. This module is that behavior written out — same regexes, same messages,
same scratchpad — so it can be pinned by tests and varied on purpose through
``ReActConfig`` instead of by upgrading a library.

The whole rendered prompt goes out as a single user message and the model is
stopped at ``"\\nObservation"``, exactly as the reference's chat model was.
"""

import ast
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Sequence

from video_agent.agent.config import ReActConfig
from video_agent.agent.runner import (
    AgentResult,
    ParamSpec,
    ToolResult,
    ToolSpec,
    render_tool_descriptions,
    render_tool_names,
)
from video_agent.agent.transcript import LLMTurn, ReActStep, ToolCall, Transcript
from video_agent.models.interfaces import ChatModel
from video_agent.models.types import ChatMessage, Usage

FINAL_ANSWER_ACTION = "Final Answer:"
# The unbalanced quote is the reference's own typo, reproduced: the model
# reads this string, so fixing it would change the prompt.
MISSING_ACTION_MESSAGE = "Invalid Format: Missing 'Action:' after 'Thought:"
MISSING_ACTION_INPUT_MESSAGE = "Invalid Format: Missing 'Action Input:' after 'Action:'"
FINAL_ANSWER_AND_ACTION_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)
"""Interpolated as ``f"{message}: {text}"`` upstream, hence the doubled colon."""
INVALID_RESPONSE_OBSERVATION = "Invalid or incomplete response"
"""What the executor observed on a parse error it would not explain to the model."""

INVALID_TUPLE_OBSERVATION = "Invalid input tuple!"
"""What the reference's two-argument tools returned on an unusable argument."""

UNKNOWN_TOOL_ERROR = "unknown tool"
"""``ToolCall.error`` of an action naming something that is not a tool.

The model writes the action name as free text, so ``name`` can be anything —
a misspelt call, a whole paragraph. This is the executor's own verdict on it,
and a reader of the transcript should trust it rather than re-parse the name.
"""

EDITED_TURN_MODEL = "human-edit"
"""``LLMTurn.model`` of a completion written by hand instead of by the model.

A ``ReActSession`` can be advanced with a completion typed by hand, to ask what the
loop would have done with an action the model never produced. The turn is
recorded like any other, so the transcript has to say where it came from: a
record whose turns name a served checkpoint and one names this is still
evidence, one that hides the edit is not.
"""

EDITED_FINISH_REASON = "edited"
"""``LLMTurn.finish_reason`` of an injected completion: no server ended it."""

SESSION_RUNNING = "running"
"""Status of an episode that has neither answered nor given up yet.

It is deliberately not one of the terminal statuses a ``Transcript`` records,
and it exists because an episode can be stopped and picked up again: a
record written mid-episode says ``running`` instead of pretending to be one
of the outcomes the evaluation counts.
"""

# Verbatim from ``ReActSingleInputOutputParser``: the first pattern accepts an
# action, the other two decide which half is missing when it does not.
_ACTION_PATTERN = re.compile(
    r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", re.DOTALL
)
_ACTION_ONLY_PATTERN = re.compile(r"Action\s*\d*\s*:[\s]*(.*?)", re.DOTALL)
_ACTION_INPUT_ONLY_PATTERN = re.compile(
    r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", re.DOTALL
)


@dataclass(frozen=True)
class Parsed:
    """What one completion asked for: an action, an answer, or neither.

    ``send_to_llm`` carries the reference's distinction between a parse error
    worth explaining to the model (``observation`` says what was missing) and
    one that only earns ``"Invalid or incomplete response"``.
    """

    kind: Literal["action", "finish", "error"]
    tool: str | None = None
    tool_input: str | None = None
    answer: str | None = None
    error: str | None = None
    observation: str | None = None
    send_to_llm: bool = False


class ReActParseError(Exception):
    """A completion the parser could not turn into an action or an answer.

    The analogue of LangChain's ``OutputParserException``, kept for callers
    that prefer failing loudly: ``ReActRunner`` never raises it, since a parse
    error is data about the model and belongs in the transcript.
    """

    def __init__(
        self, error: str, *, observation: str | None = None, send_to_llm: bool = False
    ) -> None:
        super().__init__(error)
        self.error = error
        self.observation = observation
        self.send_to_llm = send_to_llm

    @classmethod
    def from_parsed(cls, parsed: Parsed) -> "ReActParseError":
        return cls(
            parsed.error or "",
            observation=parsed.observation,
            send_to_llm=parsed.send_to_llm,
        )


class ReActError(Exception):
    """A failure mid-episode, carrying the transcript of what had already run.

    Without it, a model call that fails (for example, a turn that exceeds
    the request timeout) would raise out of ``run`` and discard every
    completed turn before it. Error analysis needs those steps and their
    token counts, so a failing chat model is wrapped and the partial
    transcript handed over. Only the model call is wrapped — a tool that
    raises under ``tool_errors_as_observations=False`` still crashes as the
    reference did. The original exception stays as ``__cause__`` and its
    text is repeated in the message, so nothing a caller would log is lost.
    """

    def __init__(self, message: str, *, transcript: Transcript) -> None:
        super().__init__(message)
        self.transcript = transcript


def parse_react_output(text: str) -> Parsed:
    """Parse one completion the way ``ReActSingleInputOutputParser`` did.

    An action wins over a final answer only when the answer is absent: a
    completion holding both is an error, because the executor could not tell
    whether the model had finished.
    """
    includes_answer = FINAL_ANSWER_ACTION in text
    match = _ACTION_PATTERN.search(text)
    if match:
        if includes_answer:
            return Parsed(
                kind="error", error=f"{FINAL_ANSWER_AND_ACTION_MESSAGE}: {text}"
            )
        # ``strip(" ")`` then ``strip('"')``: the prompt asks for double-quoted
        # string arguments, and the quotes are not part of the value.
        return Parsed(
            kind="action",
            tool=match.group(1).strip(),
            tool_input=match.group(2).strip(" ").strip('"'),
        )
    if includes_answer:
        return Parsed(
            kind="finish", answer=text.split(FINAL_ANSWER_ACTION)[-1].strip()
        )
    error = f"Could not parse LLM output: `{text}`"
    if not _ACTION_ONLY_PATTERN.search(text):
        return Parsed(
            kind="error",
            error=error,
            observation=MISSING_ACTION_MESSAGE,
            send_to_llm=True,
        )
    if not _ACTION_INPUT_ONLY_PATTERN.search(text):
        return Parsed(
            kind="error",
            error=error,
            observation=MISSING_ACTION_INPUT_MESSAGE,
            send_to_llm=True,
        )
    return Parsed(kind="error", error=error)


def render_prompt(
    template: str, *, tools: Sequence[ToolSpec], question: str, scratchpad: str
) -> str:
    """Fill the four placeholders of a ReAct prompt template.

    Substitution is by ``str.replace`` and never ``str.format``: task prompts
    quote JSON and SQL, whose braces ``format`` would try to interpret.
    """
    return (
        template.replace("{tools}", render_tool_descriptions(tools))
        .replace("{tool_names}", render_tool_names(tools))
        .replace("{input}", question)
        .replace("{agent_scratchpad}", scratchpad)
    )


def invalid_tool_message(name: str, tools: Sequence[ToolSpec]) -> str:
    """The executor's observation for an action naming a tool that does not exist."""
    return f"{name} is not a valid tool, try one of [{render_tool_names(tools)}]."


def _cast_arguments(params: Sequence[ParamSpec], values: Sequence) -> list | None:
    arguments: list = []
    for param, value in zip(params, values):
        if param.kind == "int":
            try:
                arguments.append(int(value))
            except (TypeError, ValueError):
                return None
        else:
            arguments.append(str(value))
    return arguments


def _literal_values(raw: str) -> list | None:
    try:
        value = ast.literal_eval(raw)
    except (MemoryError, SyntaxError, TypeError, ValueError):
        return None
    if not isinstance(value, (list, tuple)):
        return None
    return list(value)


def _lenient_values(raw: str) -> list | None:
    """Recover ``(question, 3)`` from text ``literal_eval`` chokes on.

    Models routinely emit unbalanced or single quotes inside the argument
    tuple; splitting on the last comma keeps the string argument intact.
    """
    stripped = raw.strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1]
    head, comma, tail = stripped.rpartition(",")
    if not comma:
        return None
    return [head.strip().strip("\"'"), tail.strip().strip("\"'")]


def parse_tool_arguments(
    spec: ToolSpec, raw: str, *, lenient: bool = False
) -> list | None:
    """Turn one action input into typed arguments; ``None`` when it is unusable.

    Single-argument tools take the raw string, as LangChain handed it over.
    Multi-argument tools receive one string holding a tuple, which the
    reference read with ``ast.literal_eval`` and rejected when its length did
    not match; ``lenient`` adds a fallback for near-miss quoting.
    """
    if len(spec.params) == 1 and spec.params[0].kind == "str":
        return [raw]
    values = _literal_values(raw)
    if values is None and lenient:
        values = _lenient_values(raw)
    if values is None or len(values) != len(spec.params):
        return None
    return _cast_arguments(spec.params, values)


def observation_text(call: ToolCall, *, observation_padding: bool) -> str:
    """Render one observation the way the scratchpad received it.

    Every tool of the reference returned ``'\\n' + text + '\\n'``, its
    rejected-argument guard included. Messages the executor produced on its
    own — an unknown tool, a parse error — never went through a wrapper and
    were not padded, and neither is our tool-exception fallback, which stands
    where the reference simply crashed.
    """
    wrapped = call.error is None or call.error == "invalid arguments"
    if observation_padding and wrapped:
        return f"\n{call.observation}\n"
    return call.observation


def scratchpad_entry(step: ReActStep, *, observation_padding: bool) -> str:
    """What one step added to the scratchpad of the prompt after it.

    Deriving the scratchpad from the steps, rather than accumulating it as
    a string, is what lets an episode be rebuilt from a saved transcript and
    continued, and what makes editing an observation enough to change the
    next prompt. A final
    answer ends the episode and so contributes nothing.

    ``Parsed.send_to_llm`` — whether a parse error was explained to the model
    or only hidden behind "Invalid or incomplete response" — is not persisted
    in a transcript, so it is recomputed from the completion; that is sound
    because ``parse_react_output`` is pure.
    """
    if step.action is not None:
        observation = observation_text(
            step.action, observation_padding=observation_padding
        )
        return f"{step.turn.completion}\nObservation: {observation}\nThought: "
    if step.parse_error is not None:
        parsed = parse_react_output(step.turn.completion)
        # Reference: an explainable error shows the model its own text and
        # what was missing; anything else hides both.
        if parsed.send_to_llm:
            logged, observation = step.turn.completion, parsed.observation or ""
        else:
            logged = parsed.error or ""
            observation = INVALID_RESPONSE_OBSERVATION
        return f"{logged}\nObservation: {observation}\nThought: "
    return ""


def scratchpad_from_steps(
    steps: Sequence[ReActStep], *, observation_padding: bool
) -> str:
    """Rebuild ``{agent_scratchpad}`` from the steps an episode has run."""
    return "".join(
        scratchpad_entry(step, observation_padding=observation_padding)
        for step in steps
    )


def _usage_from_steps(steps: Sequence[ReActStep]) -> Usage:
    """Recount an episode's cost from its steps alone.

    A resumed episode inherits steps and not the counters that were kept
    beside them, and it must not double-count what a saved record already
    reports; the counters are all derivable from the steps, so they are
    derived. ``wall_s`` is not: it is time this process spent, and the
    resumed episode starts measuring its own.
    """
    usage = Usage()
    for step in steps:
        usage.llm_calls += step.turn.usage.llm_calls
        usage.tokens_in += step.turn.usage.tokens_in
        usage.tokens_out += step.turn.usage.tokens_out
        if step.action is not None:
            usage.tool_calls += 1
            if step.action.error is not None:
                usage.tool_errors += 1
        if step.parse_error is not None:
            usage.parse_errors += 1
    return usage


class ReActRunner:
    """Runs the ReAct loop of the reference against any ``ChatModel``.

    One instance can serve as its own ``sub_runner``: the nested episode gets
    its own transcript, which the parent's ``ToolCall`` carries.
    """

    def __init__(self, llm: ChatModel, config: ReActConfig = ReActConfig()) -> None:
        self.llm = llm
        self.config = config

    def run(
        self, question: str, tools: Sequence[ToolSpec], *, prompt_template: str
    ) -> AgentResult:
        """Answer ``question``, returning the answer and the whole episode.

        The loop itself lives in ``ReActSession``; running it to exhaustion is
        all this method is. A ``ReActError`` from a dead backend propagates,
        carrying the steps that had already run.
        """
        session = self.session(question, tools, prompt_template=prompt_template)
        while session.active:
            session.step()
        return session.result()

    def session(
        self,
        question: str,
        tools: Sequence[ToolSpec],
        *,
        prompt_template: str,
        steps: Sequence[ReActStep] = (),
    ) -> "ReActSession":
        """Open an episode this runner drives one turn at a time.

        ``steps`` resumes an episode that was already recorded: the scratchpad
        and the cost counters are rebuilt from them, so the prefix of a saved
        ``run.json`` can be picked up where it stopped.
        """
        return ReActSession(
            self, question, tools, prompt_template=prompt_template, steps=steps
        )

    def _dispatch(
        self,
        parsed: Parsed,
        tools: Sequence[ToolSpec],
        by_name: dict[str, ToolSpec],
        usage: Usage,
    ) -> ToolCall:
        """Run the requested tool, turning every failure into an observation."""
        name = parsed.tool or ""
        raw = parsed.tool_input or ""
        usage.tool_calls += 1
        started = time.perf_counter()

        def call(
            observation: str,
            *,
            args: list | None = None,
            error: str | None = None,
            sub: Transcript | None = None,
        ) -> ToolCall:
            return ToolCall(
                name=name,
                raw_input=raw,
                args=args,
                observation=observation,
                error=error,
                latency_s=time.perf_counter() - started,
                sub_transcript=sub,
            )

        spec = by_name.get(name)
        if spec is None:
            usage.tool_errors += 1
            return call(invalid_tool_message(name, tools), error=UNKNOWN_TOOL_ERROR)

        arguments = parse_tool_arguments(
            spec, raw, lenient=self.config.lenient_tuple_parsing
        )
        if arguments is None:
            usage.tool_errors += 1
            return call(INVALID_TUPLE_OBSERVATION, error="invalid arguments")

        try:
            result = spec.fn(*arguments)
        except Exception as error:  # a tool failure is data, not a crash
            if not self.config.tool_errors_as_observations:
                raise
            usage.tool_errors += 1
            return call(
                f"Tool error: {error}",
                args=arguments,
                error=f"{type(error).__name__}: {error}",
            )
        if isinstance(result, ToolResult):
            return call(result.observation, args=arguments, sub=result.sub_transcript)
        return call(str(result), args=arguments)

    def _observation_text(self, call: ToolCall) -> str:
        """Pad the observation as the reference's tool wrappers did.

        A thin wrapper over ``observation_text``, which the scratchpad is
        rebuilt with.
        """
        return observation_text(
            call, observation_padding=self.config.observation_padding
        )


class ReActSession:
    """One episode of the loop, advanced a turn at a time.

    ``ReActRunner.run`` is this class driven to exhaustion. Stepping an
    episode explicitly supports error analysis in a way a batch runner
    cannot: an episode can be rebuilt from the steps of a saved ``run.json``, continued one turn
    at a time, stopped after any of them, and its record rendered while it is
    still unfinished.

    The scratchpad is not accumulated as a string — it is derived from
    ``steps`` on every prompt — which is what makes editing the last
    observation a one-line operation instead of a rewrite of the episode.
    """

    def __init__(
        self,
        runner: "ReActRunner",
        question: str,
        tools: Sequence[ToolSpec],
        *,
        prompt_template: str,
        steps: Sequence[ReActStep] = (),
    ) -> None:
        inherited = deepcopy(list(steps))
        # An episode that answered is over: continuing past a final answer
        # would produce a transcript with two of them, which no reader of the
        # record could interpret.
        if any(step.final_answer is not None for step in inherited):
            raise ValueError("cannot resume an episode that already answered")
        self.question = question
        self.tools = list(tools)
        self.prompt_template = prompt_template
        self.config = runner.config
        self.llm = runner.llm
        self.steps: list[ReActStep] = inherited
        self.usage = _usage_from_steps(inherited)
        self.answer: str | None = None
        self.status = (
            "max_iterations"
            if len(self.steps) >= self.config.max_iterations
            else SESSION_RUNNING
        )
        self.initial_prompt = render_prompt(
            prompt_template, tools=self.tools, question=question, scratchpad=""
        )
        # The provider payload of the last reply, for a UI that wants to show
        # what the server actually sent. It is deliberately not persisted: a
        # transcript keeps the documented record fields, not raw JSON.
        self.last_raw: dict | None = None
        self._runner = runner
        self._by_name = {tool.name: tool for tool in self.tools}
        self._started = time.perf_counter()

    @property
    def active(self) -> bool:
        """Whether another turn is allowed; every other status is terminal."""
        return self.status == SESSION_RUNNING

    @property
    def next_index(self) -> int:
        """The index the next turn will carry, which is how many have run."""
        return len(self.steps)

    def scratchpad(self) -> str:
        """The ``{agent_scratchpad}`` the steps so far amount to."""
        return scratchpad_from_steps(
            self.steps, observation_padding=self.config.observation_padding
        )

    def prompt(self) -> str:
        """The prompt the next turn will be asked, scratchpad included."""
        return render_prompt(
            self.prompt_template,
            tools=self.tools,
            question=self.question,
            scratchpad=self.scratchpad(),
        )

    def step(self, completion: str | None = None) -> ReActStep:
        """Run one turn and return the step it produced.

        ``completion`` replaces the model call with text written by hand,
        which is how one asks what the loop would have done with an action the
        model never produced — without a backend, and without pretending: the
        turn is stamped ``EDITED_TURN_MODEL`` and costs nothing. Everything
        after the completion is the same code either way, so an injected turn
        exercises the loop rather than a simulation of it.
        """
        if not self.active:
            raise RuntimeError(f"the session is {self.status}, not running")
        prompt = self.prompt()
        if completion is None:
            turn = self._model_turn(prompt)
        else:
            turn = LLMTurn(
                index=self.next_index,
                completion=completion,
                usage=Usage(),
                latency_s=0.0,
                model=EDITED_TURN_MODEL,
                finish_reason=EDITED_FINISH_REASON,
                reasoning=None,
            )
        return self._advance(turn)

    def edit_observation(self, observation: str) -> ToolCall:
        """Replace what the last action observed, and return the call.

        This is the point of stepping an episode by hand: one wrong
        observation early on can decide the whole answer, and the way to show
        it is to put the right one back and let the loop go on. Only the last
        step is editable — every earlier observation is already quoted inside
        a completion the model wrote, and rewriting it would falsify the
        turns that followed.
        """
        if not self.active:
            raise RuntimeError(f"the session is {self.status}, not running")
        call = self.steps[-1].action if self.steps else None
        if call is None:
            raise RuntimeError("the last step is not a tool call")
        call.observation = observation
        return call

    def result(self) -> AgentResult:
        """The episode so far, shaped as ``run`` returns it — finished or not.

        A session that is still running yields ``status`` ``"running"``, which
        is what lets the record of an interrupted episode be written and read
        with the same code as any other.
        """
        self.usage.wall_s = time.perf_counter() - self._started
        transcript = Transcript(
            prompt=self.initial_prompt,
            steps=self.steps,
            final_answer=self.answer,
            status=self.status,
            usage=self.usage,
        )
        return AgentResult(
            answer=self.answer,
            status=self.status,
            transcript=transcript,
            usage=self.usage,
        )

    def _model_turn(self, prompt: str) -> LLMTurn:
        """Ask the model for one completion, keeping the episode if it dies."""
        call_started = time.perf_counter()
        try:
            reply = self.llm.complete(
                [ChatMessage(role="user", text=prompt)],
                stop=self.config.stop,
                max_tokens=self.config.max_tokens,
            )
        except Exception as failure:  # a dead backend must not erase the episode
            self.status = "error"
            self.usage.wall_s = time.perf_counter() - self._started
            raise ReActError(
                f"{type(failure).__name__}: {failure}",
                transcript=Transcript(
                    prompt=self.initial_prompt,
                    steps=self.steps,
                    final_answer=None,
                    status="error",
                    usage=self.usage,
                ),
            ) from failure
        self.usage = self.usage + reply.usage
        self.last_raw = reply.raw
        return LLMTurn(
            index=self.next_index,
            completion=reply.text,
            usage=reply.usage,
            latency_s=time.perf_counter() - call_started,
            model=reply.model,
            finish_reason=reply.finish_reason,
            reasoning=reply.reasoning,
        )

    def _advance(self, turn: LLMTurn) -> ReActStep:
        """Parse one completion, act on it, and record the step it became."""
        parsed = parse_react_output(turn.completion)
        if parsed.kind == "finish":
            step = ReActStep(turn=turn, final_answer=parsed.answer)
            self.answer = parsed.answer
            self.status = "final_answer"
        elif parsed.kind == "error":
            self.usage.parse_errors += 1
            step = ReActStep(turn=turn, parse_error=parsed.error)
            if not self.config.parse_errors_as_observations:
                self.status = "parse_failed"
        else:
            call = self._runner._dispatch(
                parsed, self.tools, self._by_name, self.usage
            )
            step = ReActStep(turn=turn, action=call)
        self.steps.append(step)
        if self.active and len(self.steps) >= self.config.max_iterations:
            self.status = "max_iterations"
        return step

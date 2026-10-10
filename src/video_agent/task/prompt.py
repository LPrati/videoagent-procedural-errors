"""The question the agent is asked, and the answer it is expected to return.

The reproduced agent is used as it is: one ReAct episode per recording, the
original four tools and the original prompt. All that changes for the
procedural task is the question text, which carries the SOP and the task
graph, and the shape of the Final Answer, which has to be machine readable
because the evaluation scores one verdict per step.

The question (``agent/prompts/task_question.txt``) defines order as the
CaptainCook4D annotations do: the prerequisite constraints chain
transitively, a step counts as performed after another only if it starts
after the other ends, both steps of an out-of-order pair are violations, and
a step that was not performed makes no other step a violation.

Rendering is by ``str.replace`` and never ``str.format``: the question
embeds a JSON schema, and every brace in it would be a format field.
"""

import json
import re
from typing import Iterator

from video_agent.task.annotations import Sop, TaskGraph

ANSWER_SCHEMA = (
    '{"steps": [{"step_id": <int>, "executed": "yes"|"no"|"unsure", '
    '"segments": [<first>, <last>] or null, '
    '"order": "ok"|"violation"|"unsure"|null, '
    '"violated_prerequisite": {"step_id": <int>, '
    '"segments": [<first>, <last>] or null} or null}]}'
)
"""The Final Answer format, quoted into the question and parsed back out.

One entry per SOP step: the execution verdict with its anchor, and the order
verdict with the step it is out of order with and that step's anchor (the
``violated_prerequisite`` key keeps its name although the cited step may be a
prerequisite or a dependent). Segment ids rather than seconds, because
segments are what the agent's tools speak in; the conversion to time anchors
happens in ``verdicts``.
"""

_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


def _render_steps(sop: Sop) -> str:
    """One line per SOP step, in canonical order."""
    return "\n".join(
        f"step {step_id}: {description}" for step_id, description in sop.steps
    )


def _render_prerequisites(sop: Sop, graph: TaskGraph) -> str:
    """One line per SOP step naming what must come before it.

    Steps without prerequisites are listed too, so the model can tell a step
    that is free to move from one the question simply did not mention.
    Prerequisites are named in SOP order, the same order the step list uses.
    """
    position = {step_id: index for index, (step_id, _) in enumerate(sop.steps)}
    lines: list[str] = []
    for step_id, _ in sop.steps:
        prerequisites = sorted(
            graph.prerequisites.get(step_id, frozenset()),
            key=lambda step: (position.get(step, len(position)), step),
        )
        if prerequisites:
            listed = ", ".join(str(step) for step in prerequisites)
            lines.append(f"step {step_id} requires steps {listed}")
        else:
            lines.append(f"step {step_id} has no prerequisites")
    return "\n".join(lines)


def build_question(
    sop: Sop, graph: TaskGraph, n_segments: int, template: str
) -> str:
    """Render the task question for one recording.

    ``n_segments`` comes from the video's own memory, so the ranges the model
    is asked for are ranges that exist.
    """
    return (
        template.replace("{recipe_name}", sop.recipe_name)
        .replace("{n_segments}", str(n_segments))
        .replace("{steps}", _render_steps(sop))
        .replace("{prerequisites}", _render_prerequisites(sop, graph))
        .replace("{schema}", ANSWER_SCHEMA)
    )


def _json_objects(text: str) -> Iterator[object]:
    """Yield every JSON value that decodes from a ``{`` of ``text``, in order.

    One ``raw_decode`` attempt per ``{``, and the span of a value that did
    decode is skipped, so the braces nested inside an answer are never tried
    again: the scan costs one attempt per *top-level* brace rather than one
    per brace in the text.
    """
    decoder = json.JSONDecoder()
    index = 0
    while True:
        start = text.find("{", index)
        if start < 0:
            return
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            index = start + 1  # not the start of an object; try the next brace
            continue
        index = end
        yield value


def final_answer_candidates(text: str) -> list[dict]:
    """Every object of ``text`` that looks like an answer, in the order written.

    A candidate is a top-level JSON object carrying a ``"steps"`` list —
    including an empty one, which is a candidate the reader will refuse but
    which the record still counts, because "how many answers did the model
    write?" is a question about the episode and not about the accepted one.

    Models wrap JSON in code fences and in commentary however they like, and
    two common shapes turn a first-``{``-to-last-``}`` slice into invalid
    JSON: the answer, a stray ``</think>``, and then the *same* answer again;
    and the answer followed by "Wait, I need to make sure..." and a paragraph
    of second thoughts. Hence the scan.
    """
    if not text:
        return []
    stripped = _FENCE.sub("", text)
    return [
        payload
        for payload in _json_objects(stripped)
        if isinstance(payload, dict) and isinstance(payload.get("steps"), list)
    ]


def parse_final_answer(text: str) -> dict | None:
    """Recover the answer object from a Final Answer, or ``None`` if there is none.

    The **last** non-empty candidate wins: a model that writes twice is
    revising, and the answer it stands by is the one it wrote last.

    An empty ``steps`` list is not an answer. It parses, and accepting it
    would give a run status ``ok`` with every step unclassified, a silent
    failure that reads in the tables like a system that abstained on purpose.
    Refusing it turns the run into the ``parse_failed`` it is.

    Returning ``None`` rather than raising is deliberate: an unparseable
    answer is a run outcome the record has to keep (status ``parse_failed``),
    not an exception that would lose the episode.
    """
    for payload in reversed(final_answer_candidates(text)):
        if payload["steps"]:
            return payload
    return None

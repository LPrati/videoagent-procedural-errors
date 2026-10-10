"""The prompt templates, shipped as package data.

The three ReAct prompts of the reference are verbatim; the procedural-task
prompt and its question template are specific to this work.

They are data, not code: a run records which template it used by hash, so a
result can be traced to the exact wording that produced it.
"""

import hashlib
from importlib.resources import files

PROMPT_NAMES = (
    "prompt",
    "multiple_choice_prompt",
    "database_query_prompt",
    "task_prompt",
    "task_question",
)
"""Open-ended orchestrator, multiple-choice orchestrator, object-memory sub-agent,
procedural-verdict orchestrator, and the question template that goes with it.

``task_prompt`` differs from ``prompt`` in one line — the Final Answer must be
the JSON object the question asks for — and ``task_question`` is not a ReAct
prompt at all but the question rendered into it (see ``task.prompt``): it
lists the recipe steps and their ordering constraints, which chain
transitively, and asks for an execution and an order verdict per step. A
step counts as performed after another only if it starts after the other
ends, both steps of an out-of-order pair are violations, and a step that was
not performed makes no other step a violation. Both are here because a run
records the templates it used by hash.
"""


def load_prompt(name: str) -> str:
    """Return the template text of a named prompt."""
    if name not in PROMPT_NAMES:
        raise ValueError(f"unknown prompt {name!r}, expected one of {PROMPT_NAMES}")
    return (files(__package__) / f"{name}.txt").read_text(encoding="utf-8")


def prompt_sha256(text: str) -> str:
    """Digest identifying a template in a run record."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

import hashlib

import pytest

from video_agent.agent.prompts import PROMPT_NAMES, load_prompt, prompt_sha256
from video_agent.memory.object_db import SCHEMA

PLACEHOLDERS = ("{tools}", "{tool_names}", "{input}", "{agent_scratchpad}")

# ``task_question`` is not a ReAct prompt: it is the question rendered into
# one, so it has its own placeholders and is checked in test_task_prompt.py.
QUESTION_TEMPLATES = ("task_question",)
REACT_PROMPTS = tuple(name for name in PROMPT_NAMES if name not in QUESTION_TEMPLATES)

# Digests of the reference's templates: a change here is a change of method.
# The task prompt adapts the reference one and is checked against it instead.
DIGESTS = {
    "prompt": "14f6d7fe41ecec91d6c91a328ff30e6185b57a9ca48adf7a02b1e934c86dc631",
    "multiple_choice_prompt": "1f42a6e73964cfb5375cc3e5807a5d22fb92e119644526361629641acede869b",
    "database_query_prompt": "86572af36c8e05034070ae503e7f76c1b355e580d8cb70c21b36274753f63b2b",
}


@pytest.mark.parametrize("name", REACT_PROMPTS)
def test_every_react_prompt_has_the_four_placeholders(name: str) -> None:
    text = load_prompt(name)

    for placeholder in PLACEHOLDERS:
        assert placeholder in text
    assert text.rstrip().endswith("Thought: {agent_scratchpad}")


@pytest.mark.parametrize("name", sorted(DIGESTS))
def test_the_prompts_are_the_reference_ones(name: str) -> None:
    assert prompt_sha256(load_prompt(name)) == DIGESTS[name]


@pytest.mark.parametrize("name", QUESTION_TEMPLATES)
def test_a_question_template_is_not_mistaken_for_a_react_prompt(name: str) -> None:
    text = load_prompt(name)

    for placeholder in PLACEHOLDERS:
        assert placeholder not in text


def test_prompt_sha256_is_the_digest_of_the_utf8_text() -> None:
    assert prompt_sha256("boats") == hashlib.sha256(b"boats").hexdigest()


def test_an_unknown_prompt_is_a_contract_violation() -> None:
    with pytest.raises(ValueError):
        load_prompt("../prompt")


def test_the_sub_agent_prompt_describes_the_tables_it_will_query() -> None:
    text = load_prompt("database_query_prompt")

    for table in ("Objects", "Segments", "Objects_Segments"):
        assert f"TABLE {table}(" in text
        assert f"CREATE TABLE {table}(" in SCHEMA


@pytest.mark.parametrize("name", REACT_PROMPTS)
def test_the_scratchpad_is_the_last_thing_a_react_prompt_says(name: str) -> None:
    """Nothing may follow ``{agent_scratchpad}`` but whitespace.

    The scratchpad is rebuilt from the steps of a saved episode and rendered
    into the template again, and a reader can show it as the tail of the
    prompt. Both hold only while the placeholder appears once and last: a
    template with a second one, or with instructions after it, would make the
    reconstruction ambiguous.
    """
    text = load_prompt(name)

    assert text.count("{agent_scratchpad}") == 1
    _, _, after = text.partition("{agent_scratchpad}")
    assert after.strip() == ""

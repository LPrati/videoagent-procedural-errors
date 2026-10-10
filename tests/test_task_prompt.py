"""The task question the agent is given, and the answer it is parsed out of."""

import json
import re

import pytest

from video_agent.agent.prompts import load_prompt
from video_agent.task.annotations import Sop, TaskGraph
from video_agent.task.prompt import (
    ANSWER_SCHEMA,
    build_question,
    final_answer_candidates,
    parse_final_answer,
)

from tests._paths import load_annotations

REQUIREMENT = re.compile(r"^step (\d+) requires steps (.+)$", re.MULTILINE)

SOP = Sop(
    activity_id=1,
    recipe_name="Microwave Egg Sandwich",
    steps=((3, "Coat a 6-oz. ramekin cup"), (1, "Pour 1 egg into the cup"), (4, "Microwave the cup")),
)
GRAPH = TaskGraph(
    activity_id=1,
    prerequisites={3: frozenset(), 1: frozenset({3}), 4: frozenset({1, 3})},
    edges=frozenset({(3, 1), (1, 4), (3, 4)}),
)


def question() -> str:
    return build_question(SOP, GRAPH, 42, load_prompt("task_question"))


def test_the_question_lists_every_step_in_sop_order() -> None:
    text = question()

    for step_id, description in SOP.steps:
        assert f"step {step_id}: {description}" in text
    assert text.index("step 3:") < text.index("step 1:") < text.index("step 4:")


def test_the_question_lists_the_prerequisites_of_every_step() -> None:
    text = question()

    assert "step 3 has no prerequisites" in text
    assert "step 1 requires steps 3" in text
    # Prerequisites are named in SOP order, like the step list above them.
    assert "step 4 requires steps 3, 1" in text


def test_the_question_names_the_recipe_and_the_size_of_the_grid() -> None:
    text = question()

    assert "Microwave Egg Sandwich" in text
    assert "42 segments" in text


def test_the_question_carries_the_answer_schema_and_the_order_definition() -> None:
    text = question()

    assert ANSWER_SCHEMA in text
    assert "unsure" in text
    # Order as the CaptainCook4D annotations define it: transitive
    # constraints, both ends of a displacement, no cascade from a missing step.
    assert "The constraints chain" in text
    assert "both of them are violations" in text
    assert "A step that was not performed at all makes no other step a violation" in text


def test_the_question_asks_for_the_answer_on_the_final_answer_line() -> None:
    """The question must not contradict the ReAct format of ``task_prompt``.

    A wording that asks for the bare JSON object can make a model emit it with
    no ``Final Answer:`` prefix, which the parser reads as a parse error: the
    episode then restarts its tool loop and burns every iteration.
    """
    text = question()

    assert "Final Answer:" in text
    assert "no text before" not in text
    assert "nothing else, with" not in text


def test_no_placeholder_survives_rendering() -> None:
    text = question()

    for placeholder in ("{recipe_name}", "{n_segments}", "{steps}", "{prerequisites}", "{schema}"):
        assert placeholder not in text


def test_the_task_prompt_only_changes_the_final_answer_line() -> None:
    reference = load_prompt("prompt").splitlines()
    task = load_prompt("task_prompt").splitlines()

    differing = [
        (before, after)
        for before, after in zip(reference, task, strict=True)
        if before != after
    ]

    assert differing == [
        (
            "Final Answer: the answer to the original input question",
            "Final Answer: the JSON object requested in the question, and nothing else",
        )
    ]


class TestParseFinalAnswer:
    def test_reads_a_bare_json_object(self) -> None:
        assert parse_final_answer('{"steps": [{"step_id": 3}]}') == {
            "steps": [{"step_id": 3}]
        }

    def test_reads_a_fenced_json_object(self) -> None:
        text = '```json\n{"steps": [{"step_id": 3}]}\n```'

        assert parse_final_answer(text) == {"steps": [{"step_id": 3}]}

    def test_reads_a_fenced_object_without_a_language_tag(self) -> None:
        assert parse_final_answer('```\n{"steps": [{"step_id": 3}]}\n```') == {
            "steps": [{"step_id": 3}]
        }

    def test_reads_an_object_buried_in_prose(self) -> None:
        text = 'Here is my answer: {"steps": [{"step_id": 1}]} — I hope it helps.'

        assert parse_final_answer(text) == {"steps": [{"step_id": 1}]}

    def test_rejects_broken_json(self) -> None:
        assert parse_final_answer('{"steps": [{"step_id": 3,]}') is None

    def test_rejects_an_answer_with_no_object_at_all(self) -> None:
        assert parse_final_answer("the cook forgot to slice the bread") is None
        assert parse_final_answer("") is None

    def test_rejects_json_that_is_not_the_expected_shape(self) -> None:
        # A list, or an object without the "steps" list, is not an answer.
        assert parse_final_answer('[{"step_id": 3}]') is None
        assert parse_final_answer('{"verdicts": []}') is None
        assert parse_final_answer('{"steps": {"3": "yes"}}') is None

    def test_reads_the_answer_a_stray_think_tag_splits_in_two(self) -> None:
        """An answer duplicated around a stray closing think tag.

        The model wrote the answer, closed a reasoning block that had never
        been opened, and wrote the same answer again.
        """
        answer = '{"steps": [{"step_id": 3, "executed": "yes"}]}'
        text = f"{answer}\n</think>\n\n{answer}"

        assert parse_final_answer(text) == {
            "steps": [{"step_id": 3, "executed": "yes"}]
        }

    def test_reads_an_answer_followed_by_second_thoughts(self) -> None:
        """Prose after a complete answer."""
        text = (
            '{"steps": [{"step_id": 3, "executed": "no"}]}\n\n'
            "Wait, I need to make sure the step was really not performed. "
            "The captions around segment 4 do mention the ramekin."
        )

        assert parse_final_answer(text) == {
            "steps": [{"step_id": 3, "executed": "no"}]
        }

    def test_an_object_of_the_wrong_shape_does_not_hide_the_answer(self) -> None:
        text = '{"verdicts": []} then, correctly: {"steps": [{"step_id": 1}]}'

        assert parse_final_answer(text) == {"steps": [{"step_id": 1}]}

    def test_the_last_of_two_answers_wins_because_the_model_revised(self) -> None:
        text = '{"steps": [{"step_id": 1}]} no wait: {"steps": [{"step_id": 2}]}'

        assert parse_final_answer(text) == {"steps": [{"step_id": 2}]}

    def test_an_empty_steps_list_is_not_an_answer(self) -> None:
        """An empty answer is not a successful run.

        ``{"steps": []}`` parses, so accepting it would give a record ``ok``
        with every step unclassified, a silent failure that reads in the
        tables like a deliberate abstention.
        """
        assert parse_final_answer('{"steps": []}') is None
        assert parse_final_answer('```json\n{"steps": []}\n```') is None

    def test_an_empty_answer_does_not_hide_the_real_one_before_it(self) -> None:
        text = '{"steps": [{"step_id": 1}]} ... on reflection: {"steps": []}'

        assert parse_final_answer(text) == {"steps": [{"step_id": 1}]}

    def test_a_real_answer_after_an_empty_one_still_wins(self) -> None:
        text = '{"steps": []} wait, here it is: {"steps": [{"step_id": 2}]}'

        assert parse_final_answer(text) == {"steps": [{"step_id": 2}]}

    def test_a_long_answer_is_scanned_once_per_top_level_brace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 10k-character answer must not cost one decode per brace.

        The span of an object that decoded is skipped, so the scan attempts
        ``raw_decode`` twice here — once for the answer, once for the stray
        brace in the prose after it — and not once per step.
        """
        steps = [
            {"step_id": i, "executed": "yes", "segments": [i, i + 1]}
            for i in range(200)
        ]
        answer = json.dumps({"steps": steps})
        assert len(answer) > 10_000 and answer.count("{") == 201
        text = f"{answer}\n\nWait \u2014 {{ this brace is not JSON."
        attempts: list[int] = []

        class CountingDecoder(json.JSONDecoder):
            """The stdlib decoder, keeping the offsets it was asked about."""

            def raw_decode(self, s, idx=0):
                attempts.append(idx)
                return super().raw_decode(s, idx)

        monkeypatch.setattr(json, "JSONDecoder", CountingDecoder)
        parsed = parse_final_answer(text)

        assert parsed is not None and len(parsed["steps"]) == 200
        assert attempts == [0, text.index("{", len(answer))]


@pytest.mark.parametrize("activity_id", [2, 20])
def test_a_recipe_that_repeats_a_step_asks_for_a_consistent_order(
    activity_id: int,
) -> None:
    """Dressed Up Meatballs and Sauted Mushrooms each cook a step twice.

    Merging those executions must not reach the model as two steps that
    require each other, which is a question no answer could satisfy.
    """
    annotations = load_annotations()
    sop = annotations.sops[activity_id]
    text = build_question(
        sop, annotations.graphs[activity_id], 100, load_prompt("task_question")
    )
    position = {step_id: index for index, (step_id, _) in enumerate(sop.steps)}

    required: dict[int, list[int]] = {
        int(step): [int(part) for part in listed.split(", ")]
        for step, listed in REQUIREMENT.findall(text)
    }

    assert required  # both recipes do constrain something
    for step_id, prerequisites in required.items():
        for prerequisite in prerequisites:
            assert position[prerequisite] < position[step_id]
            assert step_id not in required.get(prerequisite, [])


class TestFinalAnswerCandidates:
    """Every object that claimed to be an answer, counted for the record."""

    def test_lists_the_candidates_in_the_order_they_were_written(self) -> None:
        text = '{"steps": [{"step_id": 1}]} no wait: {"steps": [{"step_id": 2}]}'

        assert final_answer_candidates(text) == [
            {"steps": [{"step_id": 1}]},
            {"steps": [{"step_id": 2}]},
        ]

    def test_an_empty_answer_is_a_candidate_even_though_it_is_refused(self) -> None:
        text = '{"steps": []} and then {"steps": [{"step_id": 2}]}'

        assert len(final_answer_candidates(text)) == 2
        assert parse_final_answer(text) == {"steps": [{"step_id": 2}]}

    def test_objects_of_the_wrong_shape_are_not_candidates(self) -> None:
        assert final_answer_candidates('{"verdicts": []}') == []
        assert final_answer_candidates('{"steps": {"3": "yes"}}') == []
        assert final_answer_candidates("") == []
        assert final_answer_candidates("the cook forgot to slice the bread") == []

    def test_the_two_known_shapes_count_what_the_model_wrote(self) -> None:
        answer = '{"steps": [{"step_id": 3, "executed": "yes"}]}'

        assert len(final_answer_candidates(f"{answer}\n</think>\n\n{answer}")) == 2
        assert len(final_answer_candidates(f"{answer}\n\nWait, I need to check.")) == 1

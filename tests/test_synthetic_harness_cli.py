"""The synthetic harness as a script: selection, the sweep, and what it writes.

Offline end to end. The chat model is ``FakeChat`` replaying, per case, one
``caption_retrieval`` call — so the synthetic memory is really queried — and
then the Final Answer a perfect system would give, built from the case's own
``expected``. What is under test is the harness, not the agent: that every
case gets a record, that the scores say ``passed``, that η is the ratio of
shuffled over ordered correct verdicts, and that the table names one row per
recording and one column per case.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from tests.fakes import FakeChat

from video_agent.agent.config import ReActConfig
from video_agent.agent.prompts import load_prompt
from video_agent.agent.react import ReActRunner
from video_agent.eval.synthetic import (
    DEFAULT_SEED,
    Case,
    HashingTextEmbedder,
    make_cases,
    synthetic_video,
)
from video_agent.memory.preprocess import get_video_segments
from video_agent.memory.store import segment_index_at
from video_agent.task.annotations import AnnotationSet, Sop, load_split
from video_agent.task.run import RunStore
from video_agent.task.verdicts import Execution, Order, RecordingVerdicts

from tests._paths import DEV_SPLIT, SCRIPTS, load_annotations

BASE_ID = "1_25"
"""A normal dev recording that can host every case."""

ERROR_ID = "23_32"
"""A dev error recording, which cannot: its annotation already tags errors."""

CASE_NAMES = ("normal", "omission", "shuffled")
"""Cases a, b and g — the three the end-to-end sweep runs."""


@pytest.fixture(scope="module")
def harness_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "synthetic_harness_cli", SCRIPTS / "synthetic_harness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def annotations() -> AnnotationSet:
    return load_annotations()


class TestBackendFlags:
    def test_the_local_server_is_the_default(self, harness_module) -> None:
        from _wiring import resolve_base_url

        args = harness_module.parse_args(["--recordings", BASE_ID, "--out", "runs"])

        assert args.chat == "vllm" and args.model == "Qwen/Qwen3.5-9B"
        assert args.max_iterations == 100
        assert args.base_url is None  # filled in by the backend, see _wiring
        assert resolve_base_url(args) == "http://localhost:8000/v1"

    def test_the_aligner_of_the_record_is_the_projects_own(
        self, harness_module
    ) -> None:
        """The harness has no pixels, but its record names the agent's aligner.

        The hashing stand-in plays both towers here; what the flag decides is
        the configuration the harness vets, which is the one a real run has.
        """
        from video_agent.agent.config import DEFAULT_ALIGNER_MODEL

        args = harness_module.parse_args(["--recordings", BASE_ID, "--out", "runs"])
        chosen = harness_module.parse_args(
            [
                "--recordings", BASE_ID,
                "--out", "runs",
                "--aligner-model", "microsoft/xclip-base-patch16",
            ]
        )

        assert args.aligner_model == DEFAULT_ALIGNER_MODEL
        config = harness_module.task_config(args, "PROMPT", "QUESTION")
        assert config.tools.aligner_model == DEFAULT_ALIGNER_MODEL
        assert harness_module.task_config(
            chosen, "PROMPT", "QUESTION"
        ).tools.aligner_model == "microsoft/xclip-base-patch16"

    def test_an_unknown_aligner_is_refused_before_the_sweep(
        self, harness_module
    ) -> None:
        with pytest.raises(SystemExit):
            harness_module.parse_args(
                ["--recordings", BASE_ID, "--out", "runs",
                 "--aligner-model", "nobody/nothing"]
            )

    def test_the_local_backend_is_selectable_and_resolves_its_endpoint(
        self, harness_module
    ) -> None:
        from _wiring import resolve_base_url

        args = harness_module.parse_args(
            ["--recordings", BASE_ID, "--out", "runs", "--chat", "vllm"]
        )

        assert args.chat == "vllm"
        assert resolve_base_url(args) == "http://localhost:8000/v1"


    def test_the_turn_switches_reach_the_frozen_configuration(
        self, harness_module
    ) -> None:
        """The harness records what a real run records, flags included."""
        argv = ["--recordings", BASE_ID, "--out", "runs"]
        default = harness_module.parse_args(argv)
        args = harness_module.parse_args(argv + ["--max-tokens", "4096", "--thinking"])

        assert default.max_tokens == 16384
        assert not default.thinking and default.chat_timeout is None

        config = harness_module.task_config(args, "prompt", "question")

        assert config.thinking is True
        assert config.react.max_tokens == 4096
        assert config.to_dict()["thinking"] is True


class TestSelectedCases:
    def test_the_letters_are_translated_in_the_canonical_order(
        self, harness_module
    ) -> None:
        assert harness_module.selected_cases("g,a,b") == CASE_NAMES

    def test_whitespace_around_the_letters_is_tolerated(self, harness_module) -> None:
        assert harness_module.selected_cases(" a , b ,g ") == CASE_NAMES

    def test_every_case_is_the_default(self, harness_module) -> None:
        assert len(harness_module.selected_cases("a,b,c,d,e,f,g")) == 7

    def test_an_unknown_letter_is_a_contract_violation(self, harness_module) -> None:
        with pytest.raises(ValueError):
            harness_module.selected_cases("a,z")


class TestUnusableReason:
    def test_a_clean_normal_recording_can_host_the_cases(
        self, harness_module, annotations: AnnotationSet
    ) -> None:
        recording = annotations.recordings[BASE_ID]
        graph = annotations.graphs[recording.activity_id]

        assert harness_module.unusable_reason(recording, graph) is None

    def test_a_recording_that_already_carries_a_tag_is_refused_with_a_reason(
        self, harness_module, annotations: AnnotationSet
    ) -> None:
        recording = annotations.recordings[ERROR_ID]
        graph = annotations.graphs[recording.activity_id]

        reason = harness_module.unusable_reason(recording, graph)

        assert isinstance(reason, str) and reason


class TestSelectRecordings:
    def test_explicit_ids_are_honoured_as_written(
        self, harness_module, annotations: AnnotationSet
    ) -> None:
        chosen = harness_module.select_recordings(
            [" 7_5", ERROR_ID, BASE_ID], [], annotations, limit=2
        )

        assert chosen == ["7_5", ERROR_ID, BASE_ID]

    def test_split_ids_are_filtered_to_the_ones_that_can_host_the_cases(
        self, harness_module, annotations: AnnotationSet
    ) -> None:
        _, normal = load_split(DEV_SPLIT)

        chosen = harness_module.select_recordings([], normal, annotations)

        assert BASE_ID in chosen
        assert "5_2" not in chosen  # normal, but its order breaks the task graph
        assert chosen == sorted(set(chosen))
        assert all(
            harness_module.unusable_reason(
                annotations.recordings[recording_id],
                annotations.graphs[annotations.recordings[recording_id].activity_id],
            )
            is None
            for recording_id in chosen
        )

    def test_a_limit_truncates_the_split_selection(
        self, harness_module, annotations: AnnotationSet
    ) -> None:
        _, normal = load_split(DEV_SPLIT)

        chosen = harness_module.select_recordings([], normal, annotations, limit=3)

        assert chosen == harness_module.select_recordings([], normal, annotations)[:3]


EXECUTION_WORD = {
    Execution.EXECUTED: "yes",
    Execution.NOT_EXECUTED: "no",
    Execution.INDETERMINATE: "unsure",
}
ORDER_WORD = {
    Order.OK: "ok",
    Order.VIOLATION: "violation",
    Order.INDETERMINATE: "unsure",
    Order.NOT_APPLICABLE: None,
}


def answer_json(verdicts: RecordingVerdicts) -> str:
    """Write verdicts back in the schema the question asks the agent for."""
    steps = []
    for verdict in verdicts.steps:
        prerequisite = None
        if verdict.prerequisite_step_id is not None:
            prerequisite = {
                "step_id": verdict.prerequisite_step_id,
                "segments": (
                    None
                    if verdict.prerequisite_anchor is None
                    else list(verdict.prerequisite_anchor.segments)
                ),
            }
        steps.append(
            {
                "step_id": verdict.step_id,
                "executed": EXECUTION_WORD[verdict.execution],
                "segments": (
                    None if verdict.anchor is None else list(verdict.anchor.segments)
                ),
                "order": ORDER_WORD[verdict.order],
                "violated_prerequisite": prerequisite,
            }
        )
    return json.dumps({"steps": steps})


def caption_target(case: Case, sop: Sop) -> tuple[int, str]:
    """A segment of this case's memory and the caption it must carry.

    The midpoint of the first step the case's timeline performs: the
    synthetic memory captions a segment with the step covering its middle, so
    this is the one lookup whose answer is known in advance.
    """
    segments = get_video_segments(synthetic_video(case.recording))
    timed = [step for step in case.recording.steps if step.start_s is not None]
    step = timed[0]
    index = segment_index_at(segments, (step.start_s + step.end_s) / 2.0)
    return index, dict(sop.steps)[step.step_id]


def script_for(case: Case, sop: Sop, expected: RecordingVerdicts) -> list[str]:
    """One caption lookup and one Final Answer, as the ReAct loop reads them."""
    index, _ = caption_target(case, sop)
    return [
        "Thought: I should read the captions around this step\n"
        "Action: caption_retrieval\n"
        f"Action Input: ({index}, {index})",
        "Thought: I now know the final answer\n"
        "Final Answer: ```json\n" + answer_json(expected) + "\n```",
    ]


@pytest.fixture(scope="module")
def sweep(harness_module, annotations: AnnotationSet, tmp_path_factory) -> dict:
    """Run cases a, b and g over one recording with a perfectly scripted agent."""
    recording = annotations.recordings[BASE_ID]
    sop, graph = annotations.for_recording(BASE_ID)
    cases = {
        case.name: case
        for case in make_cases(recording, sop, graph, seed=DEFAULT_SEED)
    }
    replies: list[str] = []
    for name in CASE_NAMES:
        # The shuffled case has no right answer: the agent replays the
        # ordered case's verdicts, which is what η measures against.
        expected = cases["normal" if name == "shuffled" else name].expected
        replies.extend(script_for(cases[name], sop, expected))

    chat = FakeChat(replies)
    store = RunStore(tmp_path_factory.mktemp("harness"), "synthetic")
    args = harness_module.parse_args(
        ["--recordings", BASE_ID, "--out", str(store.root), "--cases", "a,b,g"]
    )
    prompt_template = load_prompt("task_prompt")
    question_template = load_prompt("task_question")
    lines: list[str] = []
    scores = harness_module.run_harness(
        recording_ids=[BASE_ID],
        case_names=CASE_NAMES,
        annotations=annotations,
        runner=ReActRunner(chat, ReActConfig()),
        embedder=HashingTextEmbedder(64),
        store=store,
        prompt_template=prompt_template,
        question_template=question_template,
        config=harness_module.task_config(args, prompt_template, question_template),
        seed=DEFAULT_SEED,
        log=lines.append,
    )
    return {"scores": scores, "store": store, "cases": cases, "sop": sop, "log": lines}


def case_score(sweep: dict, name: str) -> dict:
    return next(
        score
        for score in sweep["scores"]["recordings"][0]["cases"]
        if score["case"] == name
    )


class TestRunHarness:
    def test_every_case_leaves_a_record_of_its_own(self, sweep: dict) -> None:
        assert sweep["store"].recording_ids() == sorted(
            f"{BASE_ID}__{name}" for name in CASE_NAMES
        )

    def test_the_agent_really_queried_the_synthetic_memory(
        self, sweep: dict
    ) -> None:
        record = sweep["store"].load(f"{BASE_ID}__normal")
        action = record["transcript"]["steps"][0]["action"]
        _, caption = caption_target(sweep["cases"]["normal"], sweep["sop"])

        assert action["name"] == "caption_retrieval"
        assert caption in action["observation"]

    def test_the_perfect_answer_passes_the_scored_cases(self, sweep: dict) -> None:
        for name in ("normal", "omission"):
            score = case_score(sweep, name)
            assert score["status"] == "ok"
            assert score["passed"] is True
            assert score["matched"] == score["total"]
            assert score["mismatches"] == []

    def test_the_shuffled_case_is_reported_but_never_scored(
        self, sweep: dict
    ) -> None:
        score = case_score(sweep, "shuffled")

        assert score["scored"] is False
        assert score["matched"] is None and score["passed"] is None

    def test_eta_is_the_shuffled_agreement_over_the_ordered_one(
        self, sweep: dict
    ) -> None:
        value = sweep["scores"]["recordings"][0]["eta"]
        shuffled = case_score(sweep, "shuffled")["matched_vs_normal"]
        normal = case_score(sweep, "normal")["matched"]

        assert value is not None
        assert value == shuffled / normal
        # The agent replayed the ordered verdicts, so nothing was lost.
        assert value == 1.0

    def test_the_scores_carry_the_run_and_the_seed_that_made_the_cases(
        self, sweep: dict
    ) -> None:
        scores = sweep["scores"]

        assert scores["run_name"] == "synthetic"
        assert scores["seed"] == DEFAULT_SEED
        assert scores["cases"] == list(CASE_NAMES)
        assert scores["recordings"][0]["recording_id"] == BASE_ID
        assert json.loads(json.dumps(scores, ensure_ascii=False)) == scores

    def test_one_line_is_logged_per_case(self, sweep: dict) -> None:
        assert len(sweep["log"]) == len(CASE_NAMES)
        assert all("FAILED" not in line for line in sweep["log"])


class TestWriteOutputs:
    def test_the_table_has_one_row_per_recording_and_one_column_per_case(
        self, harness_module, sweep: dict, tmp_path: Path
    ) -> None:
        scores_path, report_path = harness_module.write_outputs(
            tmp_path / "synthetic", sweep["scores"], CASE_NAMES
        )

        assert scores_path.name == "harness_scores.json"
        assert json.loads(scores_path.read_text(encoding="utf-8")) == sweep["scores"]

        lines = report_path.read_text(encoding="utf-8").splitlines()
        header = next(line for line in lines if line.startswith("| Recording |"))
        columns = [cell.strip() for cell in header.strip("|").split("|")]
        rows = [line for line in lines if line.startswith(f"| {BASE_ID} |")]

        assert columns == ["Recording", *CASE_NAMES, "η"]
        assert len(rows) == 1
        assert len(rows[0].strip("|").split("|")) == len(columns)

    def test_a_passed_cell_is_marked_ok(
        self, harness_module, sweep: dict, tmp_path: Path
    ) -> None:
        _, report_path = harness_module.write_outputs(
            tmp_path / "synthetic", sweep["scores"], CASE_NAMES
        )

        text = report_path.read_text(encoding="utf-8")

        assert " ok" in text
        assert "| 1.00 |" in text  # η


class TestSummarize:
    def test_it_counts_the_cells_the_sweep_produced(
        self, harness_module, sweep: dict
    ) -> None:
        assert harness_module.summarize(sweep["scores"], CASE_NAMES) == (2, 2, 0)

    def test_a_case_that_left_no_score_is_counted_as_a_failure(
        self, harness_module, sweep: dict
    ) -> None:
        broken = {
            **sweep["scores"],
            "recordings": [
                {**sweep["scores"]["recordings"][0], "cases": []},
            ],
        }

        assert harness_module.summarize(broken, CASE_NAMES) == (0, 0, 3)

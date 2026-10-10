"""The EgoSchema report: the accuracy rule, the process measures, the pairing.

Everything is checked against numbers computed by hand over three fabricated
runs, so a change in the arithmetic is a failing test and not a moved
decimal. ``scripts/`` is not a package, so the module is loaded from its
path the way ``test_evaluate_cli`` does.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from video_agent.task.run import RunStore

from tests._paths import SCRIPTS

ANSWERS = {"q1": 0, "q2": 1, "q3": 2, "q4": 3, "q5": 4}
"""The truth of the five fabricated questions."""


@pytest.fixture(scope="module")
def module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "evaluate_egoschema_cli", SCRIPTS / "evaluate_egoschema.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def transcript(n_steps: int, tools: list[object]) -> dict:
    """A transcript of ``n_steps`` turns, ``tools`` naming the ones that acted.

    An entry is the action's name, ``None`` for a turn that took none, or a
    ``(name, error)`` pair for one the executor wrote a verdict on.
    """
    steps = []
    for index in range(n_steps):
        entry = tools[index] if index < len(tools) else None
        name, error = entry if isinstance(entry, tuple) else (entry, None)
        steps.append(
            {
                "turn": {
                    "index": index,
                    "completion": "Thought: ...",
                    "usage": {},
                    "latency_s": 0.1,
                    "model": "fake",
                    "finish_reason": "stop",
                },
                "action": (
                    None if name is None
                    else {"name": name, "error": error, "sub_transcript": None}
                ),
                "final_answer": None,
                "parse_error": None,
            }
        )
    return {
        "prompt": "P",
        "steps": steps,
        "final_answer": None,
        "status": "ok",
        "usage": {},
    }


def record(
    q_uid: str,
    prediction: int | None,
    status: str,
    *,
    seconds: float = 1.0,
    n_steps: int = 1,
    tools: list[object] | None = None,
    tokens_in: int = 100,
    tokens_out: int = 20,
) -> dict:
    return {
        "q_uid": q_uid,
        "prediction": prediction,
        "status": status,
        "truth": ANSWERS[q_uid],
        "answer_raw": str(prediction),
        "seconds": seconds,
        "config": {"config_id": "cfg0", "model": "Qwen/Qwen3.5-9B"},
        "usage_total": {"tokens_in": tokens_in, "tokens_out": tokens_out},
        "transcript": transcript(n_steps, tools or []),
    }


RUN_A = [
    # q1 right, q2 wrong, q3/q4 invalid, q5 right -> 2/5 with invalid as wrong
    record(
        "q1", 0, "ok", seconds=1.0, n_steps=3,
        tools=["caption_retrieval", "caption_retrieval"],
    ),
    record(
        "q2", 3, "ok", seconds=2.0, n_steps=5,
        tools=["caption_retrieval", "visual_question_answering"],
    ),
    record("q3", None, "parse_failed", seconds=3.0, n_steps=1),
    record(
        "q4", None, "max_iterations", seconds=4.0, n_steps=15,
        tools=["caption_retrieval"] * 10,
    ),
    record("q5", 4, "ok", seconds=5.0, n_steps=2),
]

RUN_B = [
    record("q1", 0, "ok"),
    record("q2", 1, "ok"),
    record("q3", 2, "ok"),
    record("q4", 3, "ok"),
    record("q5", 0, "ok"),
]


MALFORMED = (
    'caption_retrieval(input_tuple=("0", "14"))\n'
    'Thought: the error "Missing \'Action Input:\'" suggests the parser wants\n'
    "the tuple on its own line | or maybe without the keyword."
)
"""An ``Action:`` line that names no tool and spans lines with a pipe in it.

The report must count it as an invalid action rather than as a tool of its
own, and its newlines and pipe must not break the Markdown table.
"""


def table_blocks(table: str) -> list[list[str]]:
    """The contiguous blocks of Markdown table rows, each block as its lines."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in table.split("\n"):
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def n_columns(row: str) -> int:
    """Cells of one Markdown row: an escaped pipe is content, not a border."""
    return len(re.split(r"(?<!\\)\|", row)) - 2


def build(tmp_path: Path, name: str, records: list[dict]) -> Path:
    store = RunStore(tmp_path, name)
    for item in records:
        store.save(item["q_uid"], item)
    return store.directory


@pytest.fixture
def run_a(tmp_path: Path) -> Path:
    return build(tmp_path, "ego-9b", RUN_A)


@pytest.fixture
def run_b(tmp_path: Path) -> Path:
    return build(tmp_path, "ego-4o", RUN_B)


class TestWilson:
    def test_a_known_interval(self, module) -> None:
        # 10 of 20 at 95%: the textbook Wilson interval.
        interval = module.wilson_interval(10, 20)

        assert interval["point"] == 0.5
        assert interval["low"] == pytest.approx(0.2993, abs=1e-4)
        assert interval["high"] == pytest.approx(0.7007, abs=1e-4)

    def test_it_stays_inside_the_unit_interval_at_the_ends(self, module) -> None:
        """Where the normal approximation would leave it: 0 and 1."""
        low_end = module.wilson_interval(0, 10)
        high_end = module.wilson_interval(10, 10)

        assert low_end["point"] == 0.0 and low_end["low"] == 0.0
        assert 0.0 < low_end["high"] < 1.0
        assert high_end["point"] == 1.0
        assert high_end["high"] == pytest.approx(1.0)
        assert 0.0 < high_end["low"] < 1.0

    def test_nothing_measured_has_no_proportion(self, module) -> None:
        assert module.wilson_interval(0, 0)["point"] is None

    def test_more_successes_than_trials_is_a_contract_violation(self, module) -> None:
        with pytest.raises(ValueError):
            module.wilson_interval(3, 2)


class TestAccuracy:
    def test_invalid_answers_count_as_wrong_and_are_reported_apart(
        self, module, run_a: Path
    ) -> None:
        evaluation = module.evaluate_run(run_a, ANSWERS)

        assert evaluation["accuracy"]["point"] == pytest.approx(2 / 5)
        assert evaluation["accuracy"]["n"] == 5
        assert evaluation["accuracy_valid_only"]["point"] == pytest.approx(2 / 3)
        assert evaluation["accuracy_valid_only"]["n"] == 3
        assert evaluation["n_valid"] == 3 and evaluation["n_invalid"] == 2

    def test_the_statuses_are_counted_and_the_invalid_ones_attributed(
        self, module, run_a: Path
    ) -> None:
        evaluation = module.evaluate_run(run_a, ANSWERS)

        assert evaluation["status_counts"] == {
            "max_iterations": 1,
            "ok": 3,
            "parse_failed": 1,
        }
        assert evaluation["invalid_by_status"] == {
            "max_iterations": 1,
            "parse_failed": 1,
        }

    def test_a_question_with_no_record_counts_as_wrong_and_is_named(
        self, module, run_a: Path
    ) -> None:
        """An unfinished batch must not flatter itself by a shorter denominator."""
        answers = {**ANSWERS, "q6": 1}

        evaluation = module.evaluate_run(run_a, answers)

        assert evaluation["n_answers"] == 6 and evaluation["n_records"] == 5
        assert evaluation["n_without_record"] == 1
        assert evaluation["status_counts"]["no_record"] == 1
        assert evaluation["accuracy"]["point"] == pytest.approx(2 / 6)

    def test_records_outside_the_answered_subset_are_reported_not_scored(
        self, module, tmp_path: Path
    ) -> None:
        run = build(tmp_path, "ego", RUN_A)
        stray = {**record("q1", 0, "ok"), "q_uid": "q9"}
        RunStore(tmp_path, "ego").save("q9", stray)

        evaluation = module.evaluate_run(run, ANSWERS)

        assert evaluation["n_records_outside_subset"] == 1
        assert evaluation["accuracy"]["n"] == 5

    def test_an_empty_run_directory_scores_everything_wrong(
        self, module, tmp_path: Path
    ) -> None:
        evaluation = module.evaluate_run(tmp_path / "nothing", ANSWERS)

        assert evaluation["accuracy"]["point"] == 0.0
        assert evaluation["n_without_record"] == 5


class TestProcess:
    def test_the_iteration_distribution_is_read_off_the_transcripts(
        self, module, run_a: Path
    ) -> None:
        # 1, 2, 3, 5, 15 steps: median 3, p90 interpolated at 11, max 15.
        iterations = module.evaluate_run(run_a, ANSWERS)["process"]["iterations"]

        assert iterations["median"] == 3.0 and iterations["max"] == 15.0
        assert iterations["p90"] == pytest.approx(11.0)

    def test_each_tool_is_counted_per_question_and_by_reach(
        self, module, run_a: Path
    ) -> None:
        tools = module.evaluate_run(run_a, ANSWERS)["process"]["tools"]

        # 2 + 1 + 10 = 13 calls over 5 questions, in 3 of them.
        assert tools["caption_retrieval"]["calls"] == 13
        assert tools["caption_retrieval"]["mean_per_question"] == pytest.approx(2.6)
        assert tools["caption_retrieval"]["questions_pct"] == pytest.approx(60.0)
        assert tools["visual_question_answering"]["calls"] == 1
        assert tools["visual_question_answering"]["questions_pct"] == pytest.approx(20.0)

    def test_tokens_and_seconds_are_summed_and_ranked(
        self, module, run_a: Path
    ) -> None:
        process = module.evaluate_run(run_a, ANSWERS)["process"]

        assert process["tokens_in"] == {"mean": 100.0, "total": 500.0}
        assert process["tokens_out"] == {"mean": 20.0, "total": 100.0}
        assert process["seconds"] == {"median": 3.0, "total": 15.0}

    def test_the_table_has_one_row_per_registered_tool_even_unused_ones(
        self, module, run_a: Path
    ) -> None:
        """A closed vocabulary: the rows are the toolkit, not what was called."""
        process = module.evaluate_run(run_a, ANSWERS)["process"]

        assert list(process["tools"]) == [
            "caption_retrieval",
            "segment_localization",
            "visual_question_answering",
            "object_memory_querying",
            "invalid_action",
        ]
        assert process["tools"]["segment_localization"]["calls"] == 0
        assert process["tools"]["invalid_action"]["calls"] == 0
        assert process["n_questions_with_invalid_action"] == 0

    def test_an_action_naming_no_tool_is_counted_as_invalid_not_as_a_tool(
        self, module, tmp_path: Path
    ) -> None:
        run = build(
            tmp_path,
            "ego",
            [
                record(
                    "q1", 0, "ok", n_steps=3,
                    tools=[MALFORMED, "caption_retrieval", "segment_localization("],
                ),
                record("q2", 1, "ok", n_steps=1, tools=["caption_retrieval"]),
            ],
        )

        process = module.evaluate_run(run, {"q1": 0, "q2": 1})["process"]

        assert process["tools"]["invalid_action"]["calls"] == 2
        assert process["tools"]["invalid_action"]["questions"] == 1
        assert process["tools"]["invalid_action"]["questions_pct"] == 50.0
        assert process["tools"]["invalid_action"]["mean_per_question"] == 1.0
        assert process["tools"]["caption_retrieval"]["calls"] == 2
        assert process["n_questions_with_invalid_action"] == 1
        # No row grew out of the raw text the model wrote.
        assert MALFORMED not in process["tools"]

    def test_the_executors_own_verdict_makes_a_step_invalid(
        self, module, tmp_path: Path
    ) -> None:
        """Even under a registered name: ``error`` says no tool was found."""
        run = build(
            tmp_path,
            "ego",
            [
                record(
                    "q1", 0, "ok", n_steps=2,
                    tools=[("caption_retrieval", "unknown tool"), "caption_retrieval"],
                )
            ],
        )

        process = module.evaluate_run(run, {"q1": 0})["process"]

        assert process["tools"]["invalid_action"]["calls"] == 1
        assert process["tools"]["caption_retrieval"]["calls"] == 1

    def test_a_tool_that_was_found_and_failed_is_still_a_call_of_that_tool(
        self, module, tmp_path: Path
    ) -> None:
        run = build(
            tmp_path,
            "ego",
            [
                record(
                    "q1", 0, "ok", n_steps=1,
                    tools=[("caption_retrieval", "invalid arguments")],
                )
            ],
        )

        process = module.evaluate_run(run, {"q1": 0})["process"]

        assert process["tools"]["caption_retrieval"]["calls"] == 1
        assert process["tools"]["invalid_action"]["calls"] == 0

    def test_a_turn_the_token_cap_cut_is_counted(
        self, module, tmp_path: Path
    ) -> None:
        cut = record("q1", 0, "ok", n_steps=1)
        cut["transcript"]["steps"][0]["turn"]["finish_reason"] = "length"
        run = build(tmp_path, "ego", [cut])

        evaluation = module.evaluate_run(run, {"q1": 0})

        assert evaluation["process"]["length_turns"] == 1


class TestComparison:
    def test_the_paired_numbers_are_the_ones_computed_by_hand(
        self, module, run_a: Path, run_b: Path
    ) -> None:
        comparison = module.compare_runs(run_a, run_b, ANSWERS)

        assert comparison["n_common"] == 5
        assert comparison["accuracy_a"]["point"] == pytest.approx(2 / 5)
        assert comparison["accuracy_b"]["point"] == pytest.approx(4 / 5)
        assert comparison["difference"]["point"] == pytest.approx(-0.4)
        assert comparison["difference"]["low"] is not None
        assert comparison["difference"]["n_resamples"] == 2000

    def test_mcnemar_reads_only_the_questions_they_disagreed_on(
        self, module, run_a: Path, run_b: Path
    ) -> None:
        # b = 1 (only q5 is A's alone), c = 3 (q2, q3, q4 are B's alone);
        # the exact two-sided binomial tail of 1 of 4 is 2 * 5/16 = 0.625.
        test = module.compare_runs(run_a, run_b, ANSWERS)["mcnemar"]

        assert (test["b"], test["c"]) == (1, 3)
        assert test["p"] == pytest.approx(0.625)

    def test_agreement_is_about_the_alternative_chosen_not_about_being_right(
        self, module, run_a: Path, run_b: Path
    ) -> None:
        # Only q1 got the same label out of both runs.
        comparison = module.compare_runs(run_a, run_b, ANSWERS)

        assert comparison["n_agree"] == 1
        assert comparison["agreement"] == pytest.approx(0.2)

    def test_only_the_questions_both_runs_answered_are_compared(
        self, module, run_a: Path, tmp_path: Path
    ) -> None:
        short = build(tmp_path, "short", RUN_B[:2])

        comparison = module.compare_runs(run_a, short, ANSWERS)

        assert comparison["n_common"] == 2

    def test_the_bootstrap_is_reproducible_from_its_seed(
        self, module, run_a: Path, run_b: Path
    ) -> None:
        first = module.compare_runs(run_a, run_b, ANSWERS, n_bootstrap=200, seed=0)
        again = module.compare_runs(run_a, run_b, ANSWERS, n_bootstrap=200, seed=0)

        assert first["difference"] == again["difference"]
        assert first["difference"]["n_resamples"] == 200
        assert (
            first["difference"]["low"]
            <= first["difference"]["point"]
            <= first["difference"]["high"]
        )


class TestCli:
    def test_a_report_needs_a_run_answers_and_an_output_directory(
        self, module
    ) -> None:
        with pytest.raises(SystemExit):
            module.parse_args(["--run-dir", "runs/ego"])

    def test_the_defaults_are_the_documented_bootstrap(self, module) -> None:
        args = module.parse_args(
            ["--run-dir", "runs/ego", "--answers", "a.json", "--out", "out"]
        )

        assert args.bootstrap == 2000 and args.seed == 0
        assert args.compare is None

    def test_it_writes_the_two_files_and_the_published_rows(
        self, module, run_a: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        answers_path = tmp_path / "subset_answers.json"
        answers_path.write_text(json.dumps(ANSWERS), encoding="utf-8")
        out = tmp_path / "report"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_egoschema.py",
                "--run-dir", str(run_a),
                "--answers", str(answers_path),
                "--out", str(out),
            ],
        )

        module.main()

        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        table = (out / "table.md").read_text(encoding="utf-8")
        assert metrics["accuracy"]["point"] == pytest.approx(0.4)
        assert metrics["orchestrator"] == "Qwen/Qwen3.5-9B"
        # The whole point of the table: the run beside what was published.
        for published in ("62.8", "51.8", "36.8", "20.0"):
            assert published in table
        assert "ego-9b" in table
        assert "40.0" in table  # the run's own accuracy

    def test_compare_adds_the_paired_section(
        self, module, run_a: Path, run_b: Path, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        answers_path = tmp_path / "subset_answers.json"
        answers_path.write_text(json.dumps(ANSWERS), encoding="utf-8")
        out = tmp_path / "report"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_egoschema.py",
                "--run-dir", str(run_a),
                "--answers", str(answers_path),
                "--compare", str(run_b),
                "--out", str(out),
                "--bootstrap", "200",
            ],
        )

        module.main()

        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        table = (out / "table.md").read_text(encoding="utf-8")
        assert metrics["comparison"]["mcnemar"] == {"b": 1, "c": 3, "p": 0.625}
        assert "McNemar" in table and "ego-4o" in table


class TestTableShape:
    """A table a reader can parse: one line per row, one row per tool."""

    def test_a_malformed_action_name_never_reaches_the_table(
        self, module, tmp_path: Path
    ) -> None:
        run = build(
            tmp_path,
            "ego",
            [record("q1", 0, "ok", n_steps=2, tools=[MALFORMED, "caption_retrieval"])],
        )

        table = module.render_table(module.evaluate_run(run, {"q1": 0}))

        assert "input_tuple" not in table
        tool_rows = table_blocks(table)[-1]
        # The header, its rule, the four tools and the invalid-action row.
        assert len(tool_rows) == 7
        assert tool_rows[-1].startswith("| (invalid action) |")
        assert "| caption_retrieval | 1.00 | 100.0 |" in table
        assert "| (invalid action) | 1.00 | 100.0 |" in table

    def test_every_row_of_every_table_has_the_columns_of_its_header(
        self, module, tmp_path: Path
    ) -> None:
        """Free text in a cell must not take a row out of its table."""
        broken = record("q1", 0, "ok", n_steps=1, tools=[MALFORMED])
        broken["status"] = "read | error\nin record"
        broken["config"] = {"config_id": "cfg|0", "model": "Qwen | 9B"}
        run = build(tmp_path, "run | name", [broken])

        table = module.render_table(module.evaluate_run(run, {"q1": 0}))

        for block in table_blocks(table):
            columns = n_columns(block[0])
            assert all(n_columns(line) == columns for line in block), block
            assert len(block) >= 3

    def test_the_run_written_to_disk_carries_the_closed_vocabulary(
        self, module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run = build(
            tmp_path,
            "ego-9b",
            [record("q1", 0, "ok", n_steps=2, tools=[MALFORMED, "caption_retrieval"])],
        )
        answers_path = tmp_path / "subset_answers.json"
        answers_path.write_text(json.dumps({"q1": 0}), encoding="utf-8")
        out = tmp_path / "report"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_egoschema.py",
                "--run-dir", str(run),
                "--answers", str(answers_path),
                "--out", str(out),
            ],
        )

        module.main()

        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        process = metrics["process"]
        assert set(process["tools"]) == {
            "caption_retrieval",
            "segment_localization",
            "visual_question_answering",
            "object_memory_querying",
            "invalid_action",
        }
        assert process["n_questions_with_invalid_action"] == 1
        assert "input_tuple" not in (out / "table.md").read_text(encoding="utf-8")

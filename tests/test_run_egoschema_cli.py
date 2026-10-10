"""The EgoSchema batch CLI: what it asks, what it reads back, what it writes.

``scripts/`` is not a package, so the module is loaded from its path the way
``test_run_task_cli`` does; the directory goes on ``sys.path`` first because
the script imports its siblings ``_env``, ``_wiring`` and ``ask`` the way it
does when run directly.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from video_agent.agent.runner import AgentResult
from video_agent.agent.transcript import LLMTurn, ReActStep, Transcript
from video_agent.models.types import Usage
from video_agent.task.run import RunStore

from tests._paths import SCRIPTS

BASE_ARGV = [
    "--questions", "questions.json",
    "--answers", "subset_answers.json",
    "--videos-dir", "videos",
    "--artifacts-root", "preprocess",
    "--out", "runs",
    "--run-name", "ego-9b",
]


@pytest.fixture(scope="module")
def module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "run_egoschema_cli", SCRIPTS / "run_egoschema.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def parse(module, *extra: str):
    return module.parse_args(BASE_ARGV + list(extra))


class TestArguments:
    def test_defaults_are_the_reference_multiple_choice_setting(self, module) -> None:
        args = parse(module)

        assert args.questions == Path("questions.json")
        assert args.answers == Path("subset_answers.json")
        assert args.out == Path("runs") and args.run_name == "ego-9b"
        # The iteration ceiling of the original EgoSchema setting.
        assert args.max_iterations == 15
        assert args.temperature == 0.0
        assert args.caption_variant == "default"
        assert args.question_list is None and args.limit is None
        assert args.shard is None
        assert not args.force and not args.retry_errors

    def test_every_model_knob_of_ask_is_accepted(self, module) -> None:
        args = parse(
            module,
            "--chat", "vllm",
            "--model", "Qwen/Qwen3.5-9B",
            "--vqa", "api",
            "--vqa-style", "single",
            "--caption-variant", "ego",
            "--aligner-model", "timm/PE-Core-L-14-336",
            "--device", "cpu",
            "--max-iterations", "15",
            "--seed", "7",
            "--thinking",
        )

        assert (args.chat, args.model, args.vqa) == ("vllm", "Qwen/Qwen3.5-9B", "api")
        assert args.caption_variant == "ego" and args.device == "cpu"
        assert args.seed == 7 and args.thinking

    def test_an_unknown_aligner_is_refused_before_the_batch_starts(
        self, module
    ) -> None:
        with pytest.raises(SystemExit):
            parse(module, "--aligner-model", "nobody/nothing")

    def test_an_object_memory_slug_that_could_name_another_file_is_refused(
        self, module
    ) -> None:
        with pytest.raises(SystemExit):
            parse(module, "--object-memory", "../elsewhere")

    def test_a_run_needs_a_name_and_an_output_root(self, module) -> None:
        with pytest.raises(SystemExit):
            module.parse_args(["--questions", "q.json", "--answers", "a.json"])


class TestParseChoice:
    """Reading the label the way the reference did: the last ``Final Answer:``.

    An answer that is not one of the five labels is never guessed at — it is
    recorded invalid and counts as wrong.
    """

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("3", 3),
            ("Final Answer: 3", 3),
            ("Final Answer: choice 3", 3),
            ("Final Answer: (3)", 3),
            ("Final Answer: option 3.", 3),
            ("Thought: I now know\nFinal Answer: 0", 0),
            ("Final Answer: 1\nFinal Answer: 4", 4),  # the last one wins
            ("Final Answer: The correct choice is 2", 2),
            ("Final Answer:4", 4),
        ],
    )
    def test_a_label_is_read_however_it_is_dressed(
        self, module, text, expected
    ) -> None:
        assert module.parse_choice(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            None,
            "",
            "   ",
            "Final Answer:",
            "Final Answer: 5",  # outside 0-4
            "Final Answer: 34",  # not a single label
            "Final Answer: none of the above",
            "Final Answer: I cannot tell",
            '{"answer": 2}',  # a JSON answer object, never a label here
        ],
    )
    def test_anything_else_is_invalid_and_never_guessed(self, module, text) -> None:
        assert module.parse_choice(text) is None


class TestQuestion:
    def test_the_question_carries_the_five_labelled_choices(self, module) -> None:
        entry = {
            "q_uid": "abc",
            "question": "What does C do first?",
            "option 0": "opens the fridge",
            "option 1": "picks up a knife",
            "option 2": "washes the plate",
            "option 3": "turns on the tap",
            "option 4": "sits down",
        }

        question = module.build_question(
            entry["question"], module.options_of(entry)
        )

        assert question == (
            "What does C do first?\n"
            "0: opens the fridge\n"
            "1: picks up a knife\n"
            "2: washes the plate\n"
            "3: turns on the tap\n"
            "4: sits down"
        )

    def test_a_question_without_its_five_options_is_refused(
        self, module, tmp_path: Path
    ) -> None:
        path = tmp_path / "questions.json"
        path.write_text(
            json.dumps([{"q_uid": "abc", "question": "?", "option 0": "a"}]),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="option 1"):
            module.load_questions(path)

    def test_a_truth_outside_the_five_labels_is_refused(
        self, module, tmp_path: Path
    ) -> None:
        path = tmp_path / "answers.json"
        path.write_text(json.dumps({"abc": 7}), encoding="utf-8")

        with pytest.raises(ValueError, match="0-4"):
            module.load_answers(path)


ANSWERS = {"q1": 0, "q2": 1, "q3": 2, "q4": 3, "q5": 4}


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path, "ego-9b")


def select(module, store: RunStore, *extra: str) -> list[str]:
    return module.select_questions(ANSWERS, parse(module, *extra), store)


class TestSelection:
    def test_the_answered_subset_is_run_in_a_stable_order(
        self, module, store: RunStore
    ) -> None:
        assert select(module, store) == ["q1", "q2", "q3", "q4", "q5"]

    def test_a_question_list_runs_only_its_questions(
        self, module, store: RunStore, tmp_path: Path
    ) -> None:
        path = tmp_path / "sample.json"
        path.write_text(json.dumps(["q4", "q2"]), encoding="utf-8")

        assert select(module, store, "--question-list", str(path)) == ["q2", "q4"]

    def test_a_question_list_outside_the_subset_is_a_contract_violation(
        self, module, store: RunStore, tmp_path: Path
    ) -> None:
        # Those questions have no public answer; running them would produce
        # records nothing can score.
        path = tmp_path / "sample.json"
        path.write_text(json.dumps(["q2", "nope"]), encoding="utf-8")

        with pytest.raises(ValueError, match="nope"):
            select(module, store, "--question-list", str(path))

    def test_shards_partition_the_subset_between_workers(
        self, module, store: RunStore
    ) -> None:
        first = select(module, store, "--shard", "0/2")
        second = select(module, store, "--shard", "1/2")

        assert first == ["q1", "q3", "q5"]
        assert second == ["q2", "q4"]
        assert sorted(first + second) == select(module, store)

    def test_a_malformed_shard_is_a_contract_violation(
        self, module, store: RunStore
    ) -> None:
        with pytest.raises(ValueError):
            select(module, store, "--shard", "2/2")

    def test_limit_truncates_the_selection(self, module, store: RunStore) -> None:
        assert select(module, store, "--limit", "2") == ["q1", "q2"]

    def test_a_question_that_already_has_a_record_is_skipped(
        self, module, store: RunStore
    ) -> None:
        store.save("q2", {"status": "ok"})

        assert select(module, store) == ["q1", "q3", "q4", "q5"]

    def test_force_reruns_everything(self, module, store: RunStore) -> None:
        store.save("q2", {"status": "ok"})

        assert select(module, store, "--force") == ["q1", "q2", "q3", "q4", "q5"]

    def test_retry_errors_reruns_only_the_failed_records(
        self, module, store: RunStore
    ) -> None:
        store.save("q2", {"status": "ok"})
        store.save("q4", {"status": "error"})
        store.save("q5", {"status": "max_iterations"})  # a result, not a failure

        assert select(module, store, "--retry-errors") == ["q1", "q3", "q4"]

    def test_resumption_happens_inside_the_shard_it_belongs_to(
        self, module, store: RunStore
    ) -> None:
        store.save("q1", {"status": "ok"})

        assert select(module, store, "--shard", "0/2") == ["q3", "q5"]


def make_transcript(*completions: str, status: str = "final_answer") -> Transcript:
    steps = [
        ReActStep(
            turn=LLMTurn(
                index=index,
                completion=text,
                usage=Usage(llm_calls=1, tokens_in=10, tokens_out=5),
                latency_s=0.1,
                model="fake",
                finish_reason="stop",
            )
        )
        for index, text in enumerate(completions)
    ]
    return Transcript(
        prompt="PROMPT",
        steps=steps,
        final_answer=None,
        status=status,
        usage=Usage(llm_calls=len(steps), tokens_in=10 * len(steps), tokens_out=5),
    )


def result_with(answer: str | None, status: str) -> AgentResult:
    transcript = make_transcript("Thought: ...", status=status)
    return AgentResult(
        answer=answer, status=status, transcript=transcript, usage=transcript.usage
    )


CONFIG = {"model": "fake", "code_version": "deadbee"}


def record_for(module, result, **kwargs) -> dict:
    return module.make_record(
        q_uid="q1",
        question="Q\n0: a\n1: b\n2: c\n3: d\n4: e",
        options=["a", "b", "c", "d", "e"],
        truth=3,
        config=CONFIG,
        started_at="2000-01-01T00:00:00+00:00",
        seconds=1.5,
        result=result,
        **kwargs,
    )


class TestRecord:
    def test_a_read_label_is_ok_and_scored(self, module) -> None:
        record = record_for(module, result_with("3", "final_answer"))

        assert record["status"] == "ok"
        assert record["prediction"] == 3 and record["correct"] is True
        assert record["answer_raw"] == "3"
        assert record["truth"] == 3
        assert record["config"]["config_id"] == module.config_id(CONFIG)
        assert record["code_version"] == "deadbee"
        assert record["transcript"]["prompt"] == "PROMPT"
        assert record["seconds"] == 1.5

    def test_a_wrong_label_is_still_a_valid_answer(self, module) -> None:
        record = record_for(module, result_with("1", "final_answer"))

        assert record["status"] == "ok"
        assert record["prediction"] == 1 and record["correct"] is False

    def test_an_answer_that_is_not_a_label_is_parse_failed(self, module) -> None:
        record = record_for(module, result_with("none of them", "final_answer"))

        assert record["status"] == "parse_failed"
        # ``None`` and not ``False``: the agent gave no label to be wrong with.
        assert record["prediction"] is None and record["correct"] is None
        assert record["answer_raw"] == "none of them"

    def test_the_iteration_ceiling_keeps_its_own_status(self, module) -> None:
        record = record_for(module, result_with(None, "max_iterations"))

        assert record["status"] == "max_iterations"
        assert record["prediction"] is None

    def test_a_failed_episode_keeps_the_steps_it_had_run(self, module) -> None:
        partial = make_transcript("Thought: one", "Thought: two", status="error")

        record = record_for(
            module, None, error="TimeoutError: too slow", partial=partial
        )

        assert record["status"] == "error"
        assert record["error"] == "TimeoutError: too slow"
        assert len(record["transcript"]["steps"]) == 2
        assert record["usage"]["llm_calls"] == 2

    def test_a_failure_with_no_transcript_is_recorded_all_the_same(
        self, module
    ) -> None:
        record = record_for(module, None, error="OSError: gone")

        assert record["status"] == "error" and record["transcript"] is None
        assert record["usage"]["tokens_in"] == 0

    def test_the_configuration_id_moves_with_the_configuration(self, module) -> None:
        other = module.config_id({**CONFIG, "model": "other"})

        assert other != module.config_id(CONFIG)

    def test_the_prompt_and_the_question_format_are_part_of_the_identity(
        self, module
    ) -> None:
        """A different wording or layout is a different configuration."""
        args = parse(module)
        config = module.egoschema_config(args, "PROMPT")

        assert config["prompt_name"] == "multiple_choice_prompt"
        assert config["question_format_version"] == 1
        assert module.config_id(config) != module.config_id(
            {**config, "question_format_version": 2}
        )


class TestRunQuestion:
    ENTRY = {
        "q_uid": "q1",
        "question": "What does C do?",
        "option 0": "a",
        "option 1": "b",
        "option 2": "c",
        "option 3": "d",
        "option 4": "e",
    }

    def run(self, module, args, runner=None):
        return module.run_question(
            q_uid="q1",
            entry=self.ENTRY,
            truth=3,
            args=args,
            models=None,
            runner=runner,
            prompt_template="PROMPT",
            config=CONFIG,
        )

    def test_a_clip_with_no_memory_directory_is_recorded_missing(
        self, module, tmp_path: Path
    ) -> None:
        args = parse(module, "--artifacts-root", str(tmp_path / "preprocess"))

        record = self.run(module, args)

        assert record["status"] == "missing"
        assert record["prediction"] is None and record["transcript"] is None
        assert "q1" in record["error"]

    def test_asking_never_creates_the_memory_directory(
        self, module, tmp_path: Path
    ) -> None:
        # ``VideoArtifacts`` would, and an empty directory would read as a
        # preprocessed clip on the next pass.
        root = tmp_path / "preprocess"
        root.mkdir()
        args = parse(module, "--artifacts-root", str(root))

        self.run(module, args)

        assert list(root.iterdir()) == []

    def test_a_question_that_blows_up_is_a_record_and_not_an_exception(
        self, module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """500 questions must not stop because one clip's tools failed."""
        from video_agent.agent.react import ReActError
        from video_agent.memory import store as memory_store

        root = tmp_path / "preprocess"
        (root / "q1").mkdir(parents=True)
        args = parse(module, "--artifacts-root", str(root))
        partial = make_transcript("Thought: one", status="error")
        monkeypatch.setattr(
            memory_store, "load_video_meta", lambda *a, **k: object()
        )
        monkeypatch.setattr(
            module,
            "build_toolkit",
            lambda *a, **k: (_ for _ in ()).throw(
                ReActError("boom", transcript=partial)
            ),
        )

        record = self.run(module, args)

        assert record["status"] == "error"
        assert record["error"].startswith("ReActError: boom")
        # The steps the episode had already run are kept, not lost with it.
        assert len(record["transcript"]["steps"]) == 1


class TestPredictions:
    def build(self, module, store: RunStore) -> None:
        store.save(
            "q1", {"q_uid": "q1", "prediction": 3, "status": "ok", "answer_raw": "3"}
        )
        store.save(
            "q2",
            {
                "q_uid": "q2",
                "prediction": None,
                "status": "parse_failed",
                "answer_raw": "maybe",
            },
        )
        store.save(
            "q3", {"q_uid": "q3", "prediction": 0, "status": "ok", "answer_raw": "0"}
        )

    def test_the_labels_are_read_back_from_the_records_on_disk(
        self, module, store: RunStore
    ) -> None:
        self.build(module, store)

        assert module.collect_predictions(store.directory) == {
            "q1": 3,
            "q2": None,
            "q3": 0,
        }

    def test_an_empty_run_directory_has_no_predictions(
        self, module, tmp_path: Path
    ) -> None:
        assert module.collect_predictions(tmp_path / "nothing") == {}

    def test_the_server_file_holds_only_the_answered_questions(
        self, module, store: RunStore
    ) -> None:
        self.build(module, store)

        predictions_path, invalid_path = module.write_predictions(store.directory)

        assert json.loads(predictions_path.read_text()) == {"q1": 3, "q3": 0}
        assert json.loads(invalid_path.read_text()) == [
            {
                "q_uid": "q2",
                "status": "parse_failed",
                "answer_raw": "maybe",
                "error": None,
            }
        ]

    def test_the_aggregate_is_rebuilt_and_not_appended_to(
        self, module, store: RunStore
    ) -> None:
        """Two shards and a resumed run must agree with one process."""
        self.build(module, store)
        module.write_predictions(store.directory)
        store.save(
            "q2", {"q_uid": "q2", "prediction": 1, "status": "ok", "answer_raw": "1"}
        )

        predictions_path, invalid_path = module.write_predictions(store.directory)

        assert json.loads(predictions_path.read_text()) == {"q1": 3, "q2": 1, "q3": 0}
        assert json.loads(invalid_path.read_text()) == []

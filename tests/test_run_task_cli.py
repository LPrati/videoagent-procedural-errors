"""The batch CLI: argument defaults and which recordings a process runs.

``scripts/`` is not a package, so the module is loaded from its path; the
directory goes on ``sys.path`` first because the script imports its siblings
``_env`` and ``_wiring`` the way it does when run directly.
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

from video_agent.agent.config import DEFAULT_ALIGNER_MODEL
from video_agent.task.run import (
    PACKAGE_TEMPLATE,
    RunStore,
    TaskConfig,
    config_id,
    template_sources,
)

from tests._paths import DEFAULT_ANNOTATIONS_DIR, SCRIPTS

BASE_ARGV = [
    "--split", "data_splits/dev.json",
    "--artifacts-root", "preprocess",
    "--videos-dir", "videos",
    "--out", "runs",
    "--run-name", "agent",
]


@pytest.fixture(scope="module")
def run_task_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("run_task_cli", SCRIPTS / "run_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse(run_task_module, *extra: str):
    return run_task_module.parse_args(BASE_ARGV + list(extra))


def test_defaults_are_the_published_configuration(run_task_module) -> None:
    args = parse(run_task_module)

    assert args.split == Path("data_splits/dev.json")
    assert args.subset == "all"
    assert args.out == Path("runs") and args.run_name == "agent"
    assert args.caption_variant == "default"
    assert args.model == "Qwen/Qwen3.5-9B"
    assert args.chat == "vllm"
    assert args.base_url is None  # filled in by the backend, see _wiring
    assert args.vqa == "api" and args.vqa_style == "two_questions"
    assert args.device == "cpu"
    assert args.max_iterations == 100 and args.temperature == 0.0
    assert args.max_tokens == 16384 and args.chat_timeout is None
    assert not args.thinking
    assert args.seed is None and args.shard is None and args.limit is None
    assert args.order == "split"
    assert args.object_memory is None  # the unsuffixed memory of the default detector
    assert args.memory == "full"  # the recording's own memory
    assert not args.no_objects and not args.force and not args.retry_errors


class TestMemoryFlag:
    """The three memory conditions: full, empty and shuffled (the two controls)."""

    def test_the_three_conditions_are_the_choices(self, run_task_module) -> None:
        for mode in ("full", "empty", "shuffled"):
            assert parse(run_task_module, "--memory", mode).memory == mode
        with pytest.raises(SystemExit):
            parse(run_task_module, "--memory", "scrambled")

    def test_the_default_memory_keeps_the_full_memory_config_id(
        self, run_task_module
    ) -> None:
        """``full`` leaves the configuration exactly as without the flag.

        ``memory`` is left out of ``to_dict`` when it is ``full``, so the id
        of a full-memory run does not depend on the flag being passed — and
        it is the configuration the controls compare against.
        """
        default = TestTemplateOverrides.pinned(
            run_task_module.task_config(parse(run_task_module), "PROMPT", "QUESTION")
        )
        explicit = TestTemplateOverrides.pinned(
            run_task_module.task_config(
                parse(run_task_module, "--memory", "full"), "PROMPT", "QUESTION"
            )
        )

        assert default.memory == "full"
        assert "memory" not in default.to_dict()
        assert "memory_seed" not in default.to_dict()
        assert default.to_dict() == explicit.to_dict()
        assert config_id(default) == config_id(explicit)
        # And the id of the packaged templates is the very one already pinned.
        packaged = TestTemplateOverrides()
        assert config_id(packaged.config(run_task_module)) == (
            config_id(packaged.config(run_task_module, "--memory", "full"))
        ) == TestTemplateOverrides.PACKAGE_CONFIG_ID

    def test_each_condition_is_a_configuration_of_its_own(
        self, run_task_module
    ) -> None:
        def pinned_config(*extra: str) -> TaskConfig:
            return TestTemplateOverrides.pinned(
                run_task_module.task_config(
                    parse(run_task_module, *extra), "PROMPT", "QUESTION"
                )
            )

        full = pinned_config()
        empty = pinned_config("--memory", "empty")
        shuffled = pinned_config("--memory", "shuffled")

        assert empty.to_dict()["memory"] == "empty"
        assert "memory_seed" not in empty.to_dict()
        assert shuffled.to_dict()["memory"] == "shuffled"
        # The base seed is part of the configuration; the permutation of one
        # recording is not, so all records of a shuffled run share one id.
        assert shuffled.to_dict()["memory_seed"] == 20260903
        assert len({config_id(full), config_id(empty), config_id(shuffled)}) == 3

    def test_another_base_seed_is_another_configuration(self, run_task_module) -> None:
        import dataclasses

        shuffled = TestTemplateOverrides.pinned(
            run_task_module.task_config(
                parse(run_task_module, "--memory", "shuffled"), "PROMPT", "QUESTION"
            )
        )
        redrawn = dataclasses.replace(shuffled, memory_seed=20260904)

        assert config_id(shuffled) != config_id(redrawn)


class TestSplitOrdinals:
    """Which seed a recording's permutation is drawn from."""

    def test_the_position_is_the_recordings_place_in_the_sorted_split(
        self, run_task_module
    ) -> None:
        ordinals = run_task_module.split_ordinals(ERROR_IDS, NORMAL_IDS)

        assert ordinals == {"1_1": 0, "1_10": 1, "1_2": 2, "1_3": 3, "1_4": 4}

    def test_it_does_not_depend_on_who_runs_the_recording(
        self, run_task_module, store: RunStore
    ) -> None:
        """Sharding, subsets and limits pick the worker, not the shuffle."""
        ordinals = run_task_module.split_ordinals(ERROR_IDS, NORMAL_IDS)
        whole_split = select(run_task_module, store)

        for shard in ("0/2", "1/2"):
            for recording_id in select(run_task_module, store, "--shard", shard):
                assert ordinals[recording_id] == whole_split.index(recording_id)
        assert run_task_module.split_ordinals(ERROR_IDS, []) != ordinals
        assert (
            run_task_module.split_ordinals(list(reversed(ERROR_IDS)), NORMAL_IDS)
            == ordinals
        )


def test_the_turn_switches_are_parsed_and_frozen_into_the_configuration(
    run_task_module,
) -> None:
    """``--max-tokens``, ``--chat-timeout`` and ``--thinking`` change a run.

    All three can change a verdict, so all three have to reach the record:
    the cap and the reasoning switch through the config that ``config_id``
    hashes, the timeout only through the adapter it is passed to.
    """
    args = parse(
        run_task_module,
        "--max-tokens", "4096",
        "--chat-timeout", "600",
        "--thinking",
    )

    assert (args.max_tokens, args.chat_timeout, args.thinking) == (4096, 600.0, True)

    config = run_task_module.task_config(args, "prompt", "question")
    default = run_task_module.task_config(parse(run_task_module), "prompt", "question")

    assert config.thinking is True and default.thinking is False
    assert config.to_dict()["thinking"] is True
    assert config.react.max_tokens == 4096
    assert config_id(config) != config_id(default)


def test_an_unlimited_turn_is_asked_for_with_zero(run_task_module) -> None:
    args = parse(run_task_module, "--max-tokens", "0")

    assert run_task_module.task_config(args, "p", "q").react.max_tokens is None


class TestAligner:
    """Which video-text aligner the agent runs, and how a change is recorded."""

    def test_the_default_is_pe_core_l(self, run_task_module) -> None:
        args = parse(run_task_module)

        assert args.aligner_model == DEFAULT_ALIGNER_MODEL == "timm/PE-Core-L-14-336"

    def test_the_flag_reaches_the_configuration_and_moves_its_id(
        self, run_task_module
    ) -> None:
        # The two aligners score in different spaces off different files, so
        # their records must not be comparable by config_id.
        args = parse(run_task_module, "--aligner-model", "microsoft/xclip-base-patch16")

        config = run_task_module.task_config(args, "PROMPT", "QUESTION")
        default = run_task_module.task_config(
            parse(run_task_module), "PROMPT", "QUESTION"
        )

        assert config.tools.aligner_model == "microsoft/xclip-base-patch16"
        assert default.tools.aligner_model == DEFAULT_ALIGNER_MODEL
        assert config.to_dict()["tools"]["aligner_model"] == (
            "microsoft/xclip-base-patch16"
        )
        assert config_id(config) != config_id(default)

    def test_an_unknown_aligner_is_refused_before_the_batch_starts(
        self, run_task_module
    ) -> None:
        # A whole split would otherwise fail one at a time on the same typo.
        with pytest.raises(SystemExit):
            parse(run_task_module, "--aligner-model", "nobody/nothing")


class TestObjectMemory:
    """Which detector's object memory the sub-agent queries."""

    def test_the_flag_reaches_the_configuration_and_moves_its_id(
        self, run_task_module
    ) -> None:
        # Two detectors put different names in the memory, so what the
        # sub-agent can find differs and the records must not compare equal.
        args = parse(run_task_module, "--object-memory", "other-detector")

        config = run_task_module.task_config(args, "PROMPT", "QUESTION")
        default = run_task_module.task_config(
            parse(run_task_module), "PROMPT", "QUESTION"
        )

        assert config.tools.object_memory_slug == "other-detector"
        assert default.tools.object_memory_slug is None
        assert config.to_dict()["tools"]["object_memory_slug"] == (
            "other-detector"
        )
        assert config_id(config) != config_id(default)

    def test_a_slug_that_could_name_another_file_is_refused(
        self, run_task_module
    ) -> None:
        with pytest.raises(SystemExit):
            parse(run_task_module, "--object-memory", "../elsewhere")

    def test_the_recordings_without_that_memory_are_named_before_the_batch(
        self, run_task_module, tmp_path: Path
    ) -> None:
        """A whole split would otherwise fail one at a time, hours in."""
        root = tmp_path / "preprocess"
        (root / "1_1_360p").mkdir(parents=True)
        (root / "1_1_360p" / "object_memory.other-detector.pkl").touch()
        (root / "1_3_360p").mkdir()
        (root / "1_3_360p" / "object_memory.pkl").touch()  # the other detector's

        missing = run_task_module.check_object_memories(
            root, ["1_1", "1_3", "1_9"], "other-detector"
        )

        assert missing == ["1_3", "1_9"]

    def test_asking_about_a_recording_never_creates_its_directory(
        self, run_task_module, tmp_path: Path
    ) -> None:
        # ``VideoArtifacts`` would; a check must not leave empty directories
        # across a whole corpus of recordings.
        root = tmp_path / "preprocess"
        root.mkdir()

        run_task_module.check_object_memories(root, ["1_1"], None)

        assert list(root.iterdir()) == []


def test_the_annotations_default_resolves_from_the_repository_root(
    run_task_module,
) -> None:
    args = parse(run_task_module)

    assert args.annotations_dir.is_absolute()
    assert args.annotations_dir == DEFAULT_ANNOTATIONS_DIR


def test_rejects_an_unknown_subset(run_task_module) -> None:
    with pytest.raises(SystemExit):
        parse(run_task_module, "--subset", "reserve")


def test_requires_a_run_name(run_task_module) -> None:
    with pytest.raises(SystemExit):
        run_task_module.parse_args(
            ["--split", "s.json", "--artifacts-root", "a", "--videos-dir", "v", "--out", "r"]
        )


ERROR_IDS = ["1_2", "1_10", "1_4"]
NORMAL_IDS = ["1_1", "1_3"]


def select(run_task_module, store: RunStore, *extra: str) -> list[str]:
    return run_task_module.select_recordings(
        ERROR_IDS, NORMAL_IDS, parse(run_task_module, *extra), store
    )


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path, "agent")


def test_all_recordings_of_the_split_are_run_in_a_stable_order(
    run_task_module, store: RunStore
) -> None:
    assert select(run_task_module, store) == ["1_1", "1_10", "1_2", "1_3", "1_4"]


def test_a_subset_selects_one_half_of_the_split(
    run_task_module, store: RunStore
) -> None:
    assert select(run_task_module, store, "--subset", "error") == ["1_10", "1_2", "1_4"]
    assert select(run_task_module, store, "--subset", "normal") == ["1_1", "1_3"]


def test_shards_partition_the_split_between_workers(
    run_task_module, store: RunStore
) -> None:
    first = select(run_task_module, store, "--shard", "0/2")
    second = select(run_task_module, store, "--shard", "1/2")

    assert first == ["1_1", "1_2", "1_4"]
    assert second == ["1_10", "1_3"]
    assert sorted(first + second) == select(run_task_module, store)


def test_a_malformed_shard_is_a_contract_violation(
    run_task_module, store: RunStore
) -> None:
    with pytest.raises(ValueError):
        select(run_task_module, store, "--shard", "2/2")


def test_limit_truncates_the_selection(run_task_module, store: RunStore) -> None:
    assert select(run_task_module, store, "--limit", "2") == ["1_1", "1_10"]


def test_shuffling_reorders_the_same_recordings_for_a_given_seed(
    run_task_module, store: RunStore
) -> None:
    shuffled = select(run_task_module, store, "--order", "shuffle", "--seed", "7")

    assert sorted(shuffled) == select(run_task_module, store)
    assert shuffled == select(run_task_module, store, "--order", "shuffle", "--seed", "7")
    assert shuffled != select(run_task_module, store)


def test_recordings_that_already_have_a_record_are_skipped(
    run_task_module, store: RunStore
) -> None:
    store.save("1_2", {"status": "ok"})

    assert select(run_task_module, store) == ["1_1", "1_10", "1_3", "1_4"]


def test_force_reruns_everything(run_task_module, store: RunStore) -> None:
    store.save("1_2", {"status": "ok"})

    assert select(run_task_module, store, "--force") == [
        "1_1", "1_10", "1_2", "1_3", "1_4"
    ]


def test_retry_errors_reruns_only_the_failed_records(
    run_task_module, store: RunStore
) -> None:
    store.save("1_2", {"status": "ok"})
    store.save("1_4", {"status": "error"})

    assert select(run_task_module, store, "--retry-errors") == [
        "1_1", "1_10", "1_3", "1_4"
    ]


def test_resumption_happens_inside_the_shard_it_belongs_to(
    run_task_module, store: RunStore
) -> None:
    store.save("1_1", {"status": "ok"})

    assert select(run_task_module, store, "--shard", "0/2") == ["1_2", "1_4"]


def test_the_local_backend_is_selectable_and_resolves_its_endpoint(
    run_task_module,
) -> None:
    from _wiring import resolve_base_url

    args = parse(run_task_module, "--chat", "vllm")

    assert args.chat == "vllm" and args.base_url is None
    assert resolve_base_url(args) == "http://localhost:8000/v1"
    assert (
        resolve_base_url(parse(run_task_module, "--chat", "openrouter"))
        == "https://openrouter.ai/api/v1"
    )


def test_the_backend_reaches_the_run_record(run_task_module) -> None:
    config = run_task_module.task_config(
        parse(run_task_module, "--chat", "vllm"), "PROMPT", "QUESTION"
    )

    assert config.chat_backend == "vllm"
    assert config.to_dict()["chat_backend"] == "vllm"


def test_the_task_configuration_records_both_templates(run_task_module) -> None:
    from video_agent.agent.prompts import prompt_sha256

    config = run_task_module.task_config(parse(run_task_module), "PROMPT", "QUESTION")

    assert config.prompt_sha256 == prompt_sha256("PROMPT")
    assert config.question_template_sha256 == prompt_sha256("QUESTION")
    assert config.orchestrator_model == "Qwen/Qwen3.5-9B"
    assert config.use_objects is True
    assert config.react.max_iterations == 100


class TestTemplateOverrides:
    """Running a wording that is not the packaged one (``--question-file``).

    A run with another wording is a run like any other: the text it used has
    to reach the record, and two wordings must not answer to the same
    ``config_id``. The file is read and checked at parse time, so a template
    the renderer could not fill fails before any model is called.
    """

    PACKAGE_CONFIG_ID = "f59f559d3e88f58d"
    """``config_id`` of the defaults, with ``code_version`` pinned below.

    Written down so that a change to the packaged templates, or to what the
    id is computed from, is a failing test and not a silently incomparable
    set of runs.
    """

    PACKAGE_PROMPT_SHA256 = (
        "505be46beb64f4331cc1076e4c4af591c74c3936e0b029fce10cf028e946003e"
    )
    PACKAGE_QUESTION_SHA256 = (
        "5d4db923e87aa864ee5c413b8dcc2d05ae740381003bddc23be84a7f0981775c"
    )

    QUESTION_V2_TEXT = (
        "Recipe: {recipe_name}. The video has {n_segments} segments.\n"
        "Steps:\n{steps}\n"
        "Prerequisites:\n{prerequisites}\n"
        "For every step, say whether it was executed and in which order.\n"
        "{schema}\n"
    )
    """An alternative question wording, with all five of its placeholders."""

    @classmethod
    def question_v2(cls, directory: Path) -> Path:
        """Write the alternative question into ``directory`` and return its path."""
        path = directory / "task_question_v2.txt"
        path.write_text(cls.QUESTION_V2_TEXT, encoding="utf-8")
        return path

    @staticmethod
    def pinned(config: TaskConfig) -> TaskConfig:
        """The configuration with its code version fixed, so an id can be pinned.

        ``code_version`` is the checkout's HEAD and is hashed into
        ``config_id``, so an id computed from a live configuration moves with
        every change to the code. Replacing it keeps what the pin is about: the templates
        and the defaults around them.
        """
        return dataclasses.replace(config, code_version="deadbee")

    def config(self, run_task_module, *extra: str) -> TaskConfig:
        """The configuration ``main`` would freeze for these arguments."""
        args = parse(run_task_module, *extra)
        prompt, prompt_source = run_task_module.resolve_template(
            args.task_prompt_file, "task_prompt", run_task_module.REACT_PLACEHOLDERS
        )
        question, question_source = run_task_module.resolve_template(
            args.question_file, "task_question", run_task_module.QUESTION_PLACEHOLDERS
        )
        return self.pinned(
            run_task_module.task_config(
                args,
                prompt,
                question,
                prompt_source=prompt_source,
                question_source=question_source,
            )
        )

    def test_without_the_flags_nothing_moves(self, run_task_module) -> None:
        """The packaged templates stay the default, byte for byte."""
        from video_agent.agent.prompts import load_prompt

        args = parse(run_task_module)
        assert args.question_file is None and args.task_prompt_file is None

        config = self.config(run_task_module)

        assert run_task_module.resolve_template(
            None, "task_question", run_task_module.QUESTION_PLACEHOLDERS
        ) == (load_prompt("task_question"), PACKAGE_TEMPLATE)
        assert config.prompt_sha256 == self.PACKAGE_PROMPT_SHA256
        assert config.question_template_sha256 == self.PACKAGE_QUESTION_SHA256
        assert config.prompt_template_source == PACKAGE_TEMPLATE
        assert config.question_template_source == PACKAGE_TEMPLATE
        assert config_id(config) == self.PACKAGE_CONFIG_ID

    def test_a_question_file_is_the_question_and_a_new_configuration(
        self, run_task_module, tmp_path: Path
    ) -> None:
        from video_agent.agent.prompts import prompt_sha256

        question_v2 = self.question_v2(tmp_path)
        text = question_v2.read_text(encoding="utf-8")
        config = self.config(run_task_module, "--question-file", str(question_v2))

        assert config.question_template_sha256 == prompt_sha256(text)
        assert config.question_template_sha256 != self.PACKAGE_QUESTION_SHA256
        # The ReAct prompt was not overridden and must not have moved.
        assert config.prompt_sha256 == self.PACKAGE_PROMPT_SHA256
        assert config_id(config) != self.PACKAGE_CONFIG_ID

    def test_a_prompt_file_replaces_only_the_react_prompt(
        self, run_task_module, tmp_path: Path
    ) -> None:
        from video_agent.agent.prompts import prompt_sha256

        path = tmp_path / "task_prompt_v2.txt"
        path.write_text(
            "Tools: {tools}\nNames: {tool_names}\n{input}\n{agent_scratchpad}",
            encoding="utf-8",
        )

        config = self.config(run_task_module, "--task-prompt-file", str(path))

        assert config.prompt_sha256 == prompt_sha256(
            path.read_text(encoding="utf-8")
        )
        assert config.question_template_sha256 == self.PACKAGE_QUESTION_SHA256
        assert config.prompt_template_source == str(path)
        assert config_id(config) != self.PACKAGE_CONFIG_ID

    def test_where_the_template_came_from_is_recorded_but_never_hashed(
        self, run_task_module, tmp_path: Path
    ) -> None:
        """Two machines running the same wording run the same configuration.

        The digest of the text already says which wording ran; the path says
        where to find it again, and hashing it would give a copy of the same
        question a different id under every checkout.
        """
        text = self.QUESTION_V2_TEXT
        here, there = tmp_path / "here.txt", tmp_path / "there.txt"
        here.write_text(text, encoding="utf-8")
        there.write_text(text, encoding="utf-8")

        first = self.config(run_task_module, "--question-file", str(here))
        second = self.config(run_task_module, "--question-file", str(there))

        assert first.question_template_source == str(here)
        assert second.question_template_source == str(there)
        assert config_id(first) == config_id(second)
        assert "question_template_source" not in first.to_dict()
        assert "prompt_template_source" not in first.to_dict()
        # Out of the hash, but in the record: that is what the record is for.
        assert template_sources(first) == {
            "prompt_template_source": PACKAGE_TEMPLATE,
            "question_template_source": str(here),
        }

    def test_a_missing_placeholder_is_refused_before_any_model_is_loaded(
        self, run_task_module, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """Rendering is ``str.replace``: a typo would fail silently, hours in."""
        path = tmp_path / "question.txt"
        path.write_text(
            "Steps of {recipe_name}: {steps}\n{prerequisites}\n{schema}",
            encoding="utf-8",
        )

        with pytest.raises(SystemExit):
            parse(run_task_module, "--question-file", str(path))

        assert "{n_segments}" in capsys.readouterr().err

    def test_a_question_that_spells_out_its_own_answer_object_may_skip_schema(
        self, run_task_module, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """An answer object without the order fields, written in the file."""
        path = tmp_path / "question.txt"
        path.write_text(
            "Steps of {recipe_name} in {n_segments}: {steps}\n{prerequisites}\n"
            '{"steps": [{"step_id": <int>, "executed": "yes"|"no"|"unsure", '
            '"segments": [<first>, <last>] or null}]}',
            encoding="utf-8",
        )
        args = parse(run_task_module, "--question-file", str(path))
        assert args.question_file == path

        path.write_text("Steps of {recipe_name} in {n_segments}: {steps}\n{prerequisites}\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            parse(run_task_module, "--question-file", str(path))
        assert "{schema}" in capsys.readouterr().err

    def test_a_react_prompt_without_its_scratchpad_is_refused(
        self, run_task_module, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        path = tmp_path / "prompt.txt"
        path.write_text("{tools}\n{tool_names}\n{input}", encoding="utf-8")

        with pytest.raises(SystemExit):
            parse(run_task_module, "--task-prompt-file", str(path))

        assert "{agent_scratchpad}" in capsys.readouterr().err

    def test_a_template_file_that_is_not_there_is_refused_too(
        self, run_task_module, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit):
            parse(run_task_module, "--question-file", str(tmp_path / "nowhere.txt"))


class TestSeparateVqaEndpoint:
    """A hosted orchestrator with the visual questions kept on the local model."""

    def test_by_default_the_orchestrator_endpoint_answers(self, run_task_module) -> None:
        from _wiring import vqa_chat_settings

        args = parse(run_task_module)
        config = run_task_module.task_config(args, "PROMPT", "QUESTION")

        assert vqa_chat_settings(args) is None
        assert config.vqa_endpoint_model is None
        assert "vqa_endpoint_model" not in config.to_dict()

    def test_a_vqa_endpoint_is_local_shaped_and_recorded(self, run_task_module) -> None:
        from _wiring import vqa_chat_settings

        args = parse(
            run_task_module,
            "--chat", "openrouter", "--model", "org/hosted-model",
            "--vqa-base-url", "http://localhost:8000/v1",
        )
        settings = vqa_chat_settings(args)
        config = run_task_module.task_config(args, "PROMPT", "QUESTION")

        assert settings["base_url"] == "http://localhost:8000/v1"
        assert settings["model_name"] == "Qwen/Qwen3.5-9B"
        assert settings["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert config.to_dict()["vqa_endpoint_model"] == "Qwen/Qwen3.5-9B"

    def test_it_needs_the_api_vqa(self, run_task_module) -> None:
        with pytest.raises(SystemExit):
            parse(run_task_module, "--vqa", "none", "--vqa-base-url", "http://x/v1")

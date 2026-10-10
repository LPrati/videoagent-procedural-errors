"""The report CLI: what it selects, what it writes, and the noise-floor recipe.

``scripts/`` is not a package, so the module is loaded from its path the way
``test_run_task_cli`` does. The end-to-end runs go over the hand-written store
of ``test_report``: no model, no video, and every file the script produces is
checked for existence and for being readable JSON or Markdown.
"""

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from tests.test_report import ERROR_ID, MISSING_ID, NORMAL_ID, build_store

from video_agent.task.annotations import AnnotationSet
from video_agent.task.run import RUN_FILE, RunStore

from tests._paths import (
    ANNOTATIONS_DIR,
    DEFAULT_ANNOTATIONS_DIR,
    REPO_ROOT,
    SCRIPTS,
    load_annotations,
)

RUN_FILES = (
    "metrics.json",
    "results_row.md",
    "results_row.csv",
    "literature_row.md",
    "localization.md",
    "localization.csv",
)
"""What a single-run report always writes."""


@pytest.fixture(scope="module")
def evaluate_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "evaluate_cli", SCRIPTS / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_ARGV = [
    "--runs", "runs/agent", "--split", "data_splits/dev_83.json", "--out", "out"
]


def parse(evaluate_module, *extra: str):
    return evaluate_module.parse_args(BASE_ARGV + list(extra))


class TestArguments:
    def test_defaults_score_the_whole_split_with_2000_bootstrap_resamples(
        self, evaluate_module
    ) -> None:
        args = parse(evaluate_module)

        assert args.subset == "all"
        assert args.bootstrap == 2000
        assert args.seed == 0
        assert args.runs_b is None
        assert args.noise_floor is None
        assert args.write_noise_floor is False

    def test_the_order_verdict_defaults_to_the_column_the_model_wrote(
        self, evaluate_module
    ) -> None:
        """The computed order is opt-in: the judged order is the default."""
        args = parse(evaluate_module)

        assert args.order_from == "llm"
        assert args.order_rule == "transitive"
        assert args.order_edges == "both"
        assert args.order_margin == "end"
        assert evaluate_module.order_policy(args).label == "llm"

    def test_the_four_order_flags_build_the_policy(self, evaluate_module) -> None:
        args = parse(
            evaluate_module,
            "--order-from", "anchors",
            "--order-rule", "direct",
            "--order-edges", "early",
            "--order-margin", "start",
        )

        assert evaluate_module.order_policy(args).label == "anchors:direct,early,start"

    def test_an_unknown_order_reading_is_refused(self, evaluate_module) -> None:
        with pytest.raises(SystemExit):
            parse(evaluate_module, "--order-from", "oracle")
        with pytest.raises(SystemExit):
            parse(evaluate_module, "--order-margin", "middle")

    def test_a_run_of_mixed_configurations_has_to_be_asked_for(
        self, evaluate_module
    ) -> None:
        assert parse(evaluate_module).allow_mixed_config is False
        assert parse(evaluate_module, "--allow-mixed-config").allow_mixed_config

    def test_the_annotations_default_resolves_from_the_repository_root(
        self, evaluate_module
    ) -> None:
        args = parse(evaluate_module)

        assert args.annotations_dir == DEFAULT_ANNOTATIONS_DIR

    def test_a_report_needs_a_run_and_an_output_directory(
        self, evaluate_module
    ) -> None:
        with pytest.raises(SystemExit):
            evaluate_module.parse_args(["--runs", "runs/agent"])

    def test_an_unknown_subset_is_refused(self, evaluate_module) -> None:
        with pytest.raises(SystemExit):
            parse(evaluate_module, "--subset", "reserve")

    def test_a_floor_without_a_second_run_has_nothing_to_apply_to(
        self, evaluate_module
    ) -> None:
        with pytest.raises(SystemExit):
            parse(evaluate_module, "--noise-floor", "floor.json")
        with pytest.raises(SystemExit):
            parse(evaluate_module, "--write-noise-floor")


class TestPathResolution:
    def test_a_relative_path_is_anchored_at_the_repository(
        self, evaluate_module
    ) -> None:
        assert evaluate_module.resolve_path(Path("runs/agent")) == REPO_ROOT / "runs/agent"

    def test_an_absolute_path_is_left_alone(self, evaluate_module) -> None:
        assert evaluate_module.resolve_path(Path("/tmp/runs")) == Path("/tmp/runs")

    def test_the_last_component_of_runs_names_the_run(self, evaluate_module) -> None:
        store = evaluate_module.open_store(Path("/tmp/runs/agent"))

        assert isinstance(store, RunStore)
        assert store.root == Path("/tmp/runs")
        assert store.run_name == "agent"
        assert store.directory == Path("/tmp/runs/agent")

    def test_a_trailing_slash_still_names_the_run(self, evaluate_module) -> None:
        store = evaluate_module.open_store(Path("/tmp/runs/agent/"))

        assert (store.root, store.run_name) == (Path("/tmp/runs"), "agent")

    def test_a_path_that_names_no_run_is_a_contract_violation(
        self, evaluate_module
    ) -> None:
        with pytest.raises(ValueError):
            evaluate_module.open_store(Path("/"))


ERROR_IDS = ["1_2", "1_10", "1_2"]
NORMAL_IDS = ["1_3", "1_1"]


class TestSelectRecordings:
    def test_the_whole_split_is_sorted_and_deduplicated(
        self, evaluate_module
    ) -> None:
        assert evaluate_module.select_recordings(ERROR_IDS, NORMAL_IDS, "all") == [
            "1_1",
            "1_10",
            "1_2",
            "1_3",
        ]

    def test_a_subset_scores_exactly_the_half_that_was_run(
        self, evaluate_module
    ) -> None:
        assert evaluate_module.select_recordings(ERROR_IDS, NORMAL_IDS, "error") == [
            "1_10",
            "1_2",
        ]
        assert evaluate_module.select_recordings(ERROR_IDS, NORMAL_IDS, "normal") == [
            "1_1",
            "1_3",
        ]

    def test_an_unknown_subset_is_a_contract_violation(self, evaluate_module) -> None:
        with pytest.raises(ValueError):
            evaluate_module.select_recordings(ERROR_IDS, NORMAL_IDS, "reserve")


class TestLoadNoiseFloor:
    def test_an_undefined_metric_keeps_no_floor(
        self, evaluate_module, tmp_path: Path
    ) -> None:
        path = tmp_path / "floor.json"
        path.write_text(json.dumps({"order.recall": 0.02, "order.f1": None}))

        assert evaluate_module.load_noise_floor(path) == {
            "order.recall": 0.02,
            "order.f1": None,
        }

    def test_a_file_that_is_not_a_table_of_metrics_is_refused(
        self, evaluate_module, tmp_path: Path
    ) -> None:
        path = tmp_path / "floor.json"
        path.write_text("[0.1, 0.2]")

        with pytest.raises(ValueError):
            evaluate_module.load_noise_floor(path)


@pytest.fixture(scope="module")
def annotations() -> AnnotationSet:
    return load_annotations()


def clone_run(runs: Path, source: str, name: str, **config: object) -> Path:
    """Copy a run and stamp ``config`` over the configuration of every record.

    Cheaper than rebuilding the store, and it changes exactly the field the
    configuration-consistency and replica checks read.
    """
    target = runs / name
    shutil.copytree(runs / source, target)
    for path in sorted(target.glob(f"*/{RUN_FILE}")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["config"] = {**record["config"], **config}
        path.write_text(json.dumps(record), encoding="utf-8")
    return target


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, annotations: AnnotationSet) -> dict:
    """A store, copies of it under other configurations, and a split file.

    ``agent-copy`` is the same configuration byte for byte; ``agent-seed2``
    is a noise-floor replica (only the seed moved); ``agent-objects`` is a
    different configuration; ``agent-mixed`` holds two configurations at
    once.
    """
    root = tmp_path_factory.mktemp("cli")
    runs = root / "runs"
    build_store(runs, annotations, "agent")
    shutil.copytree(runs / "agent", runs / "agent-copy")
    clone_run(runs, "agent", "agent-seed2", seed=2, config_id="fedcba9876543210")
    clone_run(
        runs,
        "agent",
        "agent-objects",
        caption_variant="objects",
        config_id="aaaabbbbccccdddd",
    )
    mixed = clone_run(runs, "agent", "agent-mixed")
    record_path = mixed / ERROR_ID / RUN_FILE
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["config"] = {**record["config"], "config_id": "9999888877776666"}
    record_path.write_text(json.dumps(record), encoding="utf-8")
    split = root / "split.json"
    split.write_text(
        json.dumps({"error": [ERROR_ID], "normal": [NORMAL_ID, MISSING_ID]}),
        encoding="utf-8",
    )
    return {"root": root, "runs": runs, "split": split}


def run_main(
    evaluate_module,
    monkeypatch,
    workspace: dict,
    out: Path,
    *extra: str,
    run: str = "agent",
):
    argv = [
        "evaluate.py",
        "--runs", str(workspace["runs"] / run),
        "--split", str(workspace["split"]),
        "--out", str(out),
        "--bootstrap", "20",
        "--annotations-dir", str(ANNOTATIONS_DIR),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate_module.main()


class TestMain:
    def test_one_run_writes_the_six_report_files(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "report"

        run_main(evaluate_module, monkeypatch, workspace, out)

        assert sorted(path.name for path in out.iterdir()) == sorted(RUN_FILES)
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["run_name"] == "agent"
        assert metrics["recording_ids"] == sorted([ERROR_ID, NORMAL_ID, MISSING_ID])
        assert metrics["status_counts"] == {"missing": 1, "ok": 2}
        assert metrics["n_bootstrap"] == 20

    def test_two_replicas_record_the_noise_floor_of_the_configuration(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "floor"

        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            out,
            "--runs-b", str(workspace["runs"] / "agent-seed2"),
            "--write-noise-floor",
        )

        assert (out / "comparison.md").is_file()
        assert (out / "absolute_difference.json").is_file()
        floor = json.loads((out / "noise_floor.json").read_text(encoding="utf-8"))
        # Same records twice: the floor of a perfectly reproducible run is zero.
        assert floor == json.loads(
            (out / "absolute_difference.json").read_text(encoding="utf-8")
        )
        assert floor["order.recall"] == 0.0

    def test_the_detail_file_shows_which_estimate_the_floor_came_from(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        """floor = max(|A − B|, half the width of the paired interval)."""
        out = tmp_path / "floor-detail"

        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            out,
            "--runs-b", str(workspace["runs"] / "agent-seed2"),
            "--write-noise-floor",
        )

        detail = json.loads(
            (out / "noise_floor_detail.json").read_text(encoding="utf-8")
        )
        floor = json.loads((out / "noise_floor.json").read_text(encoding="utf-8"))

        assert set(detail) == set(floor)
        assert set(detail["order.recall"]) == {
            "abs_difference",
            "half_ci_width",
            "floor",
        }
        for key, entry in detail.items():
            estimates = [
                value
                for value in (entry["abs_difference"], entry["half_ci_width"])
                if value is not None
            ]
            assert entry["floor"] == (max(estimates) if estimates else None), key
            assert floor[key] == entry["floor"]

    def test_the_comparison_says_the_floor_only_vetoes(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "veto"

        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            out,
            "--runs-b", str(workspace["runs"] / "agent-copy"),
        )

        text = (out / "comparison.md").read_text(encoding="utf-8")

        assert "necessary condition, not a sufficient one" in text

    def test_a_floor_written_from_two_different_configurations_is_refused(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        """The claim behind --write-noise-floor is checked against the records."""
        with pytest.raises(ValueError) as raised:
            run_main(
                evaluate_module,
                monkeypatch,
                workspace,
                tmp_path / "not-replicas",
                "--runs-b", str(workspace["runs"] / "agent-objects"),
                "--write-noise-floor",
            )

        assert "caption_variant" in str(raised.value)

    def test_a_comparison_of_two_configurations_needs_no_replica_check(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "different"

        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            out,
            "--runs-b", str(workspace["runs"] / "agent-objects"),
        )

        assert (out / "comparison.md").is_file()
        assert not (out / "noise_floor.json").exists()


class TestOneConfigurationPerRun:
    def test_a_run_of_two_configurations_is_refused(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError) as raised:
            run_main(
                evaluate_module,
                monkeypatch,
                workspace,
                tmp_path / "mixed",
                run="agent-mixed",
            )

        assert "9999888877776666" in str(raised.value)

    def test_the_flag_downgrades_it_to_a_warning_and_reports_both_ids(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path, capsys
    ) -> None:
        out = tmp_path / "mixed-allowed"

        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            out,
            "--allow-mixed-config",
            run="agent-mixed",
        )

        captured = capsys.readouterr()
        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))

        assert "mixes 2 configurations" in captured.err
        assert sorted(metrics["config_ids"]) == [
            "0123456789abcdef",
            "9999888877776666",
        ]
        assert "9999888877776666" in captured.out


class TestStatusLine:
    def test_it_says_what_was_scored_and_what_was_left_out(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path, capsys
    ) -> None:
        run_main(evaluate_module, monkeypatch, workspace, tmp_path / "line")

        first = capsys.readouterr().out.splitlines()[0]

        assert first.startswith("agent: 3 recordings")
        assert "(0 optional excluded)" in first
        assert "1 length-truncated turns" in first
        assert "config 0123456789abcdef" in first
        assert first.endswith(", order llm")

    def test_it_ends_with_the_order_reading_that_produced_the_row(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path, capsys
    ) -> None:
        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            tmp_path / "line-anchors",
            "--order-from", "anchors",
        )

        first = capsys.readouterr().out.splitlines()[0]

        assert first.endswith(", order anchors:transitive,both,end")

    def test_a_comparison_without_the_flag_records_no_floor(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "compare"

        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            out,
            "--runs-b", str(workspace["runs"] / "agent-copy"),
        )

        assert (out / "absolute_difference.json").is_file()
        assert not (out / "noise_floor.json").exists()
        assert "—" in (out / "comparison.md").read_text(encoding="utf-8")

    def test_a_supplied_floor_turns_the_verdict_column_into_an_answer(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        floor_dir = tmp_path / "floor"
        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            floor_dir,
            "--runs-b", str(workspace["runs"] / "agent-copy"),
            "--write-noise-floor",
        )
        out = tmp_path / "checked"

        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            out,
            "--runs-b", str(workspace["runs"] / "agent-copy"),
            "--noise-floor", str(floor_dir / "noise_floor.json"),
        )

        text = (out / "comparison.md").read_text(encoding="utf-8")
        assert "| no |" in text or "| yes |" in text
        assert not (out / "noise_floor.json").exists()


class TestTheOrderReadingReachesTheFiles:
    def test_the_default_report_says_the_llm_wrote_the_order_column(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "llm"

        run_main(evaluate_module, monkeypatch, workspace, out)

        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["order_policy"] == {
            "source": "llm",
            "rule": "transitive",
            "edges": "both",
            "margin": "end",
        }
        assert "Order verdict: judged order (written by the LLM)." in (
            out / "results_row.md"
        ).read_text(encoding="utf-8")
        assert (out / "results_row.csv").read_text(encoding="utf-8").rstrip().endswith(
            ",llm"
        )

    def test_the_anchors_reading_is_stamped_on_the_metrics_and_the_tables(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "anchors"

        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            out,
            "--order-from", "anchors",
            "--order-rule", "direct",
            "--order-edges", "early",
            "--order-margin", "start",
        )

        metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["order_policy"] == {
            "source": "anchors",
            "rule": "direct",
            "edges": "early",
            "margin": "start",
        }
        assert "from the anchors (direct, early, margin start)" in (
            out / "results_row.md"
        ).read_text(encoding="utf-8")
        assert (out / "results_row.csv").read_text(encoding="utf-8").rstrip().endswith(
            '"anchors:direct,early,start"'
        )

    def test_a_comparison_is_read_the_same_way_on_both_sides(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        out = tmp_path / "compare-anchors"

        run_main(
            evaluate_module,
            monkeypatch,
            workspace,
            out,
            "--runs-b", str(workspace["runs"] / "agent-copy"),
            "--order-from", "anchors",
        )

        text = (out / "comparison.md").read_text(encoding="utf-8")
        assert "Order verdict: anchors:transitive,both,end" in text
        assert json.loads(
            (out / "metrics.json").read_text(encoding="utf-8")
        )["order_policy"]["source"] == "anchors"

    def test_the_two_readings_score_the_same_records_differently(
        self, evaluate_module, monkeypatch, workspace: dict, tmp_path: Path
    ) -> None:
        """The point of the flag: the anchors of the store hold no order error.

        ``build_store`` writes one hand-made order violation on the error
        recording while anchoring every step on its annotated interval, which
        is an ordered timeline. So the judged order scores one true positive
        of order and the computed order scores none: same records, same
        missing-step column, different order column.
        """
        llm, anchors = tmp_path / "a", tmp_path / "b"

        run_main(evaluate_module, monkeypatch, workspace, llm)
        run_main(
            evaluate_module, monkeypatch, workspace, anchors, "--order-from", "anchors"
        )

        read = [
            json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            for path in (llm, anchors)
        ]
        assert read[0]["category"]["order"]["tp"] == 1
        assert read[1]["category"]["order"]["tp"] == 0
        assert read[0]["category"]["omission"] == read[1]["category"]["omission"]
        assert read[0]["instances"] == read[1]["instances"]

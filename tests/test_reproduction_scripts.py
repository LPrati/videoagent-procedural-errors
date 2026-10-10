"""The small reproduction scripts: order-rule validation, synthetic scoring, report summary."""

import json
import sys
from pathlib import Path

import pytest

from tests._paths import ANNOTATIONS_DIR, DATA_SPLITS, SCRIPTS, load_annotations
from video_agent.eval.order_from_anchors import OrderPolicy
from video_agent.task.annotations import Recording, StepInstance, TaskGraph, load_split
from video_agent.task.verdicts import Execution, Order

sys.path.insert(0, str(SCRIPTS))

import score_synthetic  # noqa: E402
import summarize_reports  # noqa: E402
import validate_order_rule  # noqa: E402

ORDER_TAG = "Order Error"


def instance(step_id, start, end, tags=()):
    return StepInstance(
        recording_id="r",
        step_id=step_id,
        occurrence=0,
        description=f"step {step_id}",
        start_s=start,
        end_s=end,
        tags=frozenset(tags),
    )


def recording(*steps):
    return Recording(recording_id="r", activity_id=1, is_error=False, steps=tuple(steps))


# 1 -> 2 -> 3: step 1 is a prerequisite of 2, and 2 of 3.
GRAPH = TaskGraph(
    activity_id=1,
    prerequisites={2: frozenset({1}), 3: frozenset({2})},
    edges=frozenset({(1, 2), (2, 3)}),
)


class TestOrderViolations:
    def test_an_ordered_recording_has_none(self):
        rec = recording(instance(1, 0, 10), instance(2, 10, 20), instance(3, 20, 30))
        assert score_synthetic.order_violations(rec, GRAPH) == set()

    def test_a_late_prerequisite_flags_both_ends_through_the_chain(self):
        # Step 1 runs last: it comes after 2 (direct) and 3 (transitive).
        rec = recording(instance(2, 0, 10), instance(3, 10, 20), instance(1, 25, 35))
        assert score_synthetic.order_violations(rec, GRAPH) == {1, 2, 3}

    def test_a_missing_prerequisite_is_not_an_order_error(self):
        rec = recording(instance(2, 0, 10), instance(3, 10, 20), instance(1, None, None, ["Missing Step"]))
        assert score_synthetic.order_violations(rec, GRAPH) == set()

    def test_overlapping_steps_are_not_out_of_order(self):
        # Step 1 starts after step 2 starts but before step 2 ends.
        rec = recording(instance(2, 0, 10), instance(1, 5, 12), instance(3, 20, 30))
        assert score_synthetic.order_violations(rec, GRAPH) == set()


class TestOracleVerdicts:
    def test_timed_instances_are_executed_on_their_interval(self):
        rec = recording(instance(1, 0, 10), instance(2, None, None, ["Missing Step"]))
        verdicts = validate_order_rule.oracle_verdicts(rec).steps
        assert verdicts[0].execution is Execution.EXECUTED
        assert (verdicts[0].anchor.start_s, verdicts[0].anchor.end_s) == (0, 10)
        assert verdicts[1].execution is Execution.NOT_EXECUTED

    def test_the_rule_recovers_a_tagged_inversion(self):
        rec = recording(
            instance(2, 0, 10, [ORDER_TAG]),
            instance(3, 10, 20, [ORDER_TAG]),
            instance(1, 25, 35, [ORDER_TAG]),
        )

        class OneRecording:
            recordings = {"r": rec}

            def for_recording(self, recording_id):
                return None, GRAPH

        counts = validate_order_rule.confusion(
            OneRecording(), ["r"], OrderPolicy(source="anchors")
        )
        assert counts == {"tp": 3, "fp": 0, "fn": 0, "tn": 0}
        assert validate_order_rule.prf(counts)["f1"] == 1.0

    def test_undefined_scores_stay_undefined(self):
        scores = validate_order_rule.prf({"tp": 0, "fp": 0, "fn": 0, "tn": 5})
        assert scores == {"precision": None, "recall": None, "f1": None}


@pytest.mark.parametrize(
    ("split", "expected_f1"), [("dev_83.json", 0.922), ("test.json", 0.900)]
)
def test_the_rule_reproduces_the_annotated_order_errors(split, expected_f1):
    annotations = load_annotations()
    ids_error, ids_normal = load_split(DATA_SPLITS / split)
    counts = validate_order_rule.confusion(
        annotations, sorted(set(ids_error) | set(ids_normal)), OrderPolicy(source="anchors")
    )
    assert validate_order_rule.prf(counts)["f1"] == pytest.approx(expected_f1, abs=5e-4)


def metrics(run, source, order, omission, *, eda=0.7, tokens=1000.0):
    def cell(p, r, f1):
        return {"precision": p, "recall": r, "f1": f1}

    return {
        "run_name": run,
        "order_policy": {"source": source},
        "category": {"order": cell(*order), "omission": cell(*omission)},
        "indeterminate": {"omission": 0.25, "order": 0.3},
        "localization": {
            "0-10": {"anchored": 10, "hit_at_half": 0.5},
            "10-20": {"anchored": 30, "hit_at_half": 0.1},
        },
        "eda": {"accuracy": eda},
        "cost": {"mean_tokens_in": tokens},
    }


def write_report(directory: Path, payload: dict) -> Path:
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


class TestSummarizeReports:
    def test_one_run_reads_both_readings(self, tmp_path, capsys):
        judged = write_report(
            tmp_path / "llm", metrics("a", "llm", (0.4, 0.1, 0.16), (0.2, 0.3, 0.24))
        )
        computed = write_report(
            tmp_path / "anchors",
            metrics("a", "anchors", (0.5, 0.4, 0.44), (0.2, 0.3, 0.24), eda=0.75),
        )
        summarize_reports.main(["--judged", str(judged), "--computed", str(computed)])
        out = capsys.readouterr().out
        assert "| Judged order (P / R / F1) | 40.0 / 10.0 / 16.0 |" in out
        assert "| Computed order (P / R / F1) | 50.0 / 40.0 / 44.0 |" in out
        assert "| hit@IoU>=0.5 | 20.0 |" in out  # (5 + 3) / 40
        assert "| EDA (judged / computed) | 70.0 / 75.0 |" in out

    def test_several_runs_are_averaged(self, tmp_path):
        columns = [
            summarize_reports.column(
                metrics("a", "llm", (0.2, 0.2, 0.2), (0.1, 0.1, 0.1), tokens=1000.0),
                metrics("a", "anchors", (0.4, 0.4, 0.4), (0.1, 0.1, 0.1)),
            ),
            summarize_reports.column(
                metrics("b", "llm", (0.4, 0.4, 0.4), (0.3, 0.3, 0.3), tokens=3000.0),
                metrics("b", "anchors", (0.6, 0.6, 0.6), (0.3, 0.3, 0.3)),
            ),
        ]
        mean = summarize_reports.average(columns)
        assert mean["judged_order_f1"] == pytest.approx(0.3)
        assert mean["computed_order_f1"] == pytest.approx(0.5)
        assert mean["tokens_in_thousands"] == pytest.approx(2.0)

    def test_reports_of_two_different_runs_are_refused(self, tmp_path):
        judged = write_report(tmp_path / "llm", metrics("a", "llm", (0, 0, 0), (0, 0, 0)))
        computed = write_report(
            tmp_path / "anchors", metrics("b", "anchors", (0, 0, 0), (0, 0, 0))
        )
        with pytest.raises(SystemExit):
            summarize_reports.main(["--judged", str(judged), "--computed", str(computed)])

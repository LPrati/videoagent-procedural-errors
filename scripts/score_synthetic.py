"""Score the synthetic-memory episodes with the order definition of the task question.

``scripts/synthetic_harness.py`` builds seven cases per normal recording
(``video_agent.eval.synthetic.make_cases``) and runs the agent on each one.
The answer it expects for the order column comes from the case builder, which
reads direct prerequisites only. The task question the agent receives defines
an order error as in the CaptainCook4D annotations instead:

* constraints chain: the whole transitive closure of the prerequisites counts;
* only performed steps take part: a missing step makes no other step a
  violation (it is a missing step, nothing else);
* a step counts as performed after another only if it starts after the other
  one ends;
* both ends of a displaced pair are violations.

This script rebuilds the same cases (same seed) and replaces only the order
column of the expected answer with this definition, read off the case's
annotated times. The execution column and the anchors are unchanged. It then
reports, per case, how many steps received the expected classification, and
the ratio eta for the shuffled memory (correct classifications with the
shuffled memory over those with the unaltered one).

Usage:
    uv run python scripts/score_synthetic.py --runs RUNS_DIR/synthetic \\
        --out reports/synthetic

With several ``--runs`` directories, an episode whose record is an error in
one directory is taken from the next one (to merge a re-run of a failed
episode).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from video_agent.eval.report import read_verdicts
from video_agent.eval.synthetic import (
    CASE_NAMES,
    DEFAULT_SEED,
    Case,
    count_matching,
    eta,
    make_cases,
)
from video_agent.task.annotations import AnnotationSet, Recording, TaskGraph
from video_agent.task.verdicts import Execution, Order, RecordingVerdicts

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "data" / "captaincook4d" / "annotations"

CASE_TITLES = {
    "normal": "Nothing altered",
    "omission": "One step removed",
    "linked_swap": "Two steps swapped, one a prerequisite of the other",
    "unlinked_swap": "Two steps swapped, no prerequisite between them",
    "omission_and_swap": "One step removed and two linked steps swapped",
    "repeated_step": "One step repeated",
    "shuffled": "Whole memory shuffled",
}


def first_intervals(recording: Recording) -> dict[int, tuple[float, float]]:
    """(start, end) of the first timed occurrence of each step."""
    intervals: dict[int, tuple[float, float]] = {}
    for step in sorted(
        recording.steps, key=lambda s: (s.start_s is None, s.start_s or 0.0)
    ):
        if (
            step.step_id not in intervals
            and step.start_s is not None
            and step.end_s is not None
        ):
            intervals[step.step_id] = (step.start_s, step.end_s)
    return intervals


def order_violations(recording: Recording, graph: TaskGraph) -> set[int]:
    """Steps out of order under the annotation's definition (see module docstring)."""
    intervals = first_intervals(recording)
    violations: set[int] = set()
    for dependent, (_, dependent_end) in intervals.items():
        if dependent not in graph.prerequisites:
            continue
        for prerequisite in graph.ancestors(dependent):
            if prerequisite in intervals and intervals[prerequisite][0] > dependent_end:
                violations.update({prerequisite, dependent})
    return violations


def expected_answer(case: Case, graph: TaskGraph) -> RecordingVerdicts:
    """The case's expected answer with the order column of the task question."""
    violations = order_violations(case.recording, graph)
    steps = []
    for verdict in case.expected.steps:
        if verdict.execution is Execution.EXECUTED:
            verdict = replace(
                verdict,
                order=Order.VIOLATION if verdict.step_id in violations else Order.OK,
                prerequisite_step_id=None,
                prerequisite_anchor=None,
            )
        steps.append(verdict)
    return replace(case.expected, steps=tuple(steps))


def load_records(directories: Iterable[Path]) -> dict[str, dict]:
    """``<recording>__<case>`` -> run record, the first non-error one wins."""
    records: dict[str, dict] = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*__*/run.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            key = path.parent.name
            if key not in records or records[key].get("status") == "error":
                records[key] = record
    return records


def score(
    records: dict[str, dict], annotations: AnnotationSet, seed: int
) -> list[dict]:
    """One row per episode: matches against the expected answer, or eta."""
    rows: list[dict] = []
    recording_ids = sorted({key.split("__")[0] for key in records})
    for recording_id in recording_ids:
        sop, graph = annotations.for_recording(recording_id)
        cases = {
            case.name: case
            for case in make_cases(
                annotations.recordings[recording_id], sop, graph, seed=seed
            )
        }
        expected = {
            name: expected_answer(case, graph)
            for name, case in cases.items()
            if case.expected is not None
        }
        normal_matches = None
        for name in CASE_NAMES:
            row: dict = {"recording_id": recording_id, "case": name}
            record = records.get(f"{recording_id}__{name}")
            if record is None or record.get("status") == "error":
                row["status"] = "missing" if record is None else "error"
                rows.append(row)
                continue
            row["status"] = record.get("status")
            got = read_verdicts(record)
            reference = expected["normal"] if name == "shuffled" else expected[name]
            matched, total = count_matching(reference, got)
            row.update(matched=matched, total=total)
            if name == "normal":
                normal_matches = matched
            if name != "shuffled":
                answers = {v.step_id: v for v in got.steps}
                wanted = [v.step_id for v in reference.steps if v.order is Order.VIOLATION]
                row["expected_violations"] = wanted
                row["found_violations"] = [
                    step_id
                    for step_id in wanted
                    if step_id in answers and answers[step_id].order is Order.VIOLATION
                ]
            rows.append(row)
        for row in rows:
            if (
                row["recording_id"] == recording_id
                and row["case"] == "shuffled"
                and "matched" in row
                and normal_matches is not None
            ):
                row["eta"] = eta(row["matched"], normal_matches)
    return rows


def summarize(rows: Sequence[dict]) -> dict[str, dict]:
    """Totals per case over all recordings."""
    summary: dict[str, dict] = {}
    for name in CASE_NAMES:
        scored = [row for row in rows if row["case"] == name and "matched" in row]
        missing = [row["recording_id"] for row in rows if row["case"] == name and "matched" not in row]
        if name == "shuffled":
            etas = [row["eta"] for row in scored if row.get("eta") is not None]
            summary[name] = {
                "eta_min": min(etas) if etas else None,
                "eta_max": max(etas) if etas else None,
                "eta_mean": sum(etas) / len(etas) if etas else None,
                "without_record": missing,
            }
        else:
            summary[name] = {
                "matched": sum(row["matched"] for row in scored),
                "total": sum(row["total"] for row in scored),
                "expected_violations": sum(len(row["expected_violations"]) for row in scored),
                "found_violations": sum(len(row["found_violations"]) for row in scored),
                "without_record": missing,
            }
    return summary


def render(summary: dict[str, dict]) -> str:
    lines = [
        "| Case | Steps with the expected classification | Expected violations found |",
        "|---|---|---|",
    ]
    for name in CASE_NAMES:
        cell = summary[name]
        if name == "shuffled":
            if cell["eta_mean"] is None:
                result, violations = "eta undefined", "—"
            else:
                result = (
                    f"eta {cell['eta_min']:.2f} to {cell['eta_max']:.2f} "
                    f"(mean {cell['eta_mean']:.2f})"
                )
                violations = "—"
        else:
            share = 100 * cell["matched"] / cell["total"] if cell["total"] else 0.0
            result = f"{cell['matched']} of {cell['total']} ({share:.0f}%)"
            violations = (
                f"{cell['found_violations']} of {cell['expected_violations']}"
                if cell["expected_violations"]
                else "—"
            )
        if cell["without_record"]:
            result += f" (no usable record: {', '.join(cell['without_record'])})"
        lines.append(f"| {CASE_TITLES[name]} | {result} | {violations} |")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="seed the harness built the cases with",
    )
    parser.add_argument("--out", type=Path, help="optional directory for JSON and Markdown")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    annotations = AnnotationSet.load(args.annotations_dir)
    rows = score(load_records(args.runs), annotations, args.seed)
    summary = summarize(rows)
    table = render(summary)
    print(table)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "synthetic_scores.json").write_text(
            json.dumps({"seed": args.seed, "summary": summary, "episodes": rows}, indent=1)
            + "\n",
            encoding="utf-8",
        )
        (args.out / "synthetic_scores.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

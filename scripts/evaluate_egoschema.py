"""Score an EgoSchema run and write its table, beside the published numbers.

Reads the records ``run_egoschema.py`` wrote, scores them with a fixed
rule — an invalid answer counts as wrong, and the invalid rate is reported by
cause — and writes ``metrics.json`` and ``table.md``. The table carries the
published rows of the same 500-question subset, because the accuracy of the
reimplementation is read as its distance to them.

With ``--compare`` it also writes the paired comparison of two runs over the
questions both answered: the difference with a bootstrap interval, the exact
McNemar test, and how often the two agreed.

Nothing here decides anything and nothing here is fitted: a question with no
record is reported as such rather than dropped, and an undefined measure
stays undefined. Reads only — no model, no GPU, no network.

Usage:
    uv run python scripts/evaluate_egoschema.py \\
        --run-dir RUNS_DIR/NAME \\
        --answers EGOSCHEMA_DIR/subset_answers.json \\
        --out REPORT_DIR
"""

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from video_agent.agent.react import UNKNOWN_TOOL_ERROR
from video_agent.agent.tools import MAIN_TOOL_NAMES
from video_agent.eval.bootstrap import discordant_counts, mcnemar, paired_difference_ci
from video_agent.eval.report import count_length_turns
from video_agent.task.run import RUN_FILE

Z_95 = 1.959963984540054
"""Two-sided normal quantile of a 95% interval, as Wilson's formula needs it."""

NO_RECORD = "no_record"
"""``status`` of a question of the subset the run never wrote a record for."""

INVALID_ACTION = "invalid_action"
"""Key of the row every action that named no tool is counted under.

A model can write the ``Action:`` line as free text and miss the tool name —
a keyword argument, a whole paragraph of deliberation. Such a step is an
invalid action and is counted as one: guessing which tool it meant would put
calls the agent never made into the table, and a row per raw string would
make the table a copy of the transcripts.
"""

INVALID_ACTION_LABEL = "(invalid action)"
"""How the invalid-action row is named in the Markdown table."""

REFERENCE_ROWS: tuple[tuple[str, str, float], ...] = (
    ("VideoAgent (Fan et al., 2024)", "GPT-4", 62.8),
    ("LLoVi", "—", 51.8),
    ("Video-LLaVA", "—", 36.8),
    ("chance", "—", 20.0),
)
"""Published accuracies on the same 500-question subset.

Fixed in the code and not read from a file: they are the yardstick every
table is printed against, and a number typed again per report is a number
that drifts.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="<out>/<run-name> of the run"
    )
    parser.add_argument(
        "--answers", type=Path, required=True, help="subset_answers.json"
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="<out>/<run-name> of a second run; adds the paired comparison "
        "over the questions both answered",
    )
    parser.add_argument("--out", type=Path, required=True, help="report directory")
    parser.add_argument(
        "--bootstrap", type=int, default=2000, help="resamples of the question bootstrap"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def wilson_interval(successes: int, n: int, z: float = Z_95) -> dict:
    """Wilson score interval of a proportion.

    Preferred to the normal approximation because it stays inside [0, 1] and
    keeps its coverage near the ends, e.g. for a low accuracy over a small
    number of questions. ``n == 0`` has no proportion at all, and an
    undefined interval is reported as such rather than as zero.
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n, got {successes}, {n}")
    if n == 0:
        return {"point": None, "low": None, "high": None, "n": 0, "successes": 0}
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    spread = (
        z
        / denominator
        * math.sqrt(proportion * (1.0 - proportion) / n + z * z / (4.0 * n * n))
    )
    return {
        "point": proportion,
        "low": max(0.0, center - spread),
        "high": min(1.0, center + spread),
        "n": n,
        "successes": successes,
    }


def load_answers(path: Path) -> dict[str, int]:
    """``subset_answers.json`` as ``q_uid -> label``: the set that is scored."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object of answers")
    return {str(key): int(value) for key, value in data.items()}


def load_records(run_dir: Path) -> dict[str, dict]:
    """Every ``run.json`` under a run directory, keyed by ``q_uid``."""
    run_dir = Path(run_dir)
    records: dict[str, dict] = {}
    if not run_dir.is_dir():
        return records
    for path in sorted(run_dir.glob(f"*/{RUN_FILE}")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[str(record.get("q_uid") or path.parent.name)] = record
    return records


def prediction_of(record: Mapping | None) -> int | None:
    """The label a record holds, or ``None`` for an invalid or absent one."""
    if record is None:
        return None
    prediction = record.get("prediction")
    return None if prediction is None else int(prediction)


def is_correct(record: Mapping | None, truth: int) -> bool:
    """Whether a question was answered right, an invalid answer being wrong.

    The scoring rule, applied in one place: everything that is not the right
    label — a missing ``Final Answer``, the iteration ceiling, a crash, a
    record that was never written — is a wrong answer, and the invalid rate
    is reported beside the accuracy rather than deducted from its
    denominator.
    """
    return prediction_of(record) == truth


def status_of(record: Mapping | None) -> str:
    """The status of a record, or ``no_record`` when the run wrote none."""
    if record is None:
        return NO_RECORD
    return str(record.get("status") or "unknown")


def _quantile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated quantile of a sorted copy of ``values``."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _steps(record: Mapping) -> list[dict]:
    transcript = record.get("transcript") or {}
    return list(transcript.get("steps") or ())


def is_invalid_action(action: Mapping) -> bool:
    """Whether a step's action named no tool of the orchestrator's toolkit.

    Two signals, and either is enough: the executor's own verdict, written on
    the call as ``error`` when it found no tool of that name, and the name
    itself, for a record whose executor did not say so.
    """
    if action.get("error") == UNKNOWN_TOOL_ERROR:
        return True
    return str(action.get("name") or "") not in MAIN_TOOL_NAMES


def tool_calls_of(record: Mapping) -> dict[str, int]:
    """How often this episode called each tool, by registered name.

    Only the orchestrator's own loop: a sub-agent's calls are calls of the
    database tools inside ``object_memory_querying`` and counting them here
    would say the agent used a tool it never chose.

    Anything that is not one of the registered names is counted under
    ``invalid_action`` rather than under itself, so the vocabulary of this
    dict is closed and a report built on it cannot grow a row per malformed
    line the model wrote.
    """
    counts: dict[str, int] = {}
    for step in _steps(record):
        action = step.get("action") or {}
        name = action.get("name")
        if not name:
            continue
        key = INVALID_ACTION if is_invalid_action(action) else str(name)
        counts[key] = counts.get(key, 0) + 1
    return counts


def process_measures(records: Sequence[Mapping]) -> dict:
    """What the episodes cost and how they got there.

    Read over the records that exist: a question with no record has no
    iteration count, and averaging a zero in its place would say the agent
    answered it cheaply.
    """
    n = len(records)
    iterations = [float(len(_steps(record))) for record in records]
    seconds = [float(record.get("seconds") or 0.0) for record in records]
    # ``usage_total`` counts the object sub-agent's episodes too; ``usage`` is
    # the fallback for a record that lacks the field.
    usages = [record.get("usage_total") or record.get("usage") or {} for record in records]
    tokens_in = [float(usage.get("tokens_in", 0)) for usage in usages]
    tokens_out = [float(usage.get("tokens_out", 0)) for usage in usages]

    per_tool: dict[str, dict[str, float]] = {}
    totals: dict[str, int] = {}
    users: dict[str, int] = {}
    for record in records:
        for name, count in tool_calls_of(record).items():
            totals[name] = totals.get(name, 0) + count
            users[name] = users.get(name, 0) + 1
    for name in (*MAIN_TOOL_NAMES, INVALID_ACTION):
        per_tool[name] = {
            "calls": totals.get(name, 0),
            "mean_per_question": totals.get(name, 0) / n if n else None,
            "questions": users.get(name, 0),
            "questions_pct": 100.0 * users.get(name, 0) / n if n else None,
        }

    return {
        "n_records": n,
        "iterations": {
            "median": _quantile(iterations, 0.5),
            "p90": _quantile(iterations, 0.9),
            "max": max(iterations) if iterations else None,
        },
        "tools": per_tool,
        "n_questions_with_invalid_action": users.get(INVALID_ACTION, 0),
        "tokens_in": {
            "mean": sum(tokens_in) / n if n else None,
            "total": sum(tokens_in),
        },
        "tokens_out": {
            "mean": sum(tokens_out) / n if n else None,
            "total": sum(tokens_out),
        },
        "seconds": {"median": _quantile(seconds, 0.5), "total": sum(seconds)},
        "length_turns": sum(count_length_turns(record) for record in records),
    }


def evaluate_run(run_dir: Path, answers: Mapping[str, int]) -> dict:
    """Score one run over the answered subset and return the whole measure."""
    records = load_records(run_dir)
    q_uids = sorted(answers)
    present = [q_uid for q_uid in q_uids if q_uid in records]

    status_counts: dict[str, int] = {}
    for q_uid in q_uids:
        status = status_of(records.get(q_uid))
        status_counts[status] = status_counts.get(status, 0) + 1

    correct = sum(
        1 for q_uid in q_uids if is_correct(records.get(q_uid), answers[q_uid])
    )
    valid = [q_uid for q_uid in q_uids if prediction_of(records.get(q_uid)) is not None]
    correct_valid = sum(
        1 for q_uid in valid if is_correct(records[q_uid], answers[q_uid])
    )
    invalid_by_status: dict[str, int] = {}
    for q_uid in q_uids:
        if prediction_of(records.get(q_uid)) is None:
            status = status_of(records.get(q_uid))
            invalid_by_status[status] = invalid_by_status.get(status, 0) + 1

    config_ids = sorted(
        {
            str((records[q_uid].get("config") or {}).get("config_id"))
            for q_uid in present
            if (records[q_uid].get("config") or {}).get("config_id")
        }
    )
    models = sorted(
        {
            str((records[q_uid].get("config") or {}).get("model"))
            for q_uid in present
            if (records[q_uid].get("config") or {}).get("model")
        }
    )
    extra = sorted(set(records) - set(answers))
    return {
        "run_name": Path(run_dir).name,
        "run_dir": str(run_dir),
        "n_answers": len(q_uids),
        "n_records": len(present),
        "n_without_record": len(q_uids) - len(present),
        "n_records_outside_subset": len(extra),
        "accuracy": wilson_interval(correct, len(q_uids)),
        "accuracy_valid_only": wilson_interval(correct_valid, len(valid)),
        "n_valid": len(valid),
        "n_invalid": len(q_uids) - len(valid),
        "status_counts": dict(sorted(status_counts.items())),
        "invalid_by_status": dict(sorted(invalid_by_status.items())),
        "process": process_measures([records[q_uid] for q_uid in present]),
        "config_ids": config_ids,
        "orchestrator": ", ".join(models) or None,
        "reference": [
            {"system": system, "orchestrator": model, "accuracy_pct": value}
            for system, model, value in REFERENCE_ROWS
        ],
    }


def compare_runs(
    run_dir_a: Path,
    run_dir_b: Path,
    answers: Mapping[str, int],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired comparison of two runs over the questions both have records for.

    Paired because the two runs saw the same questions: the bootstrap draws
    one resample of question ids and recomputes both accuracies on it, and
    McNemar reads only the questions the two disagreed on — which is where
    the whole of the evidence for a difference is.
    """
    records_a = load_records(run_dir_a)
    records_b = load_records(run_dir_b)
    common = sorted(set(records_a) & set(records_b) & set(answers))

    right_a = {q_uid: is_correct(records_a[q_uid], answers[q_uid]) for q_uid in common}
    right_b = {q_uid: is_correct(records_b[q_uid], answers[q_uid]) for q_uid in common}

    def accuracy(right: Mapping[str, bool]):
        def statistic(ids: Sequence[str]) -> float | None:
            ids = list(ids)
            if not ids:
                return None
            return sum(right[q_uid] for q_uid in ids) / len(ids)

        return statistic

    difference = paired_difference_ci(
        accuracy(right_a), accuracy(right_b), common, n=n_bootstrap, seed=seed
    )
    b, c = discordant_counts(
        [right_a[q_uid] for q_uid in common],
        [right_b[q_uid] for q_uid in common],
        [True] * len(common),
    )
    agreement = sum(
        prediction_of(records_a[q_uid]) == prediction_of(records_b[q_uid])
        for q_uid in common
    )
    return {
        "run_a": Path(run_dir_a).name,
        "run_b": Path(run_dir_b).name,
        "n_common": len(common),
        "accuracy_a": wilson_interval(sum(right_a.values()), len(common)),
        "accuracy_b": wilson_interval(sum(right_b.values()), len(common)),
        "difference": difference.to_dict(),
        "mcnemar": {"b": b, "c": c, "p": mcnemar(b, c)},
        "agreement": agreement / len(common) if common else None,
        "n_agree": agreement,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def _cell(value: object) -> str:
    """``value`` as one Markdown table cell: one line, no unescaped pipe.

    Every cell that comes from a record goes through this. A run name, a
    status, a model id are all free text written elsewhere, and a newline or a
    pipe in one of them does not make an ugly cell — it makes the row stop
    being a row, and the rest of the table stops parsing.
    """
    text = " ".join(str(value).split())
    return text.replace("|", "\\|") or "—"


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.1f}"


def _number(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_table(evaluation: Mapping, comparison: Mapping | None = None) -> str:
    """The Markdown report: the run, the published rows, and the process."""
    accuracy = evaluation["accuracy"]
    valid = evaluation["accuracy_valid_only"]
    process = evaluation["process"]
    name = evaluation["run_name"]

    lines = [
        f"# EgoSchema — {_cell(name)}",
        "",
        "## Accuracy on the 500-question subset (invalid answers count as wrong)",
        "",
        "| System | Orchestrator | Questions | Accuracy [95% CI] | Invalid |",
        "|---|---|---|---|---|",
    ]
    for row in evaluation["reference"]:
        lines.append(
            f"| {row['system']}, published | {row['orchestrator']} | 500 | "
            f"{row['accuracy_pct']:.1f} | — |"
        )
    invalid_pct = (
        100.0 * evaluation["n_invalid"] / evaluation["n_answers"]
        if evaluation["n_answers"]
        else None
    )
    lines.append(
        f"| `{_cell(name)}` | {_cell(evaluation['orchestrator'] or '—')} | "
        f"{evaluation['n_answers']} | {_pct(accuracy['point'])} "
        f"[{_pct(accuracy['low'])}, {_pct(accuracy['high'])}] | "
        f"{evaluation['n_invalid']} ({_number(invalid_pct)}%) |"
    )
    lines += [
        "",
        f"Accuracy over valid answers only: {_pct(valid['point'])} "
        f"(n = {valid['n']}).",
        f"Questions without a record: {evaluation['n_without_record']}; "
        f"records outside the subset: {evaluation['n_records_outside_subset']}.",
        "Configurations in the records: "
        f"{_cell(', '.join(evaluation['config_ids']) or '—')}.",
        "",
        "## Questions by status",
        "",
        "| Status | Questions | Invalid |",
        "|---|---|---|",
    ]
    for status, count in evaluation["status_counts"].items():
        lines.append(
            f"| {_cell(status)} | {count} | "
            f"{evaluation['invalid_by_status'].get(status, 0)} |"
        )
    lines += [
        "",
        "## Process measures",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Records read | {process['n_records']} |",
        f"| Iterations (median / p90 / max) | "
        f"{_number(process['iterations']['median'])} / "
        f"{_number(process['iterations']['p90'])} / "
        f"{_number(process['iterations']['max'], 0)} |",
        f"| Input tokens (mean / total) | "
        f"{_number(process['tokens_in']['mean'], 0)} / "
        f"{_number(process['tokens_in']['total'], 0)} |",
        f"| Output tokens (mean / total) | "
        f"{_number(process['tokens_out']['mean'], 0)} / "
        f"{_number(process['tokens_out']['total'], 0)} |",
        f"| Seconds (median / total) | "
        f"{_number(process['seconds']['median'])} / "
        f"{_number(process['seconds']['total'], 0)} |",
        f"| Turns truncated by the token cap | {process['length_turns']} |",
        "",
        "| Tool | Calls per question (mean) | Questions that called it (%) |",
        "|---|---|---|",
    ]
    for tool, measures in process["tools"].items():
        label = INVALID_ACTION_LABEL if tool == INVALID_ACTION else tool
        lines.append(
            f"| {_cell(label)} | {_number(measures['mean_per_question'], 2)} | "
            f"{_number(measures['questions_pct'])} |"
        )
    lines.append("")
    lines.append(
        f"The row \"{INVALID_ACTION_LABEL}\" counts the steps whose "
        "`Action:` line named none of the registered tools; "
        "questions with at least one: "
        f"{process['n_questions_with_invalid_action']}."
    )

    if comparison is not None:
        difference = comparison["difference"]
        test = comparison["mcnemar"]
        lines += [
            "",
            f"## Paired comparison — {_cell(comparison['run_a'])} vs "
            f"{_cell(comparison['run_b'])} ({comparison['n_common']} "
            "common questions)",
            "",
            "| Measure | Value |",
            "|---|---|",
            f"| Accuracy {_cell(comparison['run_a'])} | "
            f"{_pct(comparison['accuracy_a']['point'])} |",
            f"| Accuracy {_cell(comparison['run_b'])} | "
            f"{_pct(comparison['accuracy_b']['point'])} |",
            f"| Δ (A − B) [95% bootstrap CI, "
            f"{comparison['n_bootstrap']} resamples, seed "
            f"{comparison['seed']}] | {_pct(difference['point'])} "
            f"[{_pct(difference['low'])}, {_pct(difference['high'])}] |",
            f"| Exact McNemar (b, c, p) | {test['b']}, {test['c']}, "
            f"{test['p']:.4f} |",
            f"| Agreement (same option) | "
            f"{_pct(comparison['agreement'])}% ({comparison['n_agree']}/"
            f"{comparison['n_common']}) |",
        ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    answers = load_answers(args.answers)
    evaluation = evaluate_run(args.run_dir, answers)
    comparison = None
    if args.compare is not None:
        comparison = compare_runs(
            args.run_dir,
            args.compare,
            answers,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        evaluation["comparison"] = comparison

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    table_path = out / "table.md"
    metrics_path.write_text(
        json.dumps(evaluation, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    table_path.write_text(
        render_table(evaluation, comparison) + "\n", encoding="utf-8"
    )
    print(render_table(evaluation, comparison), flush=True)
    print(f"wrote {metrics_path}", flush=True)
    print(f"wrote {table_path}", flush=True)


if __name__ == "__main__":
    main()

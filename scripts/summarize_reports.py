"""Print one system's column of the results tables from its evaluation reports.

``scripts/evaluate.py`` writes one report directory per run and per order
reading. A column of the main results tables (judged order, computed order,
missing steps, unclassified steps, hit@IoU>=0.5, EDA, input tokens) needs
both readings of the same run: the judged order comes from the default
reading (``--order-from llm``) and the computed order from
``--order-from anchors``. The missing-step metrics, the localization and the
cost do not depend on the reading and are taken from the judged-order report.

With several runs of the same configuration (for example the three runs of
the agent with action captions), pass one report directory per run to each
flag, in the same order: every value is then the arithmetic mean of the
per-run values, which is how repeated runs are reported.

Usage:
    uv run python scripts/summarize_reports.py \\
        --judged reports/agent-abst_ctx/llm --computed reports/agent-abst_ctx/anchors
    uv run python scripts/summarize_reports.py \\
        --judged reports/action-s1/llm reports/action-s2/llm reports/action-s3/llm \\
        --computed reports/action-s1/anchors reports/action-s2/anchors reports/action-s3/anchors
"""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Sequence

METRICS_FILE = "metrics.json"


def load_report(directory: Path) -> dict:
    """The ``metrics.json`` of one report directory."""
    return json.loads((Path(directory) / METRICS_FILE).read_text(encoding="utf-8"))


def prf(report: dict, category: str) -> tuple[float | None, float | None, float | None]:
    """Precision, recall and F1 of one category, as fractions."""
    cell = report["category"][category]
    return cell["precision"], cell["recall"], cell["f1"]


def hit_at_half(report: dict) -> float | None:
    """hit@IoU>=0.5 over every anchored step, pooled across the duration bins."""
    bins = report["localization"].values()
    anchored = sum(cell["anchored"] for cell in bins)
    if not anchored:
        return None
    hits = sum(cell["hit_at_half"] * cell["anchored"] for cell in bins if cell["anchored"])
    return hits / anchored


def column(judged: dict, computed: dict) -> dict[str, float | None]:
    """The values of one run, keyed by row name, as fractions or counts."""
    judged_p, judged_r, judged_f1 = prf(judged, "order")
    computed_p, computed_r, computed_f1 = prf(computed, "order")
    missing_p, missing_r, missing_f1 = prf(judged, "omission")
    return {
        "judged_order_p": judged_p,
        "judged_order_r": judged_r,
        "judged_order_f1": judged_f1,
        "computed_order_p": computed_p,
        "computed_order_r": computed_r,
        "computed_order_f1": computed_f1,
        "missing_p": missing_p,
        "missing_r": missing_r,
        "missing_f1": missing_f1,
        "unclassified": judged["indeterminate"]["omission"],
        "hit_iou_0_5": hit_at_half(judged),
        "eda_judged": judged["eda"]["accuracy"],
        "eda_computed": computed["eda"]["accuracy"],
        "tokens_in_thousands": judged["cost"]["mean_tokens_in"] / 1000,
    }


def average(columns: Sequence[dict[str, float | None]]) -> dict[str, float | None]:
    """Mean of each value across runs; undefined if any run left it undefined."""
    out: dict[str, float | None] = {}
    for key in columns[0]:
        values = [col[key] for col in columns]
        out[key] = None if any(value is None for value in values) else mean(values)
    return out


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}"


def render(values: dict[str, float | None], runs: int) -> str:
    """The column as a small Markdown table, percentages with one decimal."""
    rows = [
        ("Judged order (P / R / F1)", " / ".join(
            _pct(values[k]) for k in ("judged_order_p", "judged_order_r", "judged_order_f1"))),
        ("Computed order (P / R / F1)", " / ".join(
            _pct(values[k]) for k in ("computed_order_p", "computed_order_r", "computed_order_f1"))),
        ("Missing steps (P / R / F1)", " / ".join(
            _pct(values[k]) for k in ("missing_p", "missing_r", "missing_f1"))),
        ("Unclassified", _pct(values["unclassified"])),
        ("hit@IoU>=0.5", _pct(values["hit_iou_0_5"])),
        ("EDA (judged / computed)", f"{_pct(values['eda_judged'])} / {_pct(values['eda_computed'])}"),
        ("Input tokens per recording (thousands)", f"{values['tokens_in_thousands']:.0f}"),
    ]
    header = "mean of %d runs" % runs if runs > 1 else "one run"
    lines = [f"| Metric | Value ({header}) |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--judged", type=Path, nargs="+", required=True,
        help="report directories written with --order-from llm (the default)",
    )
    parser.add_argument(
        "--computed", type=Path, nargs="+", required=True,
        help="report directories of the same runs written with --order-from anchors",
    )
    parser.add_argument("--json", type=Path, help="optional JSON file for the values")
    args = parser.parse_args(argv)
    if len(args.judged) != len(args.computed):
        parser.error("--judged and --computed need one directory per run each")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    columns = []
    for judged_dir, computed_dir in zip(args.judged, args.computed):
        judged, computed = load_report(judged_dir), load_report(computed_dir)
        if judged["run_name"] != computed["run_name"]:
            raise SystemExit(
                f"{judged_dir} and {computed_dir} score different runs "
                f"({judged['run_name']} and {computed['run_name']})"
            )
        if computed["order_policy"]["source"] != "anchors":
            raise SystemExit(f"{computed_dir} was not written with --order-from anchors")
        if judged["order_policy"]["source"] != "llm":
            raise SystemExit(f"{judged_dir} was not written with --order-from llm")
        columns.append(column(judged, computed))
    values = average(columns)
    print(render(values, len(columns)))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(values, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

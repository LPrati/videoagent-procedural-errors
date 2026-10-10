"""Score a run against the annotations and write the report tables.

Reads the run records written by ``scripts/run_task.py`` and writes a report
directory:

* ``metrics.json``: every evaluation quantity, with bootstrap intervals;
* ``results_row.md`` / ``results_row.csv``: the main results row (per-category
  precision/recall/F1 for order errors and missing steps, pairwise order
  accuracy, unclassified rates, run statuses and cost);
* ``literature_row.md``: the segment-level EDA row used to compare with the
  literature, beside the "all normal" baseline;
* ``localization.md`` / ``localization.csv``: anchor IoU by step duration.

A recording with no record is scored as unclassified, an undefined metric
stays undefined, and the report says how many records were missing. Reads
only: no model, no GPU, no network. Relative paths are resolved from the
repository root.

Usage:
    uv run python scripts/evaluate.py --runs runs/agent \\
        --split data_splits/dev_83.json --subset all --out reports/agent

Order readings
--------------
The order verdict can be read two ways, selected with ``--order-from``:

* ``llm`` (default), the *judged order*: the ``order`` column the
  orchestrator wrote.
* ``anchors``, the *computed order*: the column is derived by the program
  from the intervals the run localized. A step is a violation when one of
  its prerequisites (the whole transitive closure by default,
  ``--order-rule``) is anchored after it, and both ends of the displacement
  are flagged (``--order-edges``). ``--order-margin end`` requires the
  prerequisite to start after the step's anchor *ends*, so two overlapping
  anchors are not an error. A prerequisite with no anchor marks nobody: a
  missing step is scored as a missing step, not as an order error of every
  step that needed it.

The reading applies to both runs of a comparison and is recorded in
``metrics.json``, in the CSV column ``order_source`` and in the footer of
``results_row.md``.

Comparing two runs
------------------
``--runs-b`` adds a paired comparison over the same recordings
(``comparison.md``, ``absolute_difference.json``): bootstrap intervals of
the difference in each rate, on shared resamples, and exact McNemar tests.

Noise floor (optional)
----------------------
Between two executions of the *same* configuration, the disagreement is the
noise floor; between two different configurations it is the difference being
measured. The floor of a metric is the larger of |A - B| and half the width
of the 95% interval of the paired difference. ``--write-noise-floor``
declares the two runs replicas, refuses to run if their records carry
different configurations, and writes ``noise_floor.json`` and
``noise_floor_detail.json``:

    # 1. the floor: one configuration, two seeds, same recordings
    uv run python scripts/evaluate.py --runs runs/agent-seed1 \\
        --runs-b runs/agent-seed2 --split data_splits/dev_83.json \\
        --out reports/floor --write-noise-floor

    # 2. a comparison, checked against that floor
    uv run python scripts/evaluate.py --runs runs/variant \\
        --runs-b runs/agent-seed1 --split data_splits/dev_83.json \\
        --out reports/variant-vs-agent --noise-floor reports/floor/noise_floor.json

Without ``--noise-floor``, ``comparison.md`` still reports the differences
and their intervals, but its "exceeds the floor" column stays an em dash: no
difference is called detected without a floor to clear.
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

from video_agent.eval.order_from_anchors import OrderPolicy
from video_agent.eval.report import (
    Evaluation,
    compare_runs,
    evaluate_run,
    noise_floor_detail,
    noise_floor_values,
    render_comparison,
    render_localization,
    render_localization_csv,
    render_literature_row,
    render_results_row,
    render_results_row_csv,
)
from video_agent.task.annotations import AnnotationSet, load_split
from video_agent.task.run import RunStore

REPO_ROOT = Path(__file__).resolve().parents[1]
"""Repository root; relative paths resolve from here, not from the cwd."""

DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "data" / "captaincook4d" / "annotations"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs", type=Path, required=True, help="RUNS_DIR/NAME of the run to score"
    )
    parser.add_argument(
        "--runs-b",
        type=Path,
        help="RUNS_DIR/NAME of a second run; adds the paired comparison",
    )
    parser.add_argument(
        "--split",
        type=Path,
        required=True,
        help="split file listing the recordings (e.g. data_splits/dev_83.json)",
    )
    parser.add_argument(
        "--subset",
        choices=["error", "normal", "all"],
        default="all",
        help="which half of the split to score",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_ANNOTATIONS_DIR,
        help="CaptainCook4D annotations directory "
        "(default: data/captaincook4d/annotations)",
    )
    parser.add_argument("--out", type=Path, required=True, help="report directory")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="resamples of the recording bootstrap",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-floor",
        type=Path,
        help="JSON {metric: value} written by a previous comparison of two "
        "replicas of one configuration; the floor the differences must clear",
    )
    parser.add_argument(
        "--write-noise-floor",
        action="store_true",
        help="declare that --runs and --runs-b are two executions of the SAME "
        "configuration, so their disagreement is the noise floor: checks that "
        "the two configurations match and writes noise_floor.json plus "
        "noise_floor_detail.json",
    )
    parser.add_argument(
        "--order-from",
        choices=["llm", "anchors"],
        default="llm",
        help="where the order verdict comes from: the column the orchestrator "
        "wrote (judged order, default) or a rule applied to the anchors of the "
        "same record (computed order)",
    )
    parser.add_argument(
        "--order-rule",
        choices=["direct", "transitive"],
        default="transitive",
        help="with --order-from anchors: compare against the direct "
        "prerequisites only, or the whole closure (default)",
    )
    parser.add_argument(
        "--order-edges",
        choices=["early", "both"],
        default="both",
        help="with --order-from anchors: flag only the step that ran too soon, "
        "or both ends of the displacement (default)",
    )
    parser.add_argument(
        "--order-margin",
        choices=["start", "end"],
        default="end",
        help="with --order-from anchors: a prerequisite offends when it starts "
        "after the step's anchor ends (default) or after it starts",
    )
    parser.add_argument(
        "--allow-mixed-config",
        action="store_true",
        help="score records of different configurations as one row anyway; "
        "without it a mixed run is refused",
    )
    args = parser.parse_args(argv)
    if args.runs_b is None and (args.noise_floor or args.write_noise_floor):
        parser.error("--noise-floor and --write-noise-floor need --runs-b")
    return args


def order_policy(args: argparse.Namespace) -> OrderPolicy:
    """Build the order reading the four ``--order-*`` flags describe."""
    return OrderPolicy(
        source=args.order_from,
        rule=args.order_rule,
        edges=args.order_edges,
        margin=args.order_margin,
    )


def load_noise_floor(path: Path) -> dict[str, float | None]:
    """Read a ``{metric: value}`` floor table, keeping undefined metrics undefined.

    A metric that was undefined in either replica has no floor, and the file
    stores it as ``null``; turning that into 0.0 here would make every
    difference in that metric clear a floor of nothing.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object of metric values")
    return {
        str(key): None if value is None else float(value)
        for key, value in data.items()
    }


def resolve_path(path: Path) -> Path:
    """Anchor a relative path at the repository root.

    The report is run from wherever a shell happens to be; resolving
    relative paths from the repository root makes the commands independent
    of the working directory.
    """
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def open_store(runs: Path) -> RunStore:
    """Open ``RUNS_DIR/NAME`` as the store of the run called ``NAME``."""
    runs = resolve_path(runs)
    if not runs.name:
        raise ValueError(f"--runs {runs} does not name a run directory")
    return RunStore(runs.parent, runs.name)


def select_recordings(
    ids_error: Sequence[str], ids_normal: Sequence[str], subset: str
) -> list[str]:
    """The recordings a report covers, sorted and deduplicated.

    The same selection rule as the batch script, so a report over ``--subset
    error`` scores exactly the recordings that run produced.
    """
    if subset == "error":
        selected = list(ids_error)
    elif subset == "normal":
        selected = list(ids_normal)
    elif subset == "all":
        selected = list(ids_error) + list(ids_normal)
    else:
        raise ValueError(f"unknown subset {subset!r}")
    return sorted(set(selected))


def write_reports(
    evaluation: Evaluation, out: Path, name: str
) -> list[Path]:
    """Write one run's report files and return what was written."""
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "metrics.json": json.dumps(evaluation.to_dict(), indent=1, ensure_ascii=False),
        "results_row.md": render_results_row(evaluation, name),
        "results_row.csv": render_results_row_csv(evaluation, name),
        "literature_row.md": render_literature_row(evaluation, name),
        "localization.md": render_localization(evaluation),
        "localization.csv": render_localization_csv(evaluation),
    }
    written = []
    for filename, text in files.items():
        path = out / filename
        path.write_text(text + "\n", encoding="utf-8")
        written.append(path)
    return written


def status_line(evaluation: Evaluation, n_recordings: int) -> str:
    """One line saying what was scored, what failed, and what was left out.

    ``excluded_optional``, the configuration ids and the order reading belong
    here and not only in the JSON: they are the three facts that make the
    printed numbers readable: how many unperformed optional steps were
    dropped, whether the row is one configuration or a mixture the caller
    allowed, and
    which of the two order columns the triple came from.
    """
    policy = OrderPolicy(**evaluation.order_policy)
    return (
        f"{evaluation.run_name}: {n_recordings} recordings, "
        f"{evaluation.instances} instances "
        f"({evaluation.excluded_optional} optional excluded), "
        f"status {dict(sorted(evaluation.status_counts.items()))}, "
        f"{evaluation.length_turns} length-truncated turns, "
        f"config {', '.join(evaluation.config_ids) or 'unknown'}, "
        f"order {policy.label}"
    )


def main() -> None:
    args = parse_args()
    annotations = AnnotationSet.load(resolve_path(args.annotations_dir))
    ids_error, ids_normal = load_split(resolve_path(args.split))
    recording_ids = select_recordings(ids_error, ids_normal, args.subset)
    out = resolve_path(args.out)

    policy = order_policy(args)
    store = open_store(args.runs)
    evaluation = evaluate_run(
        store,
        recording_ids,
        annotations,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        subset=args.subset,
        allow_mixed_config=args.allow_mixed_config,
        order_policy=policy,
    )
    written = write_reports(evaluation, out, store.run_name)
    print(status_line(evaluation, len(recording_ids)), flush=True)
    print(render_results_row(evaluation, store.run_name), flush=True)

    if args.runs_b:
        store_b = open_store(args.runs_b)
        floor = (
            None if args.noise_floor is None
            else load_noise_floor(resolve_path(args.noise_floor))
        )
        comparison = compare_runs(
            store,
            store_b,
            recording_ids,
            annotations,
            n_bootstrap=args.bootstrap,
            seed=args.seed,
            noise_floor=floor,
            allow_mixed_config=args.allow_mixed_config,
            require_replica=args.write_noise_floor,
            order_policy=policy,
        )
        (out / "comparison.md").write_text(
            render_comparison(comparison) + "\n", encoding="utf-8"
        )
        (out / "absolute_difference.json").write_text(
            json.dumps(comparison["absolute_difference"], indent=1) + "\n",
            encoding="utf-8",
        )
        written.extend([out / "comparison.md", out / "absolute_difference.json"])
        if args.write_noise_floor:
            # The caller asserts these two runs are replicas of one
            # configuration; ``require_replica`` has just checked that claim
            # against what the records stamp. The floor of each metric is the
            # larger of |A - B| and half the width of the paired interval
            # and the detail file keeps both so a reader sees
            # which one won.
            detail = noise_floor_detail(comparison)
            (out / "noise_floor.json").write_text(
                json.dumps(noise_floor_values(detail), indent=1) + "\n",
                encoding="utf-8",
            )
            (out / "noise_floor_detail.json").write_text(
                json.dumps(detail, indent=1) + "\n", encoding="utf-8"
            )
            written.extend(
                [out / "noise_floor.json", out / "noise_floor_detail.json"]
            )
        print(render_comparison(comparison), flush=True)

    for path in written:
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()

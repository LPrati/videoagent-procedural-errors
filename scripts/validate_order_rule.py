"""Check the computed-order rule against the annotated order errors.

The order rule used by the computed order reading (``scripts/evaluate.py
--order-from anchors``) is not documented by CaptainCook4D; it was inferred
from the annotations themselves. This script measures how well it reproduces
the ``Order Error`` labels when it is applied to the annotated intervals, as
if a system had localized every step perfectly:

* every annotated instance with times becomes an "executed" verdict whose
  anchor is the annotated interval, and every instance without times a
  "not executed" verdict;
* the order column is then derived from those anchors with the rule of
  ``--order-from anchors`` (by default: transitive prerequisites, both ends
  of a displaced pair, a prerequisite offends only when it starts after the
  dependent step ends);
* the derived column is matched to the annotated instances and scored with
  the same precision, recall and F1 as the evaluation.

Reads only the annotations: no model, no GPU, no run records. On the
evaluated development subset (``data_splits/dev_83.json``) the rule reaches
F1 0.92, and on the test subset (``data_splits/test.json``) F1 0.90.

Usage:
    uv run python scripts/validate_order_rule.py --split data_splits/dev_83.json
    uv run python scripts/validate_order_rule.py --split data_splits/test.json \\
        --annotations-dir /path/to/captaincook4d/annotations
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

from video_agent.eval.matching import is_optional_unperformed, match_recording
from video_agent.eval.metrics import is_positive_prediction, is_positive_truth
from video_agent.eval.order_from_anchors import OrderPolicy, apply_order_policy
from video_agent.task.annotations import AnnotationSet, Recording, load_split
from video_agent.task.verdicts import (
    Anchor,
    Execution,
    Order,
    RecordingVerdicts,
    StepVerdict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "data" / "captaincook4d" / "annotations"


def oracle_verdicts(recording: Recording) -> RecordingVerdicts:
    """One verdict per annotated instance, anchored on its annotated interval.

    The answer of a system that localized every step exactly and wrote no
    order column of its own. Instances of an optional step that was not
    performed are left out, as the evaluation leaves them out.
    """
    steps = []
    for instance in recording.steps:
        if is_optional_unperformed(instance):
            continue
        if instance.start_s is None or instance.end_s is None:
            steps.append(
                StepVerdict(
                    step_id=instance.step_id,
                    execution=Execution.NOT_EXECUTED,
                    order=Order.NOT_APPLICABLE,
                    anchor=None,
                    prerequisite_step_id=None,
                    prerequisite_anchor=None,
                    raw=None,
                )
            )
        else:
            steps.append(
                StepVerdict(
                    step_id=instance.step_id,
                    execution=Execution.EXECUTED,
                    order=Order.OK,
                    anchor=Anchor(
                        start_s=instance.start_s,
                        end_s=instance.end_s,
                        segments=(0, 0),
                    ),
                    prerequisite_step_id=None,
                    prerequisite_anchor=None,
                    raw=None,
                )
            )
    return RecordingVerdicts(
        recording_id=recording.recording_id, steps=tuple(steps), status="ok"
    )


def confusion(
    annotations: AnnotationSet, recording_ids: Sequence[str], policy: OrderPolicy
) -> dict[str, int]:
    """True/false positives and negatives of the rule over ``recording_ids``."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for recording_id in recording_ids:
        recording = annotations.recordings[recording_id]
        _, graph = annotations.for_recording(recording_id)
        derived = apply_order_policy(oracle_verdicts(recording), graph, policy)
        for matched in match_recording(recording, derived):
            truth = is_positive_truth(matched.instance, "order")
            prediction = is_positive_prediction(matched.verdict, "order")
            if truth and prediction:
                counts["tp"] += 1
            elif prediction:
                counts["fp"] += 1
            elif truth:
                counts["fn"] += 1
            else:
                counts["tn"] += 1
    return counts


def prf(counts: dict[str, int]) -> dict[str, float | None]:
    """Precision, recall and F1 of a confusion table (``None`` when undefined)."""
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument(
        "--order-rule", choices=["direct", "transitive"], default="transitive"
    )
    parser.add_argument("--order-edges", choices=["early", "both"], default="both")
    parser.add_argument("--order-margin", choices=["start", "end"], default="end")
    parser.add_argument("--out", type=Path, help="optional JSON file for the result")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    annotations = AnnotationSet.load(args.annotations_dir)
    ids_error, ids_normal = load_split(args.split)
    recording_ids = sorted(set(ids_error) | set(ids_normal))
    policy = OrderPolicy(
        source="anchors",
        rule=args.order_rule,
        edges=args.order_edges,
        margin=args.order_margin,
    )
    counts = confusion(annotations, recording_ids, policy)
    scores = prf(counts)
    result = {
        "split": str(args.split),
        "recordings": len(recording_ids),
        "order_rule": policy.label,
        **counts,
        **scores,
    }
    print(
        f"{len(recording_ids)} recordings, rule {policy.label}: "
        f"TP {counts['tp']} FP {counts['fp']} FN {counts['fn']} TN {counts['tn']}; "
        + " ".join(
            f"{name} {value:.3f}" if value is not None else f"{name} undefined"
            for name, value in scores.items()
        )
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Evaluation: run records in, metrics and report tables out.

This package reads what a run wrote, pairs every verdict with the annotated
instance it refers to, and computes the evaluation quantities: per-category
precision/recall/F1, the per-recording variant, pairwise order accuracy,
localization by duration bin, the unclassified rate, EDA and cost. It also
provides bootstrap confidence intervals by recording, the constant baselines,
and a synthetic harness that checks the metric code against cases with known
answers.

Two rules run through every module. A metric that is not defined on the data
it was given returns ``None`` rather than 0 or NaN: the precision of a system
that predicted nothing is not zero, it is undefined, and it is reported as
such. And everything except ``report`` is a pure function of its arguments,
so every number can be reproduced from a run record alone: nothing here reads
a clock or a global.
"""

# Re-exported here because the order reading is a choice about the
# evaluation as a whole, not a detail of one report: ``scripts/evaluate.py``
# builds an ``OrderPolicy`` before it touches ``report``.
from video_agent.eval.order_from_anchors import (
    LLM_ORDER,
    OrderPolicy,
    apply_order_policy,
    derive_order_from_anchors,
)

__all__ = [
    "LLM_ORDER",
    "OrderPolicy",
    "apply_order_policy",
    "derive_order_from_anchors",
]

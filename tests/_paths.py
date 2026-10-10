"""Locations the tests read, resolved from the repository root.

The CaptainCook4D annotations are not part of this repository. Tests that
need them read ``data/captaincook4d/annotations`` (or the directory named by
``CC4D_ANNOTATIONS_DIR``) and are skipped when it is absent, so the suite
runs on a fresh clone with no data and no GPU.
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
DATA_SPLITS = REPO_ROOT / "data_splits"
DEV_SPLIT = DATA_SPLITS / "dev.json"

DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "data" / "captaincook4d" / "annotations"
"""Where the scripts look for the annotations unless told otherwise."""

ANNOTATIONS_DIR = Path(
    os.environ.get("CC4D_ANNOTATIONS_DIR") or DEFAULT_ANNOTATIONS_DIR
)
HAVE_ANNOTATIONS = (
    ANNOTATIONS_DIR / "annotation_json" / "error_annotations.json"
).is_file()

requires_annotations = pytest.mark.skipif(
    not HAVE_ANNOTATIONS,
    reason="CaptainCook4D annotations not found (see README, 'Data')",
)


def load_annotations():
    """The annotation set, or skip the calling test when it is not available."""
    if not HAVE_ANNOTATIONS:
        pytest.skip("CaptainCook4D annotations not found (see README, 'Data')")
    from video_agent.task.annotations import AnnotationSet

    return AnnotationSet.load(ANNOTATIONS_DIR)

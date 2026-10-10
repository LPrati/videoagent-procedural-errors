from pathlib import Path

import numpy as np
import pytest

from video_agent.memory.artifacts import VideoArtifacts


@pytest.fixture
def artifacts(tmp_path: Path) -> VideoArtifacts:
    return VideoArtifacts(tmp_path, "video_01")


def test_roundtrips(artifacts: VideoArtifacts) -> None:
    artifacts.save_json("captions.json", ["a", "b"])
    artifacts.save_array("emb.npy", np.arange(6).reshape(2, 3))
    artifacts.save_pickle("obj.pkl", {"x": (1, 2)})

    assert artifacts.load_json("captions.json") == ["a", "b"]
    assert artifacts.load_array("emb.npy").tolist() == [[0, 1, 2], [3, 4, 5]]
    assert artifacts.load_pickle("obj.pkl") == {"x": (1, 2)}
    assert artifacts.has("emb.npy") and not artifacts.has("missing.npy")


def test_lineage_accumulates_per_stage(artifacts: VideoArtifacts) -> None:
    artifacts.record_stage(
        "detect",
        artifact="detections.pkl",
        model="PekingU/rtdetr_v2_r50vd",
        params={"fps": 15.0, "threshold": 0.25},
        code_version="abc1234",
    )
    artifacts.record_stage("track", artifact="tracklets.pkl", model="bytetrack")

    detect = artifacts.stage_meta("detect")
    assert detect["model"] == "PekingU/rtdetr_v2_r50vd"
    assert detect["params"] == {"fps": 15.0, "threshold": 0.25}
    assert detect["code_version"] == "abc1234"
    assert "created_at" in detect
    assert artifacts.stage_meta("track")["artifact"] == "tracklets.pkl"
    assert artifacts.stage_meta("caption") is None


def test_failed_write_leaves_no_artifact(artifacts: VideoArtifacts) -> None:
    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        artifacts.save_pickle("obj.pkl", Unpicklable())
    assert not artifacts.has("obj.pkl")
    assert list(artifacts.directory.glob(".obj.pkl.*")) == []  # temp cleaned up

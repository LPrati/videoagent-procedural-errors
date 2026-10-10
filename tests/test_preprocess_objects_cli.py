"""The object preprocessing CLI: which detector, which memory, which lineage.

``scripts/`` is not a package, so the module is loaded from its path. Nothing
here loads a checkpoint: the detector and the crop embedder are the fakes of
``tests.fakes``, injected into ``LazyModels`` through the attributes it caches
them in, so the four stages run end to end over the synthetic video on a
machine with no GPU. What is under test is the plumbing around the models —
the slug the artifacts are named with, the lineage the stages record, and the
resume rules — which is where a mistake would be silent rather than loud.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.fakes import FakeCropEmbedder, FakeDetector
from video_agent.adapters.detector import DEFAULT_DETECTOR
from video_agent.media import Video
from video_agent.memory.artifacts import VideoArtifacts
from video_agent.memory.types import ObjectMemory

from tests._paths import SCRIPTS

BASE_ARGV = ["preprocess_objects.py", "--out", "preprocess"]

@pytest.fixture(scope="module")
def preprocess_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "preprocess_objects_cli", SCRIPTS / "preprocess_objects.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parse(preprocess_module, monkeypatch: pytest.MonkeyPatch):
    def parse_argv(*extra: str):
        monkeypatch.setattr(sys, "argv", BASE_ARGV + list(extra))
        return preprocess_module.parse_args()

    return parse_argv


class TestDetectorSelection:
    def test_the_default_detector_writes_the_unsuffixed_artifacts(
        self, parse, preprocess_module
    ) -> None:
        from video_agent.memory.store import detections_file, object_memory_file

        args = parse()

        assert args.detector_model == DEFAULT_DETECTOR
        assert args.object_slug == "rtdetr-v2-r50"
        assert detections_file(args.object_slug) == "detections.pkl"
        assert object_memory_file(args.object_slug) == "object_memory.pkl"

    def test_the_sampling_rate_is_a_flag_with_the_old_constant_as_default(
        self, parse, preprocess_module
    ) -> None:
        assert parse().detection_fps == preprocess_module.DETECTION_FPS == 15.0
        assert parse("--detection-fps", "5").detection_fps == 5.0

    def test_the_detector_knobs_default_to_the_adapter_and_can_be_overridden(
        self, parse
    ) -> None:
        args = parse()

        # ``None`` means "the adapter's own" score cut.
        assert args.score_threshold is None
        assert args.detector_batch == 16

        args = parse("--score-threshold", "0.2", "--detector-batch", "8")
        assert (args.score_threshold, args.detector_batch) == (0.2, 8)

    def test_an_unknown_detector_is_refused_before_any_video_is_read(
        self, parse
    ) -> None:
        with pytest.raises(SystemExit):
            parse("--detector-model", "nobody/nothing")


def inject(models_class, args, detector, crop_embedder=None):
    """``LazyModels`` with both models already in the attributes it caches.

    The models are injected rather than the class replaced, so the stages run
    against the real one and a stage that reached for a third model would
    still try to load a checkpoint.
    """
    models = models_class(args)
    models._detector = detector
    models._crop_embedder = (
        FakeCropEmbedder() if crop_embedder is None else crop_embedder
    )
    return models


def fake_models(preprocess_module, args, detector, crop_embedder=None):
    """``inject`` against the script's own class, for the stage-level tests."""
    return inject(preprocess_module.LazyModels, args, detector, crop_embedder)


def run_stages(preprocess_module, args, video: Video, artifacts, models) -> dict:
    """The four stages of one video, in order, as ``main`` would run them."""
    slug = args.object_slug
    counts = {
        "detect": preprocess_module.stage_detect(
            video, artifacts, models, args, "abc1234", slug
        ),
        "track": preprocess_module.stage_track(
            video, artifacts, args, "abc1234", slug
        ),
        "embed": preprocess_module.stage_embed(
            video, artifacts, models, args, "abc1234", slug
        ),
    }
    counts["assemble"] = preprocess_module.stage_assemble(
        video, artifacts, "abc1234", slug
    )
    return counts


class TestStages:
    """The four stages over the synthetic video, with fakes for both models."""

    def test_the_default_detector_writes_the_files_every_recording_has(
        self, preprocess_module, parse, synthetic_video: Video, tmp_path: Path
    ) -> None:
        pytest.importorskip("trackers")
        args = parse()
        artifacts = VideoArtifacts(tmp_path, synthetic_video.path.stem)

        counts = run_stages(
            preprocess_module,
            args,
            synthetic_video,
            artifacts,
            fake_models(preprocess_module, args, FakeDetector(("cup",))),
        )

        # 150 frames at 30 fps sampled at 15 fps is one detection every
        # other frame, and one tracked object.
        assert counts["detect"] == 75
        assert counts["track"] == 1 and counts["assemble"] == 1
        for name in (
            "detections.pkl",
            "tracklets.pkl",
            "tracklet_embeddings.npy",
            "object_memory.pkl",
        ):
            assert artifacts.has(name)
        memory = artifacts.load_pickle("object_memory.pkl")
        assert isinstance(memory, ObjectMemory)
        assert memory.records[0].category == "cup"

    def test_the_detect_lineage_carries_the_detector_the_timing_and_the_libraries(
        self, preprocess_module, parse, synthetic_video: Video, tmp_path: Path
    ) -> None:
        args = parse()
        artifacts = VideoArtifacts(tmp_path, synthetic_video.path.stem)

        preprocess_module.stage_detect(
            synthetic_video,
            artifacts,
            fake_models(preprocess_module, args, FakeDetector(("cup",))),
            args,
            "abc1234",
            args.object_slug,
        )

        stage = artifacts.stage_meta("detect")
        assert stage["model"] == DEFAULT_DETECTOR
        assert stage["artifact"] == "detections.pkl"
        params = stage["params"]
        assert params["detection_fps"] == 15.0
        # Whatever the adapter reports about itself rides along verbatim.
        assert params["checkpoint"] == "fake-detector"
        assert params["score_threshold"] == 0.5
        timing = params["timing"]
        assert timing["n_frames"] == 75
        assert timing["decode_s"] >= 0.0 and timing["detect_s"] >= 0.0
        assert timing["stage_s"] >= timing["detect_s"]
        assert timing["frames_per_s_stage"] > 0
        # Nothing here runs on a GPU, and "unknown" is a value.
        assert "peak_vram_bytes" in timing
        assert set(params["library_versions"]) == {"torch", "transformers"}

    def test_the_detections_file_holds_the_frames_and_their_detections(
        self, preprocess_module, parse, synthetic_video: Video, tmp_path: Path
    ) -> None:
        args = parse()
        artifacts = VideoArtifacts(tmp_path, synthetic_video.path.stem)

        preprocess_module.stage_detect(
            synthetic_video,
            artifacts,
            fake_models(preprocess_module, args, FakeDetector()),
            args,
            "abc1234",
            args.object_slug,
        )

        assert sorted(artifacts.load_pickle("detections.pkl")) == [
            "frame_indices",
            "per_frame",
        ]

    def test_the_tracker_is_told_the_rate_the_frames_were_sampled_at(
        self, preprocess_module, parse, synthetic_video: Video, tmp_path: Path
    ) -> None:
        """A wrong frame rate halves ByteTrack's lost-track buffer silently."""
        pytest.importorskip("trackers")
        args = parse("--detection-fps", "5")
        artifacts = VideoArtifacts(tmp_path, synthetic_video.path.stem)

        run_stages(
            preprocess_module,
            args,
            synthetic_video,
            artifacts,
            fake_models(preprocess_module, args, FakeDetector()),
        )

        assert artifacts.stage_meta("track")["params"]["frame_rate"] == 5.0

    def test_the_track_stage_reads_the_rate_off_the_detect_record(
        self, preprocess_module, parse, synthetic_video: Video, tmp_path: Path
    ) -> None:
        """A resumed run may carry another flag than the one that detected."""
        pytest.importorskip("trackers")
        artifacts = VideoArtifacts(tmp_path, synthetic_video.path.stem)
        slow = parse("--detection-fps", "5")
        preprocess_module.stage_detect(
            synthetic_video,
            artifacts,
            fake_models(preprocess_module, slow, FakeDetector()),
            slow,
            "abc1234",
            "rtdetr-v2-r50",
        )

        preprocess_module.stage_track(
            synthetic_video, artifacts, parse(), "abc1234", "rtdetr-v2-r50"
        )

        assert artifacts.stage_meta("track")["params"]["frame_rate"] == 5.0

    def test_the_track_stage_records_the_bytetrack_thresholds_that_ran(
        self, preprocess_module, parse, synthetic_video: Video, tmp_path: Path
    ) -> None:
        """The record carries the thresholds, defaulted or not.

        Two memories are comparable only if the record carries the numbers
        that ran, including the library's defaults.
        """
        pytest.importorskip("trackers")
        artifacts = VideoArtifacts(tmp_path, synthetic_video.path.stem)
        args = parse()
        run_stages(
            preprocess_module,
            args,
            synthetic_video,
            artifacts,
            fake_models(preprocess_module, args, FakeDetector()),
        )
        defaults = artifacts.stage_meta("track")["params"]
        assert defaults["track_activation_threshold"] == 0.7
        assert defaults["high_conf_det_threshold"] == 0.6

        lenient = parse(
            "--track-activation-threshold", "0.37",
            "--track-high-conf-threshold", "0.32",
        )
        preprocess_module.stage_track(
            synthetic_video, artifacts, lenient, "abc1234", "rtdetr-v2-r50"
        )
        params = artifacts.stage_meta("track")["params"]
        assert params["track_activation_threshold"] == 0.37
        assert params["high_conf_det_threshold"] == 0.32
        assert params["frame_rate"] == args.detection_fps


def run_main(
    preprocess_module, monkeypatch, detector, *argv: str, crop_embedder=None
) -> None:
    """Drive ``main`` end to end with both models injected into ``LazyModels``."""
    monkeypatch.setattr(sys, "argv", ["preprocess_objects.py", *argv])
    # Captured before the patch: the factory below still builds the real
    # class, it just hands it the models instead of letting it load them.
    models_class = preprocess_module.LazyModels
    monkeypatch.setattr(
        preprocess_module,
        "LazyModels",
        lambda args: inject(models_class, args, detector, crop_embedder),
    )
    preprocess_module.main()


class TestResume:
    def test_a_second_run_recomputes_nothing(
        self, preprocess_module, monkeypatch, synthetic_video: Video, tmp_path: Path,
        capsys,
    ) -> None:
        pytest.importorskip("trackers")
        out = tmp_path / "preprocess"
        argv = (str(synthetic_video.path), "--out", str(out))
        run_main(preprocess_module, monkeypatch, FakeDetector(("cup",)), *argv)
        capsys.readouterr()

        # A detector that refuses to run: reaching it means the cached
        # detections were ignored.
        run_main(preprocess_module, monkeypatch, NeverDetects(), *argv)

        assert "all stages cached" in capsys.readouterr().out

    def test_forcing_one_stage_redoes_it_and_everything_after_it(
        self, preprocess_module, monkeypatch, synthetic_video: Video, tmp_path: Path
    ) -> None:
        pytest.importorskip("trackers")
        out = tmp_path / "preprocess"
        argv = (str(synthetic_video.path), "--out", str(out))
        run_main(preprocess_module, monkeypatch, FakeDetector(("cup",)), *argv)
        artifacts = VideoArtifacts(out, synthetic_video.path.stem)
        before = artifacts.stage_meta("embed")

        run_main(
            preprocess_module,
            monkeypatch,
            NeverDetects(),
            *argv,
            "--force-stages",
            "embed",
            crop_embedder=FakeCropEmbedder((3, 1)),
        )

        # detect and track were cached (the poison detector was never
        # asked), embed and assemble ran again with the new embedder.
        assert artifacts.stage_meta("detect")["params"]["checkpoint"] == "fake-detector"
        assert artifacts.stage_meta("embed")["params"]["widths"] == [3, 1]
        assert before["params"]["widths"] == [2, 2]
        assert artifacts.stage_meta("assemble")["params"]["widths"] == [3, 1]


class NeverDetects:
    """A detector that must not be reached; caching is what keeps it unused."""

    score_threshold = 0.5
    batch_size = 1

    def detect(self, frames):
        raise AssertionError("the detect stage ran over a cached artifact")

    def lineage(self) -> dict:
        raise AssertionError("the detect stage ran over a cached artifact")

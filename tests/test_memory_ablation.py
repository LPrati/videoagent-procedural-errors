"""The two memory controls: the withheld memory and the shuffled one.

Condition A is the recording's own memory (``--memory full``), B the
withheld one (``--memory empty``) and C the shuffled one (``--memory
shuffled``). What the tests are about is that B answers nothing at all, without loading a model or
opening an artifact, and that C is the *same* memory renamed: the same
captions, the same vectors, the same object-segment pairs, all moved by one
permutation — including the clip the visual-question-answering tool plays,
which has to be the clip the caption the agent just read describes.
"""

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from video_agent.adapters.aligner import aligner_slug
from video_agent.agent.config import DEFAULT_ALIGNER_MODEL as ALIGNER
from video_agent.agent.config import ToolConfig
from video_agent.agent.react import ReActRunner
from video_agent.agent.tools import MAIN_TOOL_NAMES, MEMORY_UNAVAILABLE, ToolKit
from video_agent.media import Video
from video_agent.memory.ablation import (
    MEMORY_MODES,
    PERMUTATION_BASE_SEED,
    SegmentPermutation,
    WithheldVideoQA,
    segment_permutation,
    shuffle_memories,
    shuffle_object_memory,
    shuffle_temporal_memory,
    withheld_temporal_memory,
)
from video_agent.memory.object_db import ObjectDatabase
from video_agent.memory.preprocess import get_video_segments
from video_agent.memory.store import object_memory_file

from tests.fakes import (
    FakeAligner,
    FakeChat,
    FakeSiglip,
    FakeTextEmbedder,
    FakeVideoQA,
    make_object_memory,
    make_temporal_memory,
    object_record,
)

VIDEO = Video(path=Path("unused.mp4"), fps=30, frame_count=300)  # 5 segments

from tests._paths import ANNOTATIONS_DIR, HAVE_ANNOTATIONS, SCRIPTS, load_annotations


@pytest.fixture(scope="module")
def run_task_module():
    """``scripts/run_task.py``, loaded from its path as the shell runs it."""
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("run_task_cli", SCRIPTS / "run_task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wiring():
    """``scripts/_wiring.py``, loaded from its path as the scripts load it."""
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("wiring", SCRIPTS / "_wiring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def objects():
    """Three objects over five segments: eight object-segment pairs in all."""
    return make_object_memory(
        object_record(1, "person", (0, 1, 4), [1.0, 0.0, 9.0]),
        object_record(2, "bowl", (2, 3), [0.2, 1.0, 9.0]),
        object_record(3, "cup", (0, 4, 4), [0.5, 0.5, 9.0]),
    )


def memory(n: int = 5):
    return make_temporal_memory(n)


def kit(**overrides) -> ToolKit:
    arguments = {
        "video": VIDEO,
        "temporal": memory(),
        "text_embedder": FakeTextEmbedder(),
        "aligner": FakeAligner(),
    }
    arguments.update(overrides)
    return ToolKit(**arguments)


class TestPermutation:
    """One permutation per recording, drawn from the split's seed."""

    def test_the_seed_is_the_base_plus_the_recordings_position(self) -> None:
        permutation = segment_permutation(6, ordinal=7)

        assert permutation.ordinal == 7
        assert permutation.seed == PERMUTATION_BASE_SEED + 7 == 20260910

    def test_the_same_ordinal_always_draws_the_same_order(self) -> None:
        # Four shards of one run each draw the permutations of their own
        # recordings; a rerun of one recording has to reproduce its own.
        first = segment_permutation(40, ordinal=3)
        second = segment_permutation(40, ordinal=3)

        assert first.perm == second.perm
        assert first == second

    def test_different_ordinals_draw_different_orders(self) -> None:
        orders = {segment_permutation(40, ordinal=n).perm for n in range(20)}

        assert len(orders) == 20

    def test_the_order_is_a_permutation_and_moves_the_segments(self) -> None:
        permutation = segment_permutation(50, ordinal=0)

        assert sorted(permutation.perm) == list(range(50))
        assert permutation.perm != tuple(range(50))
        assert permutation.n_segments == 50

    def test_the_inverse_renames_an_original_segment(self) -> None:
        permutation = segment_permutation(12, ordinal=5)
        inverse = permutation.inverse()

        for new, old in enumerate(permutation.perm):
            assert inverse[old] == new

    def test_an_order_that_is_not_a_permutation_is_refused(self) -> None:
        with pytest.raises(ValueError):
            SegmentPermutation(perm=(0, 1, 1), seed=1, ordinal=0)

    def test_the_record_of_a_permutation_reproduces_its_digest(self) -> None:
        """``permutation.json`` and the record's digest have to agree."""
        permutation = segment_permutation(9, ordinal=2)

        written = json.loads(json.dumps(permutation.to_dict(), indent=1))
        read_back = SegmentPermutation.from_dict(written)

        assert read_back == permutation
        assert read_back.sha256() == permutation.sha256()
        assert written["perm"] == list(permutation.perm)
        assert written["seed"] == permutation.seed
        assert written["ordinal"] == 2 and written["n_segments"] == 9
        assert permutation.sha256() != segment_permutation(9, ordinal=3).sha256()

    def test_the_three_modes_are_the_three_conditions(self) -> None:
        assert MEMORY_MODES == ("full", "empty", "shuffled")


class TestShuffledTemporalMemory:
    """Condition C: the same content, renamed."""

    def test_the_captions_are_the_same_multiset_in_another_order(self) -> None:
        original = memory(7)
        permutation = segment_permutation(7, ordinal=1)

        shuffled = shuffle_temporal_memory(original, permutation)

        assert sorted(shuffled.captions) == sorted(original.captions)
        assert shuffled.captions != original.captions
        assert shuffled.captions == [original.captions[old] for old in permutation.perm]

    def test_both_embedding_channels_move_with_their_caption(self) -> None:
        original = memory(7)
        permutation = segment_permutation(7, ordinal=1)

        shuffled = shuffle_temporal_memory(original, permutation)

        for new, old in enumerate(permutation.perm):
            assert np.array_equal(
                shuffled.caption_embeddings[new], original.caption_embeddings[old]
            )
            assert np.array_equal(
                shuffled.visual_embeddings[new], original.visual_embeddings[old]
            )

    def test_a_row_keeps_the_time_of_the_segment_it_describes(self) -> None:
        """The grid is renamed, not re-timed: row ``k`` is when ``perm[k]`` was."""
        original = memory(7)
        permutation = segment_permutation(7, ordinal=1)

        shuffled = shuffle_temporal_memory(original, permutation)

        for new, old in enumerate(permutation.perm):
            assert shuffled.segments[new].index == new  # its new name
            assert shuffled.segments[new].start_s == original.segments[old].start_s
            assert shuffled.segments[new].end_s == original.segments[old].end_s
            assert (
                shuffled.segments[new].frame_range
                == original.segments[old].frame_range
            )

    def test_the_original_memory_is_never_edited(self) -> None:
        original = memory(7)
        captions = list(original.captions)
        embeddings = original.caption_embeddings.copy()

        shuffle_temporal_memory(original, segment_permutation(7, ordinal=1))

        assert original.captions == captions
        assert np.array_equal(original.caption_embeddings, embeddings)

    def test_caption_retrieval_answers_with_the_original_caption(self) -> None:
        """Asking for segment ``k`` reads original segment ``perm[k]``."""
        original = memory(5)
        permutation = segment_permutation(5, ordinal=4)
        view = shuffle_memories(original, None, permutation)
        shuffled = kit(
            temporal=view.temporal,
            permutation=permutation,
            video_segments=view.video_segments,
        )

        for new, old in enumerate(permutation.perm):
            assert shuffled.caption_retrieval(new, new) == (
                "There are 5 segments in total, ranging from 0 to 4. "
                f"{{{new}: '{original.captions[old]}'}}"
            )

    def test_a_caption_window_is_the_shuffled_memorys_own_window(self) -> None:
        original = memory(5)
        permutation = segment_permutation(5, ordinal=4)
        view = shuffle_memories(original, None, permutation)
        shuffled = kit(temporal=view.temporal, permutation=permutation)

        answer = shuffled.caption_retrieval(1, 3)

        assert answer.endswith(
            str({index: original.captions[permutation.perm[index]] for index in (1, 2, 3)})
        )

    def test_a_permutation_of_another_size_is_refused(self) -> None:
        with pytest.raises(ValueError):
            shuffle_temporal_memory(memory(5), segment_permutation(6, ordinal=0))
        with pytest.raises(ValueError):
            kit(permutation=segment_permutation(6, ordinal=0))


class TestShuffledObjectMemory:
    """(iii): the object memory is renamed by the same permutation."""

    def test_every_appearance_is_renamed_and_none_is_lost(self) -> None:
        permutation = segment_permutation(5, ordinal=2)
        inverse = permutation.inverse()
        original = objects()

        shuffled = shuffle_object_memory(original, permutation)

        for before, after in zip(original.records, shuffled.records):
            assert after.object_id == before.object_id
            assert after.category == before.category
            assert set(after.segment_indices) == {
                inverse[index] for index in before.segment_indices
            }

    def test_the_database_holds_exactly_the_same_pairs_renamed(self) -> None:
        permutation = segment_permutation(5, ordinal=2)
        inverse = permutation.inverse()
        original = objects()
        shuffled = shuffle_object_memory(original, permutation)

        with ObjectDatabase(original, 5) as before, ObjectDatabase(shuffled, 5) as after:
            pairs = "SELECT object_id, segment_id FROM Objects_Segments"
            # The tool answers ``str(rows)``, which is what the agent reads.
            ordered = ast.literal_eval(before.query(pairs))
            renamed = ast.literal_eval(after.query(pairs))

        assert len(renamed) == len(ordered) == 7  # the duplicate (3, 4) is one pair
        assert sorted(renamed) == sorted(
            (object_id, inverse[segment]) for object_id, segment in ordered
        )

    def test_the_sub_agents_lookup_still_finds_the_same_objects(self) -> None:
        """Only the segments move: a crop is the crop it was."""
        permutation = segment_permutation(5, ordinal=2)

        def retrieve(object_memory, **overrides) -> str:
            return kit(
                objects=object_memory,
                object_text_embedder=FakeSiglip(("cup", "dog")),
                object_text_columns=slice(0, 2),
                **overrides,
            ).open_vocabulary_object_retrieval("cup")

        assert retrieve(
            shuffle_object_memory(objects(), permutation),
            temporal=shuffle_temporal_memory(memory(5), permutation),
            permutation=permutation,
        ) == retrieve(objects())


class TestShuffledVisualQuestionAnswering:
    """(iv): the tool plays the clip the shuffled memory calls ``k``."""

    def test_the_clip_is_the_original_segment_the_caption_came_from(
        self, synthetic_video: Video
    ) -> None:
        # Two segments of 2 s at 30 fps; no context, so the window is exactly
        # one segment and the frames say which.
        segments = get_video_segments(synthetic_video)
        permutation = SegmentPermutation(perm=(1, 0), seed=1, ordinal=0)
        original = make_temporal_memory(2)
        original.segments = segments
        view = shuffle_memories(original, None, permutation)
        model = FakeVideoQA()
        shuffled = ToolKit(
            video=synthetic_video,
            temporal=view.temporal,
            text_embedder=FakeTextEmbedder(),
            aligner=FakeAligner(),
            vqa=model,
            permutation=permutation,
            video_segments=view.video_segments,
            config=ToolConfig(vqa_context_segments=0),
        )

        shuffled.visual_question_answering("what happens?", 0)

        # Shuffled segment 0 carries original segment 1: frames 60..119.
        assert [frame.index for frame in model.clips[0].frames] == list(range(60, 120, 6))

    def test_an_unshuffled_toolkit_plays_the_segment_it_was_asked_about(
        self, synthetic_video: Video
    ) -> None:
        segments = get_video_segments(synthetic_video)
        original = make_temporal_memory(2)
        original.segments = segments
        model = FakeVideoQA()
        full = ToolKit(
            video=synthetic_video,
            temporal=original,
            text_embedder=FakeTextEmbedder(),
            aligner=FakeAligner(),
            vqa=model,
            config=ToolConfig(vqa_context_segments=0),
        )

        full.visual_question_answering("what happens?", 0)

        assert [frame.index for frame in model.clips[0].frames] == list(range(0, 60, 6))

    def test_a_segment_outside_the_grid_is_still_refused(self) -> None:
        permutation = segment_permutation(5, ordinal=0)
        view = shuffle_memories(memory(5), None, permutation)
        shuffled = kit(
            temporal=view.temporal,
            vqa=FakeVideoQA(),
            permutation=permutation,
            video_segments=view.video_segments,
        )

        assert shuffled.visual_question_answering("q", 9) == (
            "Segment ID 9 not in range 0-4."
        )


class TestWithheldMemory:
    """Condition B: the tools are there, and there is nothing behind them."""

    def withheld(self) -> tuple[ToolKit, dict]:
        """A withheld toolkit and the fakes it must never call."""
        fakes = {
            "text_embedder": FakeTextEmbedder(),
            "aligner": FakeAligner(),
            "siglip": FakeSiglip(),
            "vqa": FakeVideoQA(),
            "chat": FakeChat([]),
        }
        toolkit = ToolKit(
            video=VIDEO,
            temporal=withheld_temporal_memory(make_temporal_memory(5).segments),
            text_embedder=fakes["text_embedder"],
            aligner=fakes["aligner"],
            objects=objects(),
            object_text_embedder=fakes["siglip"],
            object_text_columns=slice(0, 2),
            vqa=fakes["vqa"],
            sub_runner=ReActRunner(fakes["chat"]),
            sub_prompt="{tools} {tool_names} {input} {agent_scratchpad}",
            memory_available=False,
        )
        return toolkit, fakes

    def test_every_tool_of_the_orchestrator_answers_the_same_sentence(self) -> None:
        toolkit, _ = self.withheld()

        assert toolkit.caption_retrieval(0, 2) == MEMORY_UNAVAILABLE
        assert toolkit.segment_localization("pour the milk") == MEMORY_UNAVAILABLE
        assert toolkit.visual_question_answering("is it poured?", 1) == (
            MEMORY_UNAVAILABLE
        )
        assert toolkit.object_memory_querying("how many bowls?").observation == (
            MEMORY_UNAVAILABLE
        )
        assert MEMORY_UNAVAILABLE == (
            "MEMORY UNAVAILABLE: no memory exists for this recording"
        )

    def test_the_tools_of_the_sub_agent_answer_it_too(self) -> None:
        toolkit, _ = self.withheld()

        assert toolkit.database_querying("SELECT * FROM Objects") == MEMORY_UNAVAILABLE
        assert toolkit.open_vocabulary_object_retrieval("a cup") == MEMORY_UNAVAILABLE

    def test_no_model_is_asked_anything(self) -> None:
        toolkit, fakes = self.withheld()

        toolkit.caption_retrieval(0, 2)
        toolkit.segment_localization("pour the milk")
        toolkit.visual_question_answering("is it poured?", 1)
        toolkit.object_memory_querying("how many bowls?")
        toolkit.database_querying("SELECT * FROM Objects")
        toolkit.open_vocabulary_object_retrieval("a cup")

        assert fakes["text_embedder"].texts == []
        assert fakes["aligner"].texts == []
        assert fakes["siglip"].texts == []
        assert fakes["vqa"].questions == [] and fakes["vqa"].clips == []
        # The object sub-agent is never run: no nested episode, no LLM call.
        assert fakes["chat"].prompts == []

    def test_the_object_database_is_never_even_built(self) -> None:
        """An empty SQLite database is still a memory the condition forbids."""
        toolkit, _ = self.withheld()

        assert toolkit.database is None
        assert toolkit.object_memory_querying("how many?").sub_transcript is None

    def test_the_agent_is_still_told_how_long_the_video_is(self) -> None:
        toolkit, _ = self.withheld()

        assert toolkit.n_segments == 5

    def test_the_withheld_grid_is_the_recordings_own(self) -> None:
        segments = get_video_segments(VIDEO)
        temporal = withheld_temporal_memory(segments)

        assert temporal.segments == segments
        assert temporal.captions == [""] * len(segments)
        assert temporal.caption_embeddings.shape == (len(segments), 0)
        assert temporal.visual_embeddings.shape == (len(segments), 0)

    def test_the_stand_in_for_the_vqa_model_refuses_to_answer(self) -> None:
        with pytest.raises(RuntimeError):
            WithheldVideoQA().answer([], [])


class TestBuildToolkitModes:
    """What ``_wiring.build_toolkit`` does with each ``--memory``."""

    @staticmethod
    def args(**overrides):
        import argparse

        values = {
            "memory": "full",
            "caption_variant": "default",
            "no_objects": False,
            "vqa": "qwen",
            "vqa_style": "two_questions",
            "aligner_model": ALIGNER,
            "object_memory": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    class ExplodingModels:
        """``LazyModels`` that fails if a model is so much as touched."""

        def __getattr__(self, name: str):
            raise AssertionError(f"the withheld condition loaded {name}")

    def test_the_withheld_condition_loads_no_model_and_reads_no_artifact(
        self, wiring, tmp_path: Path
    ) -> None:
        from video_agent.memory.artifacts import VideoArtifacts

        # An artifacts directory with nothing in it: the full condition
        # cannot get past it, the withheld one never looks.
        artifacts = VideoArtifacts(tmp_path, "1_1_360p")
        runner = ReActRunner(FakeChat([]))

        toolkit = wiring.build_toolkit(
            VIDEO, artifacts, self.ExplodingModels(), self.args(memory="empty"), runner
        )

        assert toolkit.memory_available is False
        assert toolkit.n_segments == len(get_video_segments(VIDEO)) == 5
        assert [spec.name for spec in toolkit.main_tools()] == list(MAIN_TOOL_NAMES)
        assert toolkit.caption_retrieval(0, 1) == MEMORY_UNAVAILABLE
        assert list(tmp_path.glob("1_1_360p/*")) == []
        with pytest.raises(FileNotFoundError):
            wiring.build_toolkit(
                VIDEO, artifacts, self.ExplodingModels(), self.args(), runner
            )

    def test_an_ablation_still_drops_the_tools_it_ablates(
        self, wiring, tmp_path: Path
    ) -> None:
        from video_agent.memory.artifacts import VideoArtifacts

        toolkit = wiring.build_toolkit(
            VIDEO,
            VideoArtifacts(tmp_path, "1_1_360p"),
            self.ExplodingModels(),
            self.args(memory="empty", no_objects=True, vqa="none"),
            ReActRunner(FakeChat([])),
        )

        assert [spec.name for spec in toolkit.main_tools()] == [
            "caption_retrieval",
            "segment_localization",
        ]

    def test_a_shuffled_run_needs_the_recordings_position(
        self, wiring, tmp_path: Path
    ) -> None:
        from video_agent.memory.artifacts import VideoArtifacts

        with pytest.raises(ValueError):
            wiring.build_toolkit(
                VIDEO,
                VideoArtifacts(tmp_path, "1_1_360p"),
                self.ExplodingModels(),
                self.args(memory="shuffled"),
                ReActRunner(FakeChat([])),
                None,
            )

    def test_the_shuffled_toolkit_is_a_view_over_the_files_on_disk(
        self, wiring, tmp_path: Path
    ) -> None:
        """Condition C reads the artifacts of condition A and renames them."""
        from video_agent.memory.artifacts import VideoArtifacts
        from video_agent.memory.store import (
            caption_embeddings_file,
            captions_file,
            visual_embeddings_file,
        )

        artifacts = VideoArtifacts(tmp_path, "1_1_360p")
        captions = [f"caption {index}" for index in range(5)]
        artifacts.save_json(captions_file("default"), captions)
        rows = np.eye(5, dtype=np.float32)
        artifacts.save_array(caption_embeddings_file("default"), rows)
        artifacts.save_array(visual_embeddings_file(aligner_slug(ALIGNER)), rows)
        artifacts.save_pickle(object_memory_file(None), objects())
        artifacts.record_stage(
            "embed", artifact="x", params={"widths": [2, 1]}
        )

        class Models:
            text_embedder = FakeTextEmbedder()
            aligner = FakeAligner()
            siglip = FakeSiglip()
            vqa = None

        toolkit = wiring.build_toolkit(
            VIDEO,
            artifacts,
            Models(),
            self.args(memory="shuffled"),
            ReActRunner(FakeChat([])),
            3,
        )
        permutation = segment_permutation(5, ordinal=3)

        assert toolkit.permutation == permutation
        assert sorted(toolkit.temporal.captions) == sorted(captions)
        assert toolkit.temporal.captions == [
            captions[old] for old in permutation.perm
        ]
        assert artifacts.load_json(captions_file("default")) == captions  # untouched
        inverse = permutation.inverse()
        assert toolkit.objects.records[0].segment_indices == tuple(
            sorted(inverse[index] for index in (0, 1, 4))
        )
        # The grid the clips are still cut on is the video's own, in time order.
        assert toolkit.video_segments == get_video_segments(VIDEO)


class TestBatchRunner:
    """``run_task.py`` end to end with the empty and shuffled memories, offline.

    The batch is where the pieces meet: the recording's position in the split
    seeds its permutation, the permutation is written beside the record and
    named in it, and the agent's observations are the ones the condition
    promises. Everything is faked except the script.
    """

    RECORDING = "1_25"
    SPLIT = {"error": ["1_10"], "normal": ["1_25"]}
    """Two ids, and only the second runs: the ordinal is the split's, not the
    selection's (``--subset normal`` below leaves the first one out)."""

    SCRIPT = [
        "Thought: I should read the captions\n"
        "Action: caption_retrieval\n"
        "Action Input: (0, 2)",
        "Thought: I now know the final answer\n"
        'Final Answer: ```json\n{"steps": [{"step_id": 1, "executed": "unsure", '
        '"segments": null, "order": "unsure"}]}\n```',
    ]

    @pytest.fixture
    def dataset(self, tmp_path: Path) -> dict:
        """A split of two recordings, one of which has artifacts on disk."""
        from video_agent.memory.artifacts import VideoArtifacts
        from video_agent.memory.store import (
            VIDEO_META_FILE,
            caption_embeddings_file,
            captions_file,
            visual_embeddings_file,
        )

        if not HAVE_ANNOTATIONS:
            pytest.skip("CaptainCook4D annotations not found")
        split = tmp_path / "split.json"
        split.write_text(json.dumps(self.SPLIT), encoding="utf-8")
        artifacts = VideoArtifacts(tmp_path / "preprocess", f"{self.RECORDING}_360p")
        artifacts.save_json(
            VIDEO_META_FILE,
            {"path": "nowhere.mp4", "fps": 30.0, "frame_count": 300},
        )
        captions = [f"caption {index}" for index in range(5)]
        artifacts.save_json(captions_file("default"), captions)
        rows = np.eye(5, dtype=np.float32)
        artifacts.save_array(caption_embeddings_file("default"), rows)
        artifacts.save_array(visual_embeddings_file(aligner_slug(ALIGNER)), rows)
        return {
            "captions": captions,
            "split": split,
            "annotations": ANNOTATIONS_DIR,
            "artifacts_root": tmp_path / "preprocess",
            "out": tmp_path / "runs",
        }

    def run(self, run_task_module, monkeypatch, dataset: dict, *extra: str) -> dict:
        """Run the batch over the one recording and return its record."""

        class Models:
            def __init__(self, args) -> None:
                self.chat = FakeChat(list(TestBatchRunner.SCRIPT))
                self.text_embedder = FakeTextEmbedder()
                self.aligner = FakeAligner()
                self.siglip = FakeSiglip()
                self.vqa = None

        monkeypatch.setattr(run_task_module, "LazyModels", Models)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_task.py",
                "--split", str(dataset["split"]),
                "--annotations-dir", str(dataset["annotations"]),
                "--artifacts-root", str(dataset["artifacts_root"]),
                "--videos-dir", str(dataset["artifacts_root"]),
                "--out", str(dataset["out"]),
                "--run-name", "control",
                "--subset", "normal",
                "--no-objects",
                "--vqa", "none",
                "--max-iterations", "4",
                *extra,
            ],
        )

        run_task_module.main()

        from video_agent.task.run import RunStore

        return RunStore(dataset["out"], "control")

    def test_the_withheld_condition_answers_without_the_artifacts(
        self, run_task_module, monkeypatch, dataset: dict, tmp_path: Path
    ) -> None:
        from video_agent.memory.artifacts import VideoArtifacts
        from video_agent.memory.store import VIDEO_META_FILE

        # A directory holding the video's metadata and no memory at all: the
        # grid the prompt quotes comes from the video, and nothing else is
        # read. (A full run over the same directory would stop at the
        # missing captions — ``TestBuildToolkitModes`` above.)
        bare = tmp_path / "bare"
        VideoArtifacts(bare, f"{self.RECORDING}_360p").save_json(
            VIDEO_META_FILE, {"path": "nowhere.mp4", "fps": 30.0, "frame_count": 300}
        )
        store = self.run(
            run_task_module,
            monkeypatch,
            {**dataset, "artifacts_root": bare},
            "--memory", "empty",
        )
        record = store.load(self.RECORDING)

        assert [path.name for path in (bare / f"{self.RECORDING}_360p").iterdir()] == [
            VIDEO_META_FILE
        ]
        assert record["status"] == "ok"
        assert record["config"]["memory"] == "empty"
        assert record["n_segments"] == 5  # the video's own grid, as the prompt says
        assert f"{record['n_segments']} segments" in record["question"]
        observations = [
            step["action"]["observation"]
            for step in record["transcript"]["steps"]
            if step["action"] is not None
        ]
        assert observations == [MEMORY_UNAVAILABLE]
        assert not store.has_permutation(self.RECORDING)

    def test_the_shuffled_condition_writes_the_permutation_it_ran_on(
        self, run_task_module, monkeypatch, dataset: dict
    ) -> None:
        store = self.run(
            run_task_module, monkeypatch, dataset, "--memory", "shuffled"
        )
        record = store.load(self.RECORDING)
        permutation = store.load_permutation(self.RECORDING)

        # "1_25" is the second of the two sorted ids of the split.
        assert permutation.ordinal == 1
        assert permutation.seed == PERMUTATION_BASE_SEED + 1
        assert permutation == segment_permutation(5, ordinal=1)
        assert record["config"]["memory"] == "shuffled"
        assert record["config"]["memory_seed"] == PERMUTATION_BASE_SEED
        assert record["config"]["memory_permutation_sha256"] == permutation.sha256()
        observation = record["transcript"]["steps"][0]["action"]["observation"]
        assert observation.endswith(
            str(
                {
                    index: dataset["captions"][permutation.perm[index]]
                    for index in (0, 1, 2)
                }
            )
        )

    def test_the_full_condition_is_the_run_it_always_was(
        self, run_task_module, monkeypatch, dataset: dict
    ) -> None:
        store = self.run(run_task_module, monkeypatch, dataset)
        record = store.load(self.RECORDING)

        assert "memory" not in record["config"]
        assert not store.has_permutation(self.RECORDING)
        assert record["transcript"]["steps"][0]["action"]["observation"].endswith(
            str({index: dataset["captions"][index] for index in (0, 1, 2)})
        )


class TestSyntheticCaseG:
    """Case (g) of the synthetic harness, over the harness's own memory."""

    def test_the_shuffled_harness_memory_says_the_same_things_in_another_order(
        self,
    ) -> None:
        from video_agent.eval.synthetic import (
            HashingTextEmbedder,
            synthetic_temporal_memory,
            synthetic_video,
        )
        from video_agent.task.annotations import AnnotationSet

        annotations = load_annotations()
        recording = annotations.recordings["1_25"]
        sop, _ = annotations.for_recording("1_25")
        embedder = HashingTextEmbedder(64)
        video = synthetic_video(recording)
        original = synthetic_temporal_memory(
            recording, sop, video=video, text_embedder=embedder, aligner=embedder
        )
        permutation = segment_permutation(len(original.segments), ordinal=0)

        shuffled = shuffle_temporal_memory(original, permutation)

        assert sorted(shuffled.captions) == sorted(original.captions)
        assert shuffled.captions != original.captions
        assert shuffled.caption_embeddings.shape == original.caption_embeddings.shape
        for new, old in enumerate(permutation.perm):
            assert shuffled.captions[new] == original.captions[old]

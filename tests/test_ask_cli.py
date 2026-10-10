"""The inference CLI: argument defaults and the wiring seams it exposes.

``scripts/`` is not a package, so the module is loaded from its path; the
directory goes on ``sys.path`` first because the script imports its sibling
``_env`` the way it does when run directly.
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from video_agent.media import Video
from video_agent.memory.artifacts import VideoArtifacts

from tests._paths import SCRIPTS


@pytest.fixture(scope="module")
def ask_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("ask_cli", SCRIPTS / "ask.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTextEmbedder:
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


class FakeAligner:
    def encode_video(self, clips: Sequence[object]) -> np.ndarray:
        return np.ones((len(clips), 3), dtype=np.float32)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


class FakeModels:
    """Stands in for ``LazyModels`` without loading a checkpoint."""

    def __init__(self) -> None:
        self.text_embedder = FakeTextEmbedder()
        self.aligner = FakeAligner()
        self.siglip = None
        self.vqa = None


class StubToolKit:
    def __init__(self) -> None:
        self.n_segments = 3

    def main_tools(self) -> list[str]:
        return ["caption_retrieval", "segment_localization"]


class StubRunner:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def run(self, question: str, tools, *, prompt_template: str):
        self.calls.append((question, list(tools), prompt_template))
        return self.result


def test_defaults_are_the_published_configuration(ask_module) -> None:
    args = ask_module.parse_args(["clip.mp4", "why?", "--artifacts-root", "preprocess"])

    assert args.video == Path("clip.mp4") and args.question == "why?"
    assert args.artifacts_root == Path("preprocess")
    assert args.caption_variant == "default"
    assert args.model == "Qwen/Qwen3.5-9B"
    assert args.chat == "vllm"
    assert args.base_url is None  # filled in by the backend, see _wiring
    assert args.vqa == "api" and args.device == "cpu"
    assert args.vqa_style == "two_questions"
    assert args.max_iterations == 15
    assert args.temperature == 0.0
    assert args.seed is None
    assert args.out is None
    assert not args.multiple_choice and not args.no_objects and not args.print_log


def test_parses_the_ablation_and_backend_switches(ask_module) -> None:
    args = ask_module.parse_args(
        [
            "clip.mp4", "why?",
            "--artifacts-root", "runs",
            "--caption-variant", "state",
            "--multiple-choice",
            "--chat", "transformers",
            "--model", "org/local-model",
            "--vqa", "none",
            "--vqa-style", "single",
            "--no-objects",
            "--max-iterations", "3",
            "--temperature", "0.7",
            "--seed", "17",
            "--out", "runs/t.json",
            "--print-log",
        ]
    )

    assert args.chat == "transformers" and args.vqa == "none"
    assert args.vqa_style == "single"
    assert args.multiple_choice and args.no_objects and args.print_log
    assert (args.max_iterations, args.temperature, args.seed) == (3, 0.7, 17)
    assert args.out == Path("runs/t.json")


def test_the_local_backend_is_selectable_and_resolves_its_endpoint(ask_module) -> None:
    from _wiring import resolve_base_url

    args = ask_module.parse_args(
        ["clip.mp4", "why?", "--artifacts-root", "runs", "--chat", "vllm"]
    )

    assert args.chat == "vllm" and args.base_url is None
    assert resolve_base_url(args) == "http://localhost:8000/v1"


def test_an_explicit_endpoint_overrides_the_backend_default(ask_module) -> None:
    from _wiring import resolve_base_url

    args = ask_module.parse_args(
        [
            "clip.mp4", "why?",
            "--artifacts-root", "runs",
            "--chat", "vllm",
            "--base-url", "http://gpu-server:8000/v1",
        ]
    )

    assert resolve_base_url(args) == "http://gpu-server:8000/v1"


def test_rejects_an_unknown_backend(ask_module) -> None:
    with pytest.raises(SystemExit):
        ask_module.parse_args(
            ["clip.mp4", "why?", "--artifacts-root", "runs", "--vqa", "videollava"]
        )


def test_loads_the_dotenv_file_before_talking_to_a_provider(ask_module) -> None:
    import _env

    assert ask_module.load_dotenv is _env.load_dotenv
    assert "load_dotenv()" in inspect.getsource(ask_module.main)


def test_ask_runs_the_main_tools_with_the_prompt_template(ask_module) -> None:
    runner = StubRunner(result="agent-result")
    toolkit = StubToolKit()

    result = ask_module.ask("why?", toolkit, runner, "TEMPLATE {tools}")

    assert result == "agent-result"
    assert runner.calls == [
        ("why?", ["caption_retrieval", "segment_localization"], "TEMPLATE {tools}")
    ]


def test_api_vqa_gets_the_reference_prompt_only_in_single_style(ask_module) -> None:
    from video_agent.adapters.vqa_openai import GPT4V_PROMPT

    class ChatlessModels(ask_module.LazyModels):
        """``LazyModels`` whose chat backend needs no API key."""

        @property
        def chat(self) -> object:
            return object()

    def vqa_for(style: str):
        args = ask_module.parse_args(
            [
                "clip.mp4", "why?",
                "--artifacts-root", "runs",
                "--vqa", "api",
                "--vqa-style", style,
            ]
        )
        return ChatlessModels(args).vqa

    assert vqa_for("single").question_template == GPT4V_PROMPT
    assert vqa_for("two_questions").question_template is None


def test_the_vqa_style_reaches_the_tool_configuration(ask_module) -> None:
    pytest.importorskip("video_agent.agent.config")

    args = ask_module.parse_args(
        ["clip.mp4", "why?", "--artifacts-root", "runs", "--vqa-style", "single"]
    )

    assert ask_module.tool_config(args).vqa_style == "single"


def write_memory(artifacts: VideoArtifacts, visual_name: str) -> None:
    """The three artifacts one video's temporal memory is assembled from.

    ``visual_name`` is the aligner's own embeddings file, which is the whole
    point of these tests: a recording can hold one file per aligner.
    """
    artifacts.save_json("captions.default.json", ["a", "b", "c"])
    artifacts.save_array("caption_embeddings.default.npy", np.eye(3, dtype=np.float32))
    artifacts.save_array(visual_name, np.eye(3, dtype=np.float32))


def toolkit_args(ask_module, root: Path, *extra: str):
    """Flags of an ablated run against ``root``: no objects, no VQA model."""
    return ask_module.parse_args(
        [
            "clip.mp4", "why?",
            "--artifacts-root", str(root),
            "--no-objects",
            "--vqa", "none",
            *extra,
        ]
    )


def test_build_toolkit_wires_the_temporal_memory_and_ablations(
    ask_module, tmp_path: Path
) -> None:
    pytest.importorskip("video_agent.agent.tools")

    artifacts = VideoArtifacts(tmp_path, "clip")
    write_memory(artifacts, "segment_visual_embeddings.pe-core-l.npy")
    video = Video(path=tmp_path / "clip.mp4", fps=30.0, frame_count=180)
    args = toolkit_args(ask_module, tmp_path)

    toolkit = ask_module.build_toolkit(
        video, artifacts, FakeModels(), args, StubRunner(None)
    )

    assert toolkit.n_segments == 3
    # Ablations drop their tools: no VQA model, no object memory.
    assert [tool.name for tool in toolkit.main_tools()] == [
        "caption_retrieval",
        "segment_localization",
    ]
    toolkit.close()


def test_pe_core_l_is_the_default_aligner_and_reaches_the_configuration(
    ask_module,
) -> None:
    from video_agent.agent.config import DEFAULT_ALIGNER_MODEL

    args = ask_module.parse_args(["clip.mp4", "why?", "--artifacts-root", "runs"])

    assert args.aligner_model == DEFAULT_ALIGNER_MODEL == "timm/PE-Core-L-14-336"
    assert ask_module.tool_config(args).aligner_model == DEFAULT_ALIGNER_MODEL


def test_the_xclip_aligner_reads_the_unsuffixed_file(
    ask_module, tmp_path: Path
) -> None:
    """``--aligner-model`` picks the memory, not only the text tower.

    X-CLIP's embedding files carry no suffix, so asking for X-CLIP has to
    land on ``segment_visual_embeddings.npy`` and not on a name no recording
    has.
    """
    pytest.importorskip("video_agent.agent.tools")

    artifacts = VideoArtifacts(tmp_path, "clip")
    write_memory(artifacts, "segment_visual_embeddings.npy")
    video = Video(path=tmp_path / "clip.mp4", fps=30.0, frame_count=180)
    args = toolkit_args(
        ask_module, tmp_path, "--aligner-model", "microsoft/xclip-base-patch16"
    )

    with ask_module.build_toolkit(
        video, artifacts, FakeModels(), args, StubRunner(None)
    ) as toolkit:
        assert toolkit.n_segments == 3


def test_another_aligners_embeddings_are_never_read_in_place_of_the_missing_ones(
    ask_module, tmp_path: Path
) -> None:
    """The two spaces are not interchangeable, so this must fail loudly."""
    pytest.importorskip("video_agent.agent.tools")

    artifacts = VideoArtifacts(tmp_path, "clip")
    write_memory(artifacts, "segment_visual_embeddings.npy")  # X-CLIP's only
    video = Video(path=tmp_path / "clip.mp4", fps=30.0, frame_count=180)

    with pytest.raises(FileNotFoundError) as error:
        ask_module.build_toolkit(
            video, artifacts, FakeModels(), toolkit_args(ask_module, tmp_path),
            StubRunner(None),
        )

    message = str(error.value)
    assert "segment_visual_embeddings.pe-core-l.npy" in message
    assert "clip" in message  # which recording is missing it

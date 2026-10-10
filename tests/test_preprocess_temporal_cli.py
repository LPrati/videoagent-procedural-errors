"""The preprocessing CLI: which instruction, which artifact, which captioner.

``scripts/`` is not a package, so the module is loaded from its path; the
directory goes on ``sys.path`` first because the script imports its siblings
``_env`` and ``_wiring`` the way it does when run directly. ``parse_args``
reads ``sys.argv``, so each case sets it.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from tests.fakes import FakeAligner
from video_agent.adapters.aligner import DEFAULT_ALIGNER
from video_agent.adapters.captioner import (
    CAPTION_INSTRUCTIONS,
    DEFAULT_PROMPT,
    load_prompt_file,
)
from video_agent.agent.config import DEFAULT_ALIGNER_MODEL
from video_agent.media import Video
from video_agent.memory.artifacts import VideoArtifacts
from video_agent.memory.preprocess import get_video_segments

from tests._paths import REPO_ROOT, SCRIPTS

BASE_ARGV = ["preprocess_temporal.py", "--out", "preprocess"]


@pytest.fixture(scope="module")
def preprocess_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "preprocess_temporal_cli", SCRIPTS / "preprocess_temporal.py"
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


class TestCaptionInstructions:
    def test_the_three_built_in_instructions_are_named(self) -> None:
        assert sorted(CAPTION_INSTRUCTIONS) == ["action", "order", "state"]

    def test_the_action_instruction_is_the_adapter_default(self) -> None:
        assert CAPTION_INSTRUCTIONS["action"] == DEFAULT_PROMPT


class TestPromptSelection:
    def test_no_flag_at_all_keeps_the_default_instruction(self, parse, preprocess_module) -> None:
        args = parse()

        assert args.prompt_name is None and args.prompt_text is None
        assert args.variant_name == "default"
        assert preprocess_module.resolve_prompt(args) == DEFAULT_PROMPT

    def test_a_named_instruction_resolves_to_its_text(self, parse, preprocess_module) -> None:
        args = parse("--prompt-name", "state")

        assert preprocess_module.resolve_prompt(args) == CAPTION_INSTRUCTIONS["state"]

    def test_a_named_instruction_names_the_artifact_too(self, parse) -> None:
        assert parse("--prompt-name", "state").variant_name == "state"

    def test_an_explicit_variant_name_wins(self, parse) -> None:
        args = parse("--prompt-name", "state", "--variant-name", "state-f4")

        assert args.variant_name == "state-f4"

    def test_a_free_prompt_does_not_rename_the_artifact(self, parse, preprocess_module) -> None:
        args = parse("--prompt-text", "Describe the pan.")

        assert args.variant_name == "default"
        assert preprocess_module.resolve_prompt(args) == "Describe the pan."

    def test_naming_and_writing_an_instruction_at_once_is_a_contract_violation(
        self, parse
    ) -> None:
        with pytest.raises(SystemExit):
            parse("--prompt-name", "state", "--prompt-text", "Describe the pan.")

    def test_an_unknown_instruction_is_rejected(self, parse) -> None:
        with pytest.raises(SystemExit):
            parse("--prompt-name", "objects")


class TestCaptionerBackend:
    def test_the_served_captioner_is_the_default(self, parse) -> None:
        args = parse()

        assert args.captioner == "api"
        assert args.captioner_model == "Qwen/Qwen3.5-9B"

    def test_the_local_captioner_is_selectable(self, parse, preprocess_module) -> None:
        settings = preprocess_module.captioner_settings(
            parse("--captioner", "qwen", "--captioner-model", "Qwen/Qwen3-VL-8B-Instruct")
        )

        assert settings["backend"] == "qwen"
        assert settings["model"] == "Qwen/Qwen3-VL-8B-Instruct"
        assert "chat" not in settings
        assert settings["captioner"] == {
            "prompt": DEFAULT_PROMPT,
            "max_frames_per_clip": None,
            "do_sample": False,
            "top_p": 0.95,
            "temperature": 0.7,
            "num_candidates": 1,
        }

    def test_the_api_captioner_talks_to_the_local_endpoint_without_thinking(
        self, parse, preprocess_module, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VLLM_API_KEY", raising=False)

        settings = preprocess_module.captioner_settings(
            parse(
                "--captioner", "api",
                "--captioner-model", "org/served-model",
                "--prompt-name", "state",
            )
        )

        assert settings["backend"] == "api"
        assert settings["model"] == "org/served-model"
        assert settings["chat"] == {
            "base_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
            "temperature": 0.0,
            "video_fps": 5.0,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "mm_processor_kwargs": {"do_sample_frames": False},
            },
        }
        assert settings["captioner"] == {
            "prompt": CAPTION_INSTRUCTIONS["state"],
            "frames_per_clip": None,
            "max_tokens": 64,
            "workers": 8,
            "transport": "video",
        }

    def test_the_concurrency_and_the_budget_are_flags(
        self, parse, preprocess_module
    ) -> None:
        settings = preprocess_module.captioner_settings(
            parse(
                "--captioner", "api",
                "--workers", "16",
                "--caption-max-tokens", "96",
                "--caption-max-frames", "4",
                "--base-url", "http://gpu-server:8000/v1",
            )
        )

        assert settings["chat"]["base_url"] == "http://gpu-server:8000/v1"
        assert settings["captioner"]["workers"] == 16
        assert settings["captioner"]["max_tokens"] == 96
        assert settings["captioner"]["frames_per_clip"] == 4


class TestCaptionTransport:
    def test_a_clip_travels_as_one_video_by_default(
        self, parse, preprocess_module
    ) -> None:
        settings = preprocess_module.captioner_settings(parse("--captioner", "api"))

        assert settings["captioner"]["transport"] == "video"
        # The clips were decoded at CLIP_SAMPLE_FPS, and that is the rate
        # the MP4 they are written into is stamped with.
        assert settings["chat"]["video_fps"] == preprocess_module.CLIP_SAMPLE_FPS
        assert settings["chat"]["video_fps"] == 5.0
        assert settings["chat"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"do_sample_frames": False},
        }

    def test_the_images_transport_says_nothing_about_frame_sampling(
        self, parse, preprocess_module
    ) -> None:
        settings = preprocess_module.captioner_settings(
            parse("--captioner", "api", "--caption-transport", "images")
        )

        assert settings["captioner"]["transport"] == "images"
        assert settings["chat"]["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_the_local_captioner_takes_no_transport_at_all(
        self, parse, preprocess_module
    ) -> None:
        # It feeds the frames to the processor as a video either way, so the
        # flag is ignored rather than an error — and recorded as None.
        settings = preprocess_module.captioner_settings(
            parse("--captioner", "qwen", "--caption-transport", "images")
        )

        assert "transport" not in settings["captioner"]
        assert "chat" not in settings

    def test_an_unknown_transport_is_rejected(self, parse) -> None:
        with pytest.raises(SystemExit):
            parse("--captioner", "api", "--caption-transport", "frames")


class TestCaptionerSampling:
    def test_candidate_sampling_is_refused_on_the_api_path(self, parse) -> None:
        with pytest.raises(SystemExit):
            parse("--captioner", "api", "--do-sample")
        with pytest.raises(SystemExit):
            parse("--captioner", "api", "--num-candidates", "5")

    def test_candidate_sampling_still_works_locally(
        self, parse, preprocess_module
    ) -> None:
        settings = preprocess_module.captioner_settings(
            parse("--captioner", "qwen", "--do-sample", "--num-candidates", "5")
        )

        assert settings["captioner"]["do_sample"] is True
        assert settings["captioner"]["num_candidates"] == 5



class TestAlignerSelection:
    """Which aligner the visual_embed stage runs, and what it writes."""

    def test_pe_core_l_is_the_default(self, parse) -> None:
        # One constant decides the project's aligner, and the stage that
        # writes the embeddings has to agree with the agent that reads them.
        args = parse()

        assert args.aligner_model == DEFAULT_ALIGNER_MODEL
        assert args.aligner_slug == "pe-core-l"

    def test_the_xclip_aligner_keeps_the_unsuffixed_artifact(
        self, parse
    ) -> None:
        args = parse("--aligner-model", DEFAULT_ALIGNER)

        assert args.aligner_slug == "xclip-base-patch16"

    def test_a_candidate_resolves_to_its_artifact_suffix(self, parse) -> None:
        assert parse("--aligner-model", "timm/PE-Core-L-14-336").aligner_slug == (
            "pe-core-l"
        )
        assert parse("--aligner-model", "OpenGVLab/ViCLIP-L-14-hf").aligner_slug == (
            "viclip-l-14"
        )

    def test_metas_own_pe_core_id_is_accepted(self, parse) -> None:
        assert parse("--aligner-model", "facebook/PE-Core-L14-336").aligner_slug == (
            "pe-core-l"
        )

    def test_an_unknown_aligner_is_refused_before_any_video_is_read(
        self, parse
    ) -> None:
        with pytest.raises(SystemExit):
            parse("--aligner-model", "microsoft/xclip-base-patch32")


class FakeModels:
    """``LazyModels`` reduced to the one model the visual stage asks for."""

    def __init__(self, aligner) -> None:
        self.aligner = aligner


def visual_args(**overrides) -> argparse.Namespace:
    values = {
        "aligner_model": DEFAULT_ALIGNER,
        "aligner_slug": "xclip-base-patch16",
        "clips_per_batch": 2,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestVisualEmbedStage:
    """What one aligner leaves on disk, run end to end on the synthetic video."""

    def run(self, preprocess_module, video: Video, root: Path, **overrides):
        artifacts = VideoArtifacts(root, video.path.stem)
        shape = preprocess_module.stage_visual_embed(
            video,
            get_video_segments(video),
            artifacts,
            FakeModels(FakeAligner(dim=4)),
            visual_args(**overrides),
            "abc1234",
        )
        return artifacts, shape

    def test_the_default_aligner_writes_the_file_every_recording_has(
        self, preprocess_module, synthetic_video: Video, tmp_path: Path
    ) -> None:
        artifacts, shape = self.run(preprocess_module, synthetic_video, tmp_path)

        assert shape == (2, 4)  # 150 frames at 30 fps is two 2-second segments
        assert artifacts.has("segment_visual_embeddings.npy")
        assert artifacts.stage_meta("visual_embed")["model"] == DEFAULT_ALIGNER

    def test_a_candidate_writes_its_own_file_and_its_own_lineage(
        self, preprocess_module, synthetic_video: Video, tmp_path: Path
    ) -> None:
        artifacts, _ = self.run(
            preprocess_module,
            synthetic_video,
            tmp_path,
            aligner_model="timm/PE-Core-L-14-336",
            aligner_slug="pe-core-l",
        )

        assert artifacts.has("segment_visual_embeddings.pe-core-l.npy")
        # The unsuffixed file stays free for the X-CLIP embeddings.
        assert not artifacts.has("segment_visual_embeddings.npy")
        stage = artifacts.stage_meta("visual_embed.pe-core-l")
        assert stage["model"] == "timm/PE-Core-L-14-336"
        assert stage["artifact"] == "segment_visual_embeddings.pe-core-l.npy"

    def test_the_lineage_records_what_the_candidate_consumed_and_produced(
        self, preprocess_module, synthetic_video: Video, tmp_path: Path
    ) -> None:
        artifacts, _ = self.run(
            preprocess_module,
            synthetic_video,
            tmp_path,
            aligner_model="timm/PE-Core-L-14-336",
            aligner_slug="pe-core-l",
        )

        params = artifacts.stage_meta("visual_embed.pe-core-l")["params"]
        assert params["aligner_slug"] == "pe-core-l"
        assert params["embedding_dim"] == 4
        # ``FakeAligner`` declares no frame count, and an unknown one is
        # recorded as unknown rather than guessed.
        assert params["frames_per_clip"] is None
        assert params["seconds_per_segment"] == 2.0

    def test_two_candidates_coexist_in_one_directory(
        self, preprocess_module, synthetic_video: Video, tmp_path: Path
    ) -> None:
        self.run(preprocess_module, synthetic_video, tmp_path)
        artifacts, _ = self.run(
            preprocess_module,
            synthetic_video,
            tmp_path,
            aligner_model="timm/PE-Core-L-14-336",
            aligner_slug="pe-core-l",
        )

        assert artifacts.has("segment_visual_embeddings.npy")
        assert artifacts.has("segment_visual_embeddings.pe-core-l.npy")
        assert artifacts.stage_meta("visual_embed")["model"] == DEFAULT_ALIGNER
        assert artifacts.stage_meta("visual_embed.pe-core-l")["model"] == (
            "timm/PE-Core-L-14-336"
        )


def test_the_stage_record_names_the_instruction_and_the_backend(
    preprocess_module,
) -> None:
    import inspect

    source = inspect.getsource(preprocess_module.stage_caption)

    for key in (
        "prompt_name",
        "captioner_backend",
        "transport",
        "workers",
        "max_tokens",
        "usage",
    ):
        assert f'"{key}"' in source


def test_the_stage_bills_only_what_it_captioned(preprocess_module) -> None:
    before = {"llm_calls": 4, "tokens_in": 100, "tokens_out": 20, "wall_s": 0.0}
    after = {"llm_calls": 11, "tokens_in": 275, "tokens_out": 55, "wall_s": 0.0}

    assert preprocess_module.usage_delta(before, after) == {
        "llm_calls": 7,
        "tokens_in": 175,
        "tokens_out": 35,
        "wall_s": 0.0,
    }


CAPTION_PROMPTS = REPO_ROOT / "prompts" / "captions"
"""The published caption instruction files read through ``--prompt-file``."""


class TestPromptFile:
    """``--prompt-file``: one block per clip position, read before any video."""

    def test_the_abstention_instruction_has_only_a_middle_block(self) -> None:
        blocks = load_prompt_file(CAPTION_PROMPTS / "abstention.md")

        assert sorted(blocks) == ["middle"]
        assert blocks["middle"].startswith("This is a 2-second clip")
        assert blocks["middle"].endswith("answer exactly: No action visible.")

    def test_the_context_instruction_has_the_three_positions(self) -> None:
        blocks = load_prompt_file(CAPTION_PROMPTS / "abstention_context.md")

        assert sorted(blocks) == ["first", "last", "middle"]
        assert "between second 2 and second 4" in blocks["middle"]
        assert "first two seconds of the clip" in blocks["first"]
        assert "last two seconds of the clip" in blocks["last"]

    def test_a_block_reaches_the_model_as_one_line(self) -> None:
        blocks = load_prompt_file(CAPTION_PROMPTS / "abstention_context.md")

        for text in blocks.values():
            assert "\n" not in text
            assert text == text.strip()
        # The prose above the first block is documentation and never reaches
        # the model.
        assert all("documentation" not in text for text in blocks.values())

    def test_a_file_without_a_middle_block_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "only_first.md"
        path.write_text("# doc\n\n## user [first]\n\nDescribe it.\n", encoding="utf-8")

        with pytest.raises(ValueError, match="middle"):
            load_prompt_file(path)

    def test_an_unknown_position_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "odd.md"
        path.write_text("## user [second]\n\nDescribe it.\n", encoding="utf-8")

        with pytest.raises(ValueError, match="second"):
            load_prompt_file(path)

    def test_an_untagged_block_is_the_middle_one(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.md"
        path.write_text(
            "# doc\n\n## user\n\nDescribe the\nclip.\n\n## notes\n\nignored\n",
            encoding="utf-8",
        )

        assert load_prompt_file(path) == {"middle": "Describe the clip."}

    def test_the_file_is_read_when_the_flags_are_parsed(self, parse, tmp_path) -> None:
        missing = tmp_path / "absent.md"

        with pytest.raises(SystemExit):
            parse("--prompt-file", str(missing))

    def test_a_prompt_file_and_a_named_instruction_are_exclusive(self, parse) -> None:
        with pytest.raises(SystemExit):
            parse(
                "--prompt-file", str(CAPTION_PROMPTS / "abstention.md"),
                "--prompt-name", "state",
            )


class TestContextSegments:
    """``--context-segments``: the widened clip and the prompt of each position."""

    def test_the_middle_block_is_the_prompt_the_run_records(
        self, parse, preprocess_module
    ) -> None:
        args = parse(
            "--captioner", "api",
            "--prompt-file", str(CAPTION_PROMPTS / "abstention_context.md"),
            "--context-segments", "1",
            "--variant-name", "abst_ctx",
        )
        settings = preprocess_module.captioner_settings(args)
        blocks = load_prompt_file(CAPTION_PROMPTS / "abstention_context.md")

        assert settings["context_segments"] == 1
        assert settings["captioner"]["prompt"] == blocks["middle"]
        assert preprocess_module.resolve_prompt(args) == blocks["middle"]
        assert preprocess_module.prompt_blocks(args) == blocks

    def test_no_context_is_the_default_and_records_as_zero(
        self, parse, preprocess_module
    ) -> None:
        settings = preprocess_module.captioner_settings(parse())

        assert settings["context_segments"] == 0

    def test_a_missing_position_falls_back_to_the_middle_one(
        self, parse, preprocess_module
    ) -> None:
        # ``abstention.md`` has no first/last block: with a window it would
        # caption the two ends with the middle instruction, and the run says so.
        args = parse(
            "--captioner", "api",
            "--prompt-file", str(CAPTION_PROMPTS / "abstention.md"),
            "--context-segments", "1",
        )
        blocks = preprocess_module.prompt_blocks(args)

        assert blocks["first"] == blocks["last"] == blocks["middle"]
        assert preprocess_module.missing_positions(args) == ["first", "last"]

    def test_nothing_is_missing_without_a_window(
        self, parse, preprocess_module
    ) -> None:
        args = parse(
            "--captioner", "api",
            "--prompt-file", str(CAPTION_PROMPTS / "abstention.md"),
        )

        assert preprocess_module.missing_positions(args) == []

    def test_a_frame_budget_would_undo_the_context(self, parse) -> None:
        with pytest.raises(SystemExit):
            parse("--captioner", "api", "--context-segments", "1",
                  "--caption-max-frames", "4")

    def test_a_negative_window_is_refused(self, parse) -> None:
        with pytest.raises(SystemExit):
            parse("--captioner", "api", "--context-segments", "-1")

    def test_the_stage_record_names_the_window_and_the_file(
        self, preprocess_module
    ) -> None:
        import inspect

        source = inspect.getsource(preprocess_module.stage_caption)

        for key in ("context_segments", "prompt_file", "prompts_by_position"):
            assert f'"{key}"' in source

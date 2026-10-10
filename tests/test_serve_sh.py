"""What ``serving/vllm/serve.sh`` resolves to, read off its ``--dry-run``.

vLLM is not needed here: ``--dry-run`` prints the command and exits without
exec'ing it, which makes the resolution a pure function of the environment.
The parts worth pinning are the defaults (the configuration of the agent
runs), the single-call configuration, and the optional quantization flag.
"""

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

SERVING = Path(__file__).resolve().parents[1] / "serving" / "vllm"
SERVE_SH = SERVING / "serve.sh"
SERVE_SINGLE_CALL_SH = SERVING / "serve_single_call.sh"

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="serve.sh is a bash script"
)


def dry_run(script: Path = SERVE_SH, **overrides: str) -> str:
    """Stdout of ``serve.sh --dry-run`` under a controlled environment.

    Every ``VLLM_*`` variable of the caller's environment is dropped first,
    so a developer who exports one does not change what these tests read;
    only the overrides are set.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    env.update(overrides)
    completed = subprocess.run(
        ["bash", str(script), "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def command(script: Path = SERVE_SH, **overrides: str) -> list[str]:
    """The argv of the printed command, unquoted back into words."""
    printed = [
        line
        for line in dry_run(script, **overrides).splitlines()
        if line.startswith("+ ")
    ]
    assert len(printed) == 1, printed
    return shlex.split(printed[0][2:])


def flag(argv: list[str], name: str) -> str | None:
    """Value that follows ``name`` in ``argv``, or ``None`` if it is absent."""
    return argv[argv.index(name) + 1] if name in argv else None


class TestTheDefaults:
    def test_the_agent_configuration_is_the_default(self) -> None:
        argv = command()

        assert flag(argv, "--model") is None  # the id is positional
        assert DEFAULT_MODEL in argv
        assert flag(argv, "--max-model-len") == "131072"
        assert flag(argv, "--gpu-memory-utilization") == "0.92"
        assert flag(argv, "--max-num-seqs") == "16"
        assert flag(argv, "--limit-mm-per-prompt") == '{"image": 16, "video": 1}'

    def test_a_json_override_is_passed_as_given(self) -> None:
        argv = command(VLLM_LIMIT_MM='{"image": 2, "video": 1}')

        assert flag(argv, "--limit-mm-per-prompt") == '{"image": 2, "video": 1}'


class TestTheSingleCallConfiguration:
    def test_it_serves_one_long_video_per_request(self) -> None:
        argv = command(SERVE_SINGLE_CALL_SH)

        assert DEFAULT_MODEL in argv
        assert flag(argv, "--max-model-len") == "77824"
        assert flag(argv, "--gpu-memory-utilization") == "0.90"
        assert flag(argv, "--max-num-seqs") == "2"
        assert flag(argv, "--limit-mm-per-prompt") == '{"image": 0, "video": 1}'
        assert flag(argv, "--media-io-kwargs") == (
            '{"video": {"num_frames": -1, "fps": 4, "max_frames": 8192}}'
        )
        assert flag(argv, "--mm-processor-kwargs") == (
            '{"size":{"longest_edge":131072000,"shortest_edge":4096}}'
        )


class TestQuantization:
    def test_the_default_checkpoint_is_served_unquantized(self) -> None:
        argv = command()

        assert "--quantization" not in argv
        # Served under its own Hugging Face id, so a script points --model at
        # the same string the model card uses: once as the checkpoint to
        # load, once as the name the endpoint answers to.
        assert argv.count(DEFAULT_MODEL) == 2
        assert "quantization: none" in dry_run()

    def test_an_explicit_method_is_passed_through(self) -> None:
        argv = command(VLLM_MODEL="org/quantized-model", VLLM_QUANTIZATION="awq")

        assert flag(argv, "--quantization") == "awq"
        assert "quantization: awq" in dry_run(VLLM_QUANTIZATION="awq")


class TestTheMediaLoader:
    def test_every_frame_of_a_video_reaches_the_processor_by_default(self) -> None:
        # A caption clip arrives as a data-URL MP4 of ten frames; a loader
        # that subsampled it would silently undo the sampling the caption
        # stage decoded, before the processor ever sees the clip.
        assert flag(command(), "--media-io-kwargs") == '{"video": {"num_frames": -1}}'

    def test_the_loader_settings_can_be_emptied(self) -> None:
        assert flag(command(VLLM_MEDIA_IO="{}"), "--media-io-kwargs") == "{}"


class TestTheOtherKnobs:
    def test_extra_args_are_appended_verbatim(self) -> None:
        argv = command(VLLM_EXTRA_ARGS="--tensor-parallel-size 2")

        assert flag(argv, "--tensor-parallel-size") == "2"

    def test_the_context_length_is_the_one_that_was_asked_for(self) -> None:
        argv = command(VLLM_MAX_MODEL_LEN="32768")

        assert flag(argv, "--max-model-len") == "32768"


class TestTheFlashInferSampler:
    def test_it_is_off_by_default_and_the_dry_run_says_so(self) -> None:
        # Not a flag but an environment variable, so --dry-run has to print it
        # for the resolution to be readable: without a CUDA toolkit,
        # FlashInfer's JIT-compiled sampler stops the server from starting.
        assert "# env: VLLM_USE_FLASHINFER_SAMPLER=0" in dry_run()

        assert "# env: VLLM_USE_FLASHINFER_SAMPLER=1" in dry_run(
            VLLM_USE_FLASHINFER_SAMPLER="1"
        )

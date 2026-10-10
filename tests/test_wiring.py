"""What a backend flag resolves to: endpoint, credentials, request body.

``scripts/`` is not a package, so ``_wiring`` is loaded from its path. The
helpers under test are pure — they say what *would* be passed to the chat
adapter — so a test can read the request shape of a flag combination with no
server, no key and no SDK.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from tests._paths import SCRIPTS


@pytest.fixture(scope="module")
def wiring():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("wiring", SCRIPTS / "_wiring.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(**overrides) -> argparse.Namespace:
    """The flags the helpers read, with the scripts' own defaults."""
    values = {
        "chat": "openrouter",
        "base_url": None,
        "model": "openai/gpt-4o-2024-05-13",
        "temperature": 0.0,
        "seed": None,
        "thinking": False,
        "chat_timeout": None,
        "max_iterations": 15,
        "max_tokens": 16384,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class TestResolveBaseUrl:
    def test_openrouter_is_the_hosted_provider(self, wiring) -> None:
        assert wiring.resolve_base_url(args()) == "https://openrouter.ai/api/v1"

    def test_vllm_is_the_local_server(self, wiring) -> None:
        assert (
            wiring.resolve_base_url(args(chat="vllm")) == "http://localhost:8000/v1"
        )

    def test_transformers_has_no_endpoint_at_all(self, wiring) -> None:
        assert wiring.resolve_base_url(args(chat="transformers")) is None

    def test_an_explicit_url_wins_over_the_backend_default(self, wiring) -> None:
        explicit = "http://gpu-server:8000/v1"

        assert wiring.resolve_base_url(args(chat="vllm", base_url=explicit)) == explicit
        assert (
            wiring.resolve_base_url(args(chat="openrouter", base_url=explicit))
            == explicit
        )


class TestLocalApiKey:
    def test_the_environment_variable_is_used_when_set(
        self, wiring, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_API_KEY", "s3cret")

        assert wiring.local_api_key() == "s3cret"

    def test_an_unset_variable_falls_back_to_the_placeholder_token(
        self, wiring, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VLLM_API_KEY", raising=False)

        assert wiring.local_api_key() == "EMPTY"

    def test_an_empty_variable_counts_as_unset(
        self, wiring, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_API_KEY", "")

        assert wiring.local_api_key() == "EMPTY"


class TestRequestBody:
    """What ``extra_body`` a pair of switches assembles."""

    def test_thinking_is_off_and_nothing_else_is_said_by_default(
        self, wiring
    ) -> None:
        assert wiring.request_body() == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_a_video_request_also_pins_the_frames_it_sent(self, wiring) -> None:
        assert wiring.request_body(video=True) == {
            "chat_template_kwargs": {"enable_thinking": False},
            # Without this the server resamples the clip to 2 fps and reads
            # four of the ten frames the caption stage decoded.
            "mm_processor_kwargs": {"do_sample_frames": False},
        }

    def test_thinking_can_be_left_on(self, wiring) -> None:
        assert wiring.request_body(thinking=True) == {}
        assert wiring.request_body(thinking=True, video=True) == {
            "mm_processor_kwargs": {"do_sample_frames": False}
        }

    def test_every_body_is_a_copy_the_caller_owns(self, wiring) -> None:
        first = wiring.request_body(video=True)
        first["chat_template_kwargs"]["enable_thinking"] = True
        first["mm_processor_kwargs"]["do_sample_frames"] = True

        assert wiring.request_body(video=True) == {
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"do_sample_frames": False},
        }


class TestChatSettings:
    def test_the_openrouter_backend_reads_its_key_from_the_environment(
        self, wiring
    ) -> None:
        settings = wiring.chat_settings(args(seed=17, temperature=0.2))

        assert settings == {
            "model_name": "openai/gpt-4o-2024-05-13",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": None,
            "api_key_env": "OPENROUTER_API_KEY",
            "temperature": 0.2,
            "seed": 17,
            "timeout_s": 120.0,  # the SDK default; only the local server waits longer
            "max_retries": 5,
            "extra_body": None,
        }

    def test_the_orchestrator_never_asks_for_the_video_switch(self, wiring) -> None:
        # It sends text only; the switch belongs to the caption stage, which
        # builds its own body (see preprocess_temporal.captioner_settings).
        settings = wiring.chat_settings(args(chat="vllm"))

        assert "mm_processor_kwargs" not in settings["extra_body"]

    def test_the_local_backend_carries_a_token_and_turns_thinking_off(
        self, wiring, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VLLM_API_KEY", raising=False)

        settings = wiring.chat_settings(
            args(chat="vllm", model="org/served-model")
        )

        assert settings["model_name"] == "org/served-model"
        assert settings["base_url"] == "http://localhost:8000/v1"
        assert settings["api_key"] == "EMPTY"
        assert settings["api_key_env"] == "VLLM_API_KEY"
        assert settings["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_a_served_key_reaches_the_adapter(
        self, wiring, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VLLM_API_KEY", "s3cret")

        assert wiring.chat_settings(args(chat="vllm"))["api_key"] == "s3cret"

    def test_only_the_local_backend_disables_thinking(self, wiring) -> None:
        for backend in ("openrouter", "transformers"):
            assert wiring.chat_settings(args(chat=backend))["extra_body"] is None

    def test_the_thinking_switch_is_a_copy_of_the_shared_default(
        self, wiring
    ) -> None:
        first = wiring.chat_settings(args(chat="vllm"))["extra_body"]
        first["chat_template_kwargs"]["enable_thinking"] = True

        assert wiring.chat_settings(args(chat="vllm"))["extra_body"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }

    def test_the_thinking_flag_reaches_the_local_request_body(self, wiring) -> None:
        # With reasoning on, the switch is simply absent: the chat template
        # thinks by default, and the server's parser splits the block out.
        settings = wiring.chat_settings(args(chat="vllm", thinking=True))

        assert settings["extra_body"] == {}

    def test_a_runaway_turn_against_the_local_server_is_not_retried_five_times(
        self, wiring
    ) -> None:
        # Five retries of a runaway turn would cost six full timeouts.
        settings = wiring.chat_settings(args(chat="vllm"))

        assert settings["timeout_s"] == 300.0
        assert settings["max_retries"] == 1

    def test_an_explicit_timeout_wins_over_the_backend_default(self, wiring) -> None:
        assert wiring.chat_timeout_s(args(chat="vllm", chat_timeout=42.5)) == 42.5
        assert wiring.chat_timeout_s(args(chat_timeout=42.5)) == 42.5

    def test_each_backend_has_its_own_timeout(self, wiring) -> None:
        assert wiring.chat_timeout_s(args(chat="vllm")) == 300.0
        assert wiring.chat_timeout_s(args()) == 120.0




class TestToolConfig:
    """Which memories the tools read, frozen into what ``config_id`` hashes."""

    def tools(self, wiring, **overrides):
        from video_agent.agent.config import DEFAULT_ALIGNER_MODEL

        values = {
            "vqa_style": "two_questions",
            "aligner_model": DEFAULT_ALIGNER_MODEL,
        }
        values.update(overrides)
        return wiring.tool_config(argparse.Namespace(**values))

    def test_the_object_memory_flag_reaches_the_tool_configuration(
        self, wiring
    ) -> None:
        config = self.tools(wiring, object_memory="other-detector")

        assert config.object_memory_slug == "other-detector"
        assert config.to_dict()["object_memory_slug"] == "other-detector"

    def test_a_script_that_does_not_declare_the_flag_reads_the_default_memory(
        self, wiring
    ) -> None:
        # ``None`` is the unsuffixed files of the default detector, which is
        # what a script with no flag is asking for.
        assert self.tools(wiring).object_memory_slug is None
        assert self.tools(wiring, object_memory=None).object_memory_slug is None


class TestObjectMemorySlug:
    """What ``--object-memory`` may say; the value becomes a file name."""

    def test_valid_slugs_are_accepted(self, wiring) -> None:
        wiring.check_object_memory_slug(None)
        wiring.check_object_memory_slug("rtdetr-v2-r50")
        wiring.check_object_memory_slug("other-detector")

    def test_anything_that_could_name_another_file_is_refused(
        self, wiring
    ) -> None:
        for slug in ("", "../secrets", "a/b", "UPPER", ".hidden", "-x", "a b"):
            with pytest.raises(ValueError):
                wiring.check_object_memory_slug(slug)


class TestReActConfig:
    """The loop settings a run is frozen with, token cap included."""

    def test_the_per_turn_cap_defaults_to_the_flag_default(self, wiring) -> None:
        # 16k, not 8k: a long but valid final answer can exceed 8k tokens.
        config = wiring.react_config(args())

        assert config.max_tokens == 16384
        assert config.max_iterations == 15

    def test_zero_or_less_means_unlimited(
        self, wiring
    ) -> None:
        assert wiring.react_config(args(max_tokens=0)).max_tokens is None
        assert wiring.react_config(args(max_tokens=-1)).max_tokens is None

    def test_the_cap_is_part_of_what_a_config_id_is_hashed_from(self, wiring) -> None:
        capped = wiring.react_config(args()).to_dict()

        assert capped["max_tokens"] == 16384
        assert capped != wiring.react_config(args(max_tokens=0)).to_dict()

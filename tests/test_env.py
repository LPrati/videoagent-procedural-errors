"""The scripts' ``.env`` loader."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _env import load_dotenv  # noqa: E402  (needs the path above)


@pytest.fixture
def clean_environ(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Swap ``os.environ`` for a plain dict so tests never leak variables."""
    environ: dict[str, str] = {}
    monkeypatch.setattr(os, "environ", environ)
    return environ


def write_env(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text)
    return path


def test_parses_pairs_and_exports_them(clean_environ: dict, tmp_path: Path) -> None:
    path = write_env(tmp_path, "OPENROUTER_API_KEY=sk-abc\nHF_TOKEN=hf_xyz\n")

    values = load_dotenv(path)

    assert values == {"OPENROUTER_API_KEY": "sk-abc", "HF_TOKEN": "hf_xyz"}
    assert clean_environ["OPENROUTER_API_KEY"] == "sk-abc"
    assert clean_environ["HF_TOKEN"] == "hf_xyz"


def test_ignores_comments_and_blank_lines(clean_environ: dict, tmp_path: Path) -> None:
    path = write_env(
        tmp_path,
        "# the orchestrator key\n\n  \nOPENROUTER_API_KEY=sk-abc\n#TRAILING=no\n",
    )

    assert load_dotenv(path) == {"OPENROUTER_API_KEY": "sk-abc"}


def test_strips_export_quotes_and_surrounding_space(
    clean_environ: dict, tmp_path: Path
) -> None:
    path = write_env(
        tmp_path,
        'export QUOTED="sk-abc"\n'
        "SINGLE='sk-def'\n"
        "  SPACED  =  sk-ghi  \n"
        "URL=https://openrouter.ai/api/v1\n",
    )

    assert load_dotenv(path) == {
        "QUOTED": "sk-abc",
        "SINGLE": "sk-def",
        "SPACED": "sk-ghi",
        "URL": "https://openrouter.ai/api/v1",
    }


def test_keeps_equals_signs_inside_the_value(
    clean_environ: dict, tmp_path: Path
) -> None:
    path = write_env(tmp_path, "EXTRA_BODY=a=1,b=2\n")

    assert load_dotenv(path) == {"EXTRA_BODY": "a=1,b=2"}


def test_does_not_override_an_exported_variable(
    clean_environ: dict, tmp_path: Path
) -> None:
    clean_environ["OPENROUTER_API_KEY"] = "sk-from-shell"
    path = write_env(tmp_path, "OPENROUTER_API_KEY=sk-from-file\n")

    values = load_dotenv(path)

    assert values == {"OPENROUTER_API_KEY": "sk-from-file"}  # what the file held
    assert clean_environ["OPENROUTER_API_KEY"] == "sk-from-shell"


def test_skips_lines_without_a_key(clean_environ: dict, tmp_path: Path) -> None:
    path = write_env(tmp_path, "not-a-pair\n=orphan\nKEY=value\n")

    assert load_dotenv(path) == {"KEY": "value"}


def test_returns_nothing_when_the_file_is_missing(
    clean_environ: dict, tmp_path: Path
) -> None:
    write_env(tmp_path, "KEY=value\n")  # a sibling file must not be picked up

    assert load_dotenv(tmp_path / "absent.env") == {}
    assert "KEY" not in clean_environ
